export qr_recursive!, qr_recursive_mixed!
using LinearAlgebra
using CUDA
include("wrappers.jl")
# include("panelmixedprec.jl")

#
# Recursive block QR (Elmroth-Gustavson RGEQR3 formulation).
#
# The factorization is stored in-place: Householder vectors V (unit lower
# trapezoidal, unit diagonal implicit) below the diagonal, and R in the upper
# triangle. Each call returns the upper triangular block reflector factor T,
# such that Q = I - V * T * V'.
#
# Recursion on A = [A1 A2]:
#   1. Factor A1 recursively                         -> V1, R11, T1
#   2. A2 <- Q1' * A2 = (I - V1 T1' V1') A2         -> R12 on top, updated tail
#   3. Factor trailing rows of A2 recursively       -> V2, R22, T2
#   4. T = [ T1  T3 ]   with  T3 = -T1 (V1' V~2) T2
#          [  0  T2 ]
#
# All block updates are performed strictly through the custom mixed-precision
# subroutines `recgemm!` (rectangular GEMM pieces) and `unified_rectrxm!`
# (triangular TRMM pieces on the unit-lower V triangles and the upper T factors).
#

"""
    _transpose_copy(X::AbstractMatrix)

Materializes the transpose of `X` into a contiguous array so that transposed
operands can be routed through `recgemm!`, which operates in 'N','N' form.
"""
function _transpose_copy(X::AbstractMatrix)
    return permutedims(X, (2, 1))
end

"""
    _accum_type(T::DataType)

Accumulation precision used for the block reflector factors `T`:
`Float64` inputs keep `Float64`; everything else (including `Float16`)
accumulates in `Float32`.
"""
_accum_type(::Type{Float64}) = Float64
_accum_type(::Type{T}) where {T} = Float32

"""
    _form_T(Vpanel::AbstractMatrix, tau::AbstractVector)

Forms the upper triangular block reflector factor `T` (LAPACK `larft` recurrence)
from the Householder vectors stored below the diagonal of `Vpanel` and the scalar
factors `tau`, so that `Q = I - V * T * V'`.

The single large inner product `W = V' * V` is computed on-device through
`recgemm!`; the small `k x k` triangular recurrence is then resolved as base-case
work and uploaded back to the device.
"""
function _form_T(Vpanel::AbstractMatrix, tau::AbstractVector)
    m, k = size(Vpanel)
    TT = _accum_type(eltype(Vpanel))

    V = Vpanel .* ((1:m) .> (1:k)') .+ TT.((1:m) .== (1:k)')

    W = fill!(similar(V, TT, k, k), zero(TT))
    recgemm!(1.0f0, _transpose_copy(V), V, 0.0f0, W)

    W_h = Array(W)
    tau_h = Array(TT.(tau))

    T_h = zeros(TT, k, k)
    for j in 1:k
        T_h[j, j] = tau_h[j]
        if j > 1
            T_h[1:j-1, j] .= -tau_h[j] .* (UpperTriangular(T_h[1:j-1, 1:j-1]) * W_h[1:j-1, j])
        end
    end

    T_dev = similar(V, TT, k, k)
    copyto!(T_dev, T_h)
    return T_dev
end

"""
    _qr_base!(A::AbstractMatrix)

Base case of the recursive QR: unblocked Householder factorization via
`CUSOLVER.geqrf!`, followed by formation of the block reflector factor `T`.
`Float16` panels are routed through a `Float32` staging buffer, since CUSOLVER
does not provide a half-precision `geqrf`.
"""
function _qr_base!(A::AbstractMatrix)
    if eltype(A) == Float16
        A_hp = Float32.(A)
        _, tau = CUSOLVER.geqrf!(A_hp)
        A .= A_hp
        return _form_T(A_hp, tau)
    else
        _, tau = CUSOLVER.geqrf!(A)
        return _form_T(A, tau)
    end
end

"""
    _apply_block_reflector!(Vpanel::AbstractMatrix, Tm::AbstractMatrix, C::AbstractMatrix)

Applies `C <- Q' * C = (I - V * T' * V') * C`, where the Householder vectors `V`
are stored below the diagonal of the factored panel `Vpanel` (unit diagonal
implicit) and `Tm` is the panel's block reflector factor.

The trapezoidal `V = [V11; V21]` is split into its unit lower triangular head
`V11` (applied with `unified_rectrxm!` in TRMM mode) and its rectangular tail
`V21` (applied with `recgemm!`), so that the entire update flows through the
custom mixed-precision subroutines:

    W  = T' * (V11' * C1 + V21' * C2)
    C1 = C1 - V11 * W
    C2 = C2 - V21 * W
"""
function _apply_block_reflector!(Vpanel::AbstractMatrix, Tm::AbstractMatrix, C::AbstractMatrix)
    m, k = size(Vpanel)
    TT = eltype(Tm)

    C1 = view(C, 1:k, :)
    V11 = TT.(view(Vpanel, 1:k, 1:k))

    W = TT.(C1)
    unified_rectrxm!('L', 'L', 'T', 'U', 1.0f0, 'M', V11, W)

    if m > k
        C2 = view(C, k+1:m, :)
        V21 = view(Vpanel, k+1:m, 1:k)
        recgemm!(1.0f0, _transpose_copy(V21), C2, 1.0f0, W)
    end

    unified_rectrxm!('L', 'U', 'T', 'N', 1.0f0, 'M', Tm, W)

    V11W = copy(W)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'M', V11, V11W)
    C1 .-= V11W

    if m > k
        recgemm!(-1.0f0, view(Vpanel, k+1:m, 1:k), W, 1.0f0, view(C, k+1:m, :))
    end

    return C
end

"""
    _combine_T(P1::AbstractMatrix, P2::AbstractMatrix, T1::AbstractMatrix, T2::AbstractMatrix)

Assembles the block reflector factor of the merged panel `[P1 P2]` from the
factors `T1`, `T2` of its factored sub-panels (Elmroth-Gustavson coupling):

    T = [ T1  T3 ],   T3 = -T1 * (V1' * V~2) * T2
        [  0  T2 ]

where `V~2` is `V2` padded with `n1` leading zero rows. The inner product
`V1' * V~2` is split into a unit-lower-triangular TRMM piece (`unified_rectrxm!`)
against the triangle of `V2` and a rectangular `recgemm!` piece against its tail;
the two triangular scalings by `T1` and `T2` are applied with `unified_rectrxm!`.
"""
function _combine_T(P1::AbstractMatrix, P2::AbstractMatrix, T1::AbstractMatrix, T2::AbstractMatrix)
    m, n1 = size(P1)
    n2 = size(P2, 2)
    n = n1 + n2
    TT = promote_type(eltype(T1), eltype(T2))

    T1c = (eltype(T1) == TT) ? T1 : TT.(T1)
    T2c = (eltype(T2) == TT) ? T2 : TT.(T2)

    Y = TT.(_transpose_copy(view(P1, n1+1:n1+n2, 1:n1)))
    V2tri = TT.(view(P2, n1+1:n1+n2, 1:n2))
    unified_rectrxm!('R', 'L', 'N', 'U', 1.0f0, 'M', V2tri, Y)

    if m > n1 + n2
        recgemm!(1.0f0,
                 _transpose_copy(view(P1, n1+n2+1:m, 1:n1)),
                 view(P2, n1+n2+1:m, 1:n2),
                 1.0f0, Y)
    end

    unified_rectrxm!('L', 'U', 'N', 'N', -1.0f0, 'M', T1c, Y)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'M', T2c, Y)

    Tm = fill!(similar(T1c, TT, n, n), zero(TT))
    Tm[1:n1, 1:n1] .= T1c
    Tm[1:n1, n1+1:n] .= Y
    Tm[n1+1:n, n1+1:n] .= T2c

    return Tm
end

"""
    qr_recursive!(A::AbstractMatrix, block_size::Int=256)

Flat recursive, in-place, column-wise block QR factorization driver
(Elmroth-Gustavson formulation), using `CUSOLVER.geqrf!` at the base case.
Bypasses the mixed-precision struct to directly chunk flat arrays, keeping
the bulk of operations in their native precision.

On return, `A` holds the Householder vectors `V` below the diagonal (unit
diagonal implicit) and `R` in its upper triangle. Returns the upper triangular
block reflector factor `T` such that `Q = I - V * T * V'`.
"""
function qr_recursive!(A::AbstractMatrix, block_size::Int=256)
    m = size(A, 1)
    n = size(A, 2)
    @assert m >= n "A must be square or tall (m >= n)"

    if n <= block_size
        return _qr_base!(A)
    end

    n1 = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    A1 = @view A[:, 1:n1]
    A2 = @view A[:, n1+1:end]

    T1 = qr_recursive!(A1, block_size)

    _apply_block_reflector!(A1, T1, A2)

    T2 = qr_recursive!(view(A, n1+1:m, n1+1:n), block_size)

    return _combine_T(A1, A2, T1, T2)
end

"""
    qr_recursive_mixed!(A::PanelMixedPrec{T_Base}, block_size::Int=2048) where T_Base

Performs a recursive, in-place, column-wise block QR factorization
(Elmroth-Gustavson formulation) on a `PanelMixedPrec` matrix, using the flat
`qr_recursive!` driver (CUSOLVER base case) at the leaf nodes.

At each level, the left panel is factored recursively within the mixed-precision
hierarchy; the flat right panel is updated with the left panel's block reflector
through `unified_rectrxm!` and `recgemm!`, and its trailing rows are then factored
with the flat driver. Because the reflector update and the Householder vectors
are homogeneous of degree one and zero respectively in the panel data, quantized
`Float16` panels carrying a scale factor produce exact `V`/`T` factors, with only
the stored `R` entries remaining divided by their panel scale.

On return, the structure holds `V` below the diagonal and `R` above; the
combined upper triangular block reflector factor `T` is returned, such that
`Q = I - V * T * V'`.
"""
function qr_recursive_mixed!(A::PanelMixedPrec{T_Base}, block_size::Int=2048) where {T_Base}
    if A.BaseCase !== nothing
        return qr_recursive!(A.BaseCase, block_size)
    end

    m = size(A, 1)
    n1 = size(A.Left, 2)

    T1 = qr_recursive_mixed!(A.Left, block_size)

    V1 = reconstruct_matrix(A.Left)

    _apply_block_reflector!(V1, T1, A.Right)

    T2 = qr_recursive!(view(A.Right, n1+1:m, :), block_size)

    return _combine_T(V1, A.Right, T1, T2)
end

export PanelMixedPrec, reconstruct_matrix

"""
    PanelMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to tall (or square)
dense matrices, partitioned **column-wise** for panel-based recursive factorizations
(Elmroth-Gustavson QR). Unlike `FullMixedPrec`, which uses a 2x2 grid split, this
structure recursively partitions the matrix into a left panel (`Left`, recursive) and
a right panel (`Right`, stored flat), enabling varying precision levels to be stored
dynamically at different depths of the column recursion while keeping every panel
addressable over its full row range — a requirement for the trailing-row updates and
row-subview factorizations of block QR.
"""
struct PanelMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    Left::Union{PanelMixedPrec{T_Base}, Nothing}
    Right::Union{AbstractMatrix, Nothing}
    right_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    sz::Tuple{Int, Int}
end

"""
    PanelMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs a `PanelMixedPrec` representation of the dense `m x n` matrix `A` (`m >= n`).

The columns are partitioned using a base-2 recursive splitting scheme. If the column
dimension `n` is a power of 2, it splits evenly; otherwise, it splits at the largest
power of 2 less than `n`. At each level the right panel is stored flat at `precisions[1]`
and the left panel recurses with the remaining precisions, so that the leading panel —
which drives the accuracy of the factorization — is held at the highest precision. The
recursion continues until only one precision remains, at which point it forms a base case.
For panels assigned `Float16`, dynamic per-panel quantization is applied to prevent
numerical overflow: values exceeding `65504.0f0` are scaled, clamped, and stored
alongside a `Float32` scaling factor.
"""
function PanelMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    m = size(A, 1)
    n = size(A, 2)

    @assert m >= n "A must be square or tall (m >= n) for the panel QR structure"

    if length(precisions) == 1 || n <= 1
        T_Base = precisions[1]
        local base_matrix
        local base_scale

        if T_Base == Float16
            alpha = maximum(abs, A)
            if alpha > FP16_MAX_VAL
                base_scale = Float32(alpha / FP16_MAX_VAL)
                base_matrix = similar(A, Float16, size(A))
                @. base_matrix = Float16(round(clamp(A / base_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
            else
                base_scale = nothing
                base_matrix = similar(A, Float16, size(A))
                base_matrix .= A
            end
        else
            base_matrix = similar(A, T_Base, size(A))
            base_matrix .= A
            base_scale = nothing
        end

        return PanelMixedPrec{T_Base}(nothing, nothing, nothing, base_scale, base_matrix, (m, n))
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_Right = precisions[1]
    remaining_precisions = precisions[2:end]

    Left = PanelMixedPrec(view(A, :, 1:mid); precisions=remaining_precisions)

    view_Right = view(A, :, mid+1:n)

    local right_matrix
    local right_scale = nothing

    if T_Right == Float16
        alpha_right = maximum(abs, view_Right)
        if alpha_right > FP16_MAX_VAL
            right_scale = Float32(alpha_right / FP16_MAX_VAL)
            right_matrix = similar(view_Right, Float16, size(view_Right))
            @. right_matrix = Float16(round(clamp(view_Right / right_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            right_matrix = similar(view_Right, Float16, size(view_Right))
            right_matrix .= view_Right
        end
    else
        right_matrix = similar(A, T_Right, size(view_Right))
        right_matrix .= view_Right
    end

    T_Final_Base = precisions[end]
    return PanelMixedPrec{T_Final_Base}(Left, right_matrix, right_scale, nothing, nothing, (m, n))
end

function Base.size(A::PanelMixedPrec)
    return A.sz
end

function Base.getindex(A::PanelMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end

    mid = size(A.Left, 2)

    if j <= mid
        return A.Left[i, j]
    else
        return A.Right[i, j - mid]
    end
end

"""
    reconstruct_matrix(A::PanelMixedPrec{T_Base})

Copies the hierarchical mixed-precision panel matrix back into a flat, full-precision
standard Matrix. Note that, consistent with the other mixed-precision structures, stored
values are copied verbatim: for `Float16` panels that carry a quantization scale, the
Householder vectors of a factored panel are scale-invariant and reconstruct exactly,
while the corresponding `R` entries remain divided by their panel scale factor.
"""
function reconstruct_matrix(A::PanelMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end

    C_left = reconstruct_matrix(A.Left)
    C_right = A.Right

    m, n1 = size(C_left)
    n2 = size(C_right, 2)
    n = n1 + n2

    C_full = similar(C_right, T_Base, m, n)

    C_full[:, 1:n1] .= C_left
    C_full[:, n1+1:n] .= C_right

    return C_full
end