Great — below is a drop-in implementation in the same structural style as your SymmMixedPrec / TriMixedPrec + recursive kernels.
I split it into 3 parts:
FullMixedPrec (new hierarchical full matrix structure with the same Float16 dynamic quantization logic)
recgemm! (new recursive GEMM kernel with @sync/@async parallel structure)
lu_recursive! (nonpivoting, in-place, recursive block LU with multiple dispatch for AbstractMatrix and FullMixedPrec)
Note: for the base case, I call a no-pivot CUSOLVER wrapper (getrfnp!) expected from your wrappers.jl. If your wrapper has a different name, swap that one line.
using LinearAlgebra
using CUDA
include("wrappers.jl")
include("rectrxm.jl")

# ============================================================
# 1) FullMixedPrec
# ============================================================

"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

Hierarchical recursive mixed-precision structure for full dense square matrices.

Block layout:
    A = [ A11  A12
          A21  A22 ]

- A11, A22 recurse as `FullMixedPrec`
- A12, A21 are stored as dense rectangular blocks
- Float16 blocks use dynamic quantization with per-block scaling
"""
struct FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base}, Nothing}
    A22::Union{FullMixedPrec{T_Base}, Nothing}
    A12::Union{AbstractMatrix, Nothing}
    A21::Union{AbstractMatrix, Nothing}
    A12_scale::Union{Float32, Nothing}
    A21_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    sz::Tuple{Int, Int}
end

"""
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs recursive mixed-precision full matrix representation.
Applies the exact Float16 dynamic quantization logic used in your other structures.
"""
function FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    @assert n == size(A, 2) "A must be square"

    # Base case
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
            if eltype(A) == T_Base
                base_matrix = A
            else
                base_matrix = similar(A, T_Base, size(A))
                base_matrix .= A
            end
            base_scale = nothing
        end

        return FullMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            nothing, nothing,
            base_scale, base_matrix, (n, n)
        )
    end

    # Recursive split (same style as your constructors)
    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))
    T_OffDiag = precisions[1]
    remaining_precisions = precisions[2:end]

    A11 = FullMixedPrec(view(A, 1:mid, 1:mid); precisions=remaining_precisions)
    A22 = FullMixedPrec(view(A, mid+1:n, mid+1:n); precisions=remaining_precisions)

    A12_view = view(A, 1:mid, mid+1:n)
    A21_view = view(A, mid+1:n, 1:mid)

    local A12_matrix, A21_matrix
    local A12_scale = nothing
    local A21_scale = nothing

    # A12 quantization
    if T_OffDiag == Float16
        alpha_A12 = maximum(abs, A12_view)
        if alpha_A12 > FP16_MAX_VAL
            A12_scale = Float32(alpha_A12 / FP16_MAX_VAL)
            A12_matrix = similar(A12_view, Float16, size(A12_view))
            @. A12_matrix = Float16(round(clamp(A12_view / A12_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            A12_matrix = similar(A12_view, Float16, size(A12_view))
            A12_matrix .= A12_view
        end
    else
        if eltype(A12_view) == T_OffDiag
            A12_matrix = A12_view
        else
            A12_matrix = similar(A, T_OffDiag, size(A12_view))
            A12_matrix .= A12_view
        end
        A12_scale = nothing
    end

    # A21 quantization
    if T_OffDiag == Float16
        alpha_A21 = maximum(abs, A21_view)
        if alpha_A21 > FP16_MAX_VAL
            A21_scale = Float32(alpha_A21 / FP16_MAX_VAL)
            A21_matrix = similar(A21_view, Float16, size(A21_view))
            @. A21_matrix = Float16(round(clamp(A21_view / A21_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
        else
            A21_matrix = similar(A21_view, Float16, size(A21_view))
            A21_matrix .= A21_view
        end
    else
        if eltype(A21_view) == T_OffDiag
            A21_matrix = A21_view
        else
            A21_matrix = similar(A, T_OffDiag, size(A21_view))
            A21_matrix .= A21_view
        end
        A21_scale = nothing
    end

    T_Final_Base = precisions[end]
    return FullMixedPrec{T_Final_Base}(
        A11, A22, A12_matrix, A21_matrix,
        A12_scale, A21_scale,
        nothing, nothing, (n, n)
    )
end

Base.size(A::FullMixedPrec) = A.sz

function Base.getindex(A::FullMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end

    mid = size(A.A11, 1)

    if i <= mid && j <= mid
        return A.A11[i, j]
    elseif i <= mid && j > mid
        return A.A12[i, j - mid]
    elseif i > mid && j <= mid
        return A.A21[i - mid, j]
    else
        return A.A22[i - mid, j - mid]
    end
end

# ============================================================
# 2) recgemm!
# ============================================================

const RECGEMM_PARALLEL_THRESHOLD = 4096

"""
    _gemm_dispatch_nn!(alpha, A, B, beta, C)

Dispatch GEMM with mixed-precision handling:
    C = alpha * A * B + beta * C
"""
function _gemm_dispatch_nn!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix
)
    TA, TB, TC = eltype(A), eltype(B), eltype(C)

    if TA == TB == TC && TC in (Float32, Float64)
        gemm!('N', 'N', TC(alpha), A, B, TC(beta), C)
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!('N', 'N', alpha, A, B, beta, C)
    else
        A_final = (TA == TC) ? A : TC.(A)
        B_final = (TB == TC) ? B : TC.(B)
        if TC in (Float32, Float64)
            gemm!('N', 'N', TC(alpha), A_final, B_final, TC(beta), C)
        else
            gemmEx!('N', 'N', alpha, A_final, B_final, beta, C)
        end
    end
end

function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::AbstractMatrix,
    threshold::Int; parallel::Bool
)
    n, p = size(C)

    if max(n, p) <= threshold
        _gemm_dispatch_nn!(alpha, A, B, beta, C)
        return
    end

    n1 = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))
    p1 = isinteger(log2(p)) ? div(p, 2) : 2^floor(Int, log2(p))

    A1 = @view A[1:n1, :]
    A2 = @view A[n1+1:end, :]
    B1 = @view B[:, 1:p1]
    B2 = @view B[:, p1+1:end]

    C11 = @view C[1:n1,     1:p1]
    C12 = @view C[1:n1,     p1+1:end]
    C21 = @view C[n1+1:end, 1:p1]
    C22 = @view C[n1+1:end, p1+1:end]

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A1, B2, beta, C12, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B1, beta, C21, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
        _recgemm_impl!(alpha, A1, B2, beta, C12, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B1, beta, C21, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
    end
end

function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec,
    threshold::Int; parallel::Bool
)
    if C.BaseCase !== nothing
        recgemm!(alpha, A, B, beta, C.BaseCase, 4096)
        return
    end

    n1 = size(C.A11, 1)
    A1 = @view A[1:n1, :]
    A2 = @view A[n1+1:end, :]
    B1 = @view B[:, 1:n1]
    B2 = @view B[:, n1+1:end]

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C.A11, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A1, B2, beta, C.A12, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B1, beta, C.A21, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C.A22, threshold, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C.A11, threshold, parallel=false)
        _recgemm_impl!(alpha, A1, B2, beta, C.A12, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B1, beta, C.A21, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C.A22, threshold, parallel=false)
    end
end

"""
    recgemm!(alpha, A, B, beta, C::AbstractMatrix, threshold=256; A_scale=1f0, B_scale=1f0)

Recursive GEMM:
    C = alpha * (A_scale*A) * (B_scale*B) + beta*C
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number,
    C::AbstractMatrix, threshold::Int=256;
    A_scale::Float32=1.0f0, B_scale::Float32=1.0f0
)
    alpha_eff = alpha * A_scale * B_scale
    should_parallelize = max(size(C,1), size(C,2)) > RECGEMM_PARALLEL_THRESHOLD
    _recgemm_impl!(alpha_eff, A, B, beta, C, threshold, parallel=should_parallelize)
end

"""
    recgemm!(alpha, A, B, beta, C::FullMixedPrec; A_scale=1f0, B_scale=1f0)

Recursive GEMM where C is hierarchical mixed-precision full structure.
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number,
    C::FullMixedPrec;
    A_scale::Float32=1.0f0, B_scale::Float32=1.0f0, threshold::Int=256
)
    alpha_eff = alpha * A_scale * B_scale

    if C.BaseCase !== nothing
        recgemm!(alpha_eff, A, B, beta, C.BaseCase, 4096)
        return
    end

    n_subproblem = size(C.A11, 1)
    should_parallelize = n_subproblem > RECGEMM_PARALLEL_THRESHOLD
    _recgemm_impl!(alpha_eff, A, B, beta, C, threshold, parallel=should_parallelize)
end

# ============================================================
# 3) Recursive nonpivoting LU (in-place)
# ============================================================

"""
    lu_nopiv_basecase!(A::CUDA.StridedCuArray)

CUSOLVER base case (nonpivoting).
Expected wrapper from `wrappers.jl`: `getrfnp!`
"""
function lu_nopiv_basecase!(A::CUDA.StridedCuArray)
    # Replace with your exact wrapper symbol if different.
    getrfnp!(A)
    return A
end

function lu_nopiv_basecase!(A::AbstractMatrix)
    error("Nonpivoting base case is defined here for CUDA arrays only (CUSOLVER).")
end

"""
    lu_recursive!(A::AbstractMatrix, block_size=256)

In-place nested recursive nonpivoting LU on dense matrix.
Order:
1) LU(A11)
2) A12 <- L11^{-1} A12
3) A21 <- A21 U11^{-1}
4) A22 <- A22 - A21*A12
5) LU(A22)
"""
function lu_recursive!(A::AbstractMatrix, block_size::Int=256)
    n = size(A, 1)
    @assert n == size(A, 2) "A must be square"

    if n <= block_size
        lu_nopiv_basecase!(A)
        return A
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1,       1:n1]
    A12 = @view A[1:n1,       n1+1:end]
    A21 = @view A[n1+1:end,   1:n1]
    A22 = @view A[n1+1:end,   n1+1:end]

    # 1) A11 -> L11 U11
    lu_recursive!(A11, block_size)

    # 2) A12 <- L11^{-1} A12  (L11 is unit-lower in LU storage)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A11, A12)

    # 3) A21 <- A21 U11^{-1}
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A11, A21)

    # 4) A22 <- A22 - A21*A12
    recgemm!(-1.0f0, A21, A12, 1.0f0, A22)

    # 5) A22 -> L22 U22
    lu_recursive!(A22, block_size)

    return A
end

"""
    lu_recursive!(A::FullMixedPrec)

In-place nested recursive nonpivoting LU on full mixed-precision hierarchy.
Same 5-step order as dense kernel.
"""
function lu_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        lu_recursive!(A.BaseCase, 4096)
        return A
    end

    # 1) A11 -> L11 U11
    lu_recursive!(A.A11)

    # 2) A12 <- L11^{-1} A12
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)

    # 3) A21 <- A21 U11^{-1}
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    # 4) A22 <- A22 - (A21 * A12), honoring stored quantization scales
    s21 = A.A21_scale !== nothing ? A.A21_scale : 1.0f0
    s12 = A.A12_scale !== nothing ? A.A12_scale : 1.0f0
    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22; A_scale=s21, B_scale=s12)

    # 5) A22 -> L22 U22
    lu_recursive!(A.A22)

    return A
end

If you want, I can also give you a compact reconstruct_matrix(::FullMixedPrec) + residual checks (‖A-LU‖/‖A‖) matching your existing validation style.

