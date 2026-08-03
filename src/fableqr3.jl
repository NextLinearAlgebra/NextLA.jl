# ==========================================================================
# recqr.jl
#
# In-place, column-wise (panel) block-recursive QR factorization in the
# Elmroth–Gustavson compact-WY formulation, built on the mixed-precision
# runtime (recgemm!, unified_rectrxm!, GEMM_ADD!, GEMM_SUB!) and the
# AbstractMixedPrec hierarchy.
#
# Follows the structural style, base-case fallback logic, and multiple
# dispatch architecture of potrf_recursive! and lu_recursive!/
# lu_recursive_mixed!. CUSOLVER (geqrf!) is used ONLY at the recursion
# base case.
#
# Factorization convention (LAPACK compact-WY):
#     A = Q * R,       Q = I - Y * T * Y'
# On exit:
#   - R overwrites the upper trapezoid of the storage,
#   - the Householder vectors Y (unit lower trapezoidal, unit diagonal
#     implicit, never stored) overwrite the strict lower trapezoid,
#   - the upper-triangular block reflector T is returned.
#
# Assumes the project files providing recgemm!, unified_rectrxm!,
# GEMM_ADD!, GEMM_SUB! are already included (matmul.jl / rectrxm.jl).
# ==========================================================================

export PanelMixedPrec, geqrf_recursive!, reconstruct_matrix

using CUDA
using LinearAlgebra: diagind, transpose, Transpose
include("wrappers.jl")

# Power-of-2 split rule shared by the whole codebase.
_split_dim(n::Int) = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

# Precision used for the T reflector and dense workspaces.
_work_precision(::Type{Float64}) = Float64
_work_precision(::Type{T}) where {T} = Float32

# ==========================================================================
# PanelMixedPrec: column-wise recursive mixed-precision panel structure
# ==========================================================================

"""
    PanelMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to tall
dense matrices partitioned **column-wise** into panels, as required by the
Elmroth-Gustavson recursive QR formulation (the 2x2 grid structures
`FullMixedPrec`/`SymmMixedPrec`/`TriMixedPrec` split both dimensions and are
incompatible with panel factorizations).

The matrix is split into a leading panel `A1` (recursive) and a trailing
panel `A2`. Leaves (`BaseCase`) are dense, full-height `m x w` panels stored
in a single precision. Each panel spans **all rows**, so both the Householder
vectors (below the diagonal) and the R factor (on/above the diagonal) of a
QR factorization live inside the leaf panels, exactly like the flat in-place
LAPACK layout.
"""
struct PanelMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A1::Union{PanelMixedPrec{T_Base}, Nothing}   # leading column panel (m x n1)
    A2::Union{PanelMixedPrec{T_Base}, Nothing}   # trailing column panel (m x n2)
    BaseCase::Union{AbstractMatrix, Nothing}     # dense leaf panel (m x w)
    base_scale::Union{Float32, Nothing}          # Float16 dynamic quantization scale
    sz::Tuple{Int, Int}
end

"""
    PanelMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs a `PanelMixedPrec` representation of the tall/square matrix `A`
(`m >= n` required).

Columns are partitioned with the same base-2 recursive splitting scheme used
by the other structures. Mirroring the `FullMixedPrec` convention, the current
level consumes `precisions[1]` for its trailing panel `A2` (stored as a dense
leaf), and the leading panel `A1` recurses with the remaining precisions, so
that `precisions[end]` ends up on the leading (numerically most influential)
columns. For `Float16` leaves, dynamic per-panel quantization is applied:
values exceeding `65504.0f0` are scaled, clamped, and stored alongside a
`Float32` scaling factor.
"""
function PanelMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    m, n = size(A)
    @assert m >= n "A must be tall or square (m >= n) for the QR panel structure"

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

        return PanelMixedPrec{T_Base}(nothing, nothing, base_matrix, base_scale, (m, n))
    end

    mid = _split_dim(n)

    T_Trail = precisions[1]
    remaining_precisions = precisions[2:end]

    A1 = PanelMixedPrec(view(A, :, 1:mid); precisions=remaining_precisions)
    A2 = PanelMixedPrec(view(A, :, mid+1:n); precisions=[T_Trail])

    T_Final_Base = remaining_precisions[end]
    return PanelMixedPrec{T_Final_Base}(A1, A2, nothing, nothing, (m, n))
end

Base.size(A::PanelMixedPrec) = A.sz

function Base.getindex(A::PanelMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end
    n1 = size(A.A1, 2)
    return j <= n1 ? A.A1[i, j] : A.A2[i, j - n1]
end

function Base.sizeof(A::PanelMixedPrec)
    if A.BaseCase !== nothing
        return sizeof(A.BaseCase)
    end
    return sizeof(A.A1) + sizeof(A.A2)
end

"""
    reconstruct_matrix(A::PanelMixedPrec{T_Base})

Copies the hierarchical panel structure back into a flat, full-precision
standard matrix. `Float16` leaves carrying a quantization scale are
dequantized on the way out.
"""
function reconstruct_matrix(A::PanelMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        if A.base_scale !== nothing
            C = similar(A.BaseCase, Float32, size(A.BaseCase))
            @. C = A.BaseCase * A.base_scale
            return C
        end
        return copy(A.BaseCase)
    end

    C1 = reconstruct_matrix(A.A1)
    C2 = reconstruct_matrix(A.A2)
    m, n = A.sz
    n1 = size(C1, 2)

    C = similar(C1, T_Base, m, n)
    C[:, 1:n1] .= C1
    C[:, n1+1:n] .= C2
    return C
end

# Collect leaves of a panel tree as (panel, scale, column_offset, width),
# left to right. The tuple order matches the column ordering of Y.
function _collect_leaves!(out::Vector{Tuple{AbstractMatrix, Float32, Int, Int}},
                          A::PanelMixedPrec, c0::Int)
    if A.BaseCase !== nothing
        s = A.base_scale === nothing ? 1.0f0 : A.base_scale
        push!(out, (A.BaseCase, s, c0, size(A.BaseCase, 2)))
        return out
    end
    _collect_leaves!(out, A.A1, c0)
    _collect_leaves!(out, A.A2, c0 + size(A.A1, 2))
    return out
end

_leaf_panels(A::PanelMixedPrec) =
    _collect_leaves!(Tuple{AbstractMatrix, Float32, Int, Int}[], A, 0)

# ==========================================================================
# Small shared helpers
# ==========================================================================

# Materialize a (possibly Float16-quantized) block in the work precision,
# applying the dequantization scale. Returns the original view untouched when
# no conversion is needed. Note: the *strict lower* part is what triangular
# multiplies consume (diag = 'U'), so scaling the whole block is safe -- the
# implicit unit diagonal is never read and must NOT be scaled.
function _scaled_block(X::AbstractMatrix, s::Float32, ::Type{TW}) where {TW}
    if eltype(X) == TW && s == 1.0f0
        return X
    end
    out = similar(X, TW, size(X))
    @. out = X * s
    return out
end

# C += s * A' * B  -- routed through the provided GEMM_ADD! helper so that
# Float16 x Float16 products stay on the tensor-core gemmEx! path with the
# dequantization scale folded into the accumulation (never LinearAlgebra.mul!).
function _gemm_tn_add!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, s::Float32)
    TA, TB, TC = eltype(A), eltype(B), eltype(C)
    if TA == Float16
        # Project convention (cf. unified_rec_mixed): drop the operand to the
        # Float16 compute precision and accumulate in >= Float32.
        B16 = (TB == Float16) ? B : Float16.(B)
        if TC == Float64
            C32 = Float32.(C)
            GEMM_ADD!(transpose(A), B16, C32, s)
            copyto!(C, C32)
        else
            GEMM_ADD!(transpose(A), B16, C, s)
        end
    else
        Am = (TA == TC) ? A : TC.(A)
        Bm = (TB == TC) ? B : TC.(B)
        GEMM_ADD!(transpose(Am), Bm, C, s)
    end
end

# C -= s * A * B  -- Float16 goes through GEMM_SUB! (scaled gemmEx!), every
# other combination goes through recgemm!, which owns the type dispatch.
function _gemm_nn_sub!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, s::Float32)
    if eltype(A) == Float16
        B16 = (eltype(B) == Float16) ? B : Float16.(B)
        if eltype(C) == Float64
            C32 = Float32.(C)
            GEMM_SUB!(C32, A, B16, s)
            copyto!(C, C32)
        else
            GEMM_SUB!(C, A, B16, s)
        end
    else
        recgemm!(-s, A, B, 1.0f0, C)
    end
end

# ==========================================================================
# Block reflector (T factor) construction and application
# ==========================================================================

# T12 = -T1 * ( (s1*Ylow1)' * (s2-scaled unit-lower-trapezoidal Y2) ) * T2
#
# Ylow1 : rows of Y1 strictly below the first k1 diagonal rows  ((mb) x k1)
# Y2    : unit lower trapezoidal factor of the trailing panel   ((mb) x k2)
# The unit diagonal of Y2 is implicit ('U' diag), so quantization scales only
# touch the stored strict lower part, which is exactly right.
function _t_coupling!(T12::AbstractMatrix,
                      Ylow1::AbstractMatrix, s1::Float32,
                      Y2::AbstractMatrix, s2::Float32,
                      T1::AbstractMatrix, T2::AbstractMatrix)
    k2 = size(T2, 1)
    mb = size(Y2, 1)
    TW = eltype(T12)

    # G = (s1 * Ylow1[1:k2, :])'  then  G <- G * unit_lower(Y2[1:k2, 1:k2])
    T12 .= transpose(view(Ylow1, 1:k2, :)) .* s1
    tri2 = _scaled_block(view(Y2, 1:k2, 1:k2), s2, TW)
    unified_rectrxm!('R', 'L', 'N', 'U', 1.0f0, 'M', tri2, T12)

    # G += (s1*s2) * Ylow1[k2+1:mb, :]' * Y2[k2+1:mb, :]
    if mb > k2
        _gemm_tn_add!(T12, view(Ylow1, k2+1:mb, :), view(Y2, k2+1:mb, :), s1 * s2)
    end

    # T12 <- -T1 * G * T2
    unified_rectrxm!('L', 'U', 'N', 'N', 1.0f0, 'M', T1, T12)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'M', T2, T12)
    T12 .*= -1
    return T12
end

# Recursively fill the strictly-upper blocks of T (diagonal = tau already set)
# from the Householder vectors V, via the same coupling identity used by the
# outer recursion.
function _larft_fill!(T::AbstractMatrix, V::AbstractMatrix)
    k = size(T, 1)
    k <= 1 && return T

    k1 = _split_dim(k)
    m = size(V, 1)

    T11 = view(T, 1:k1, 1:k1)
    T22 = view(T, k1+1:k, k1+1:k)

    _larft_fill!(T11, view(V, :, 1:k1))
    _larft_fill!(T22, view(V, k1+1:m, k1+1:k))

    _t_coupling!(view(T, 1:k1, k1+1:k),
                 view(V, k1+1:m, 1:k1), 1.0f0,
                 view(V, k1+1:m, k1+1:k), 1.0f0,
                 T11, T22)
    return T
end

function _build_T(V::AbstractMatrix, tau::AbstractVector, ::Type{TW}) where {TW}
    k = length(tau)
    T = fill!(similar(V, TW, k, k), zero(TW))
    view(T, diagind(T)) .= tau
    _larft_fill!(T, V)
    return T
end

"""
    _apply_block_qt!(V, s, T, B)

Applies `Q' = (I - V*T*V')' = I - V*T'*V'` to `B` in place, where `V` is a
unit lower trapezoidal Householder panel (possibly `Float16`-quantized with
scale `s`) and `T` its upper-triangular block reflector:

    W  = V' * B          (unit-lower trmm via `unified_rectrxm!` + `GEMM_ADD!`)
    W <- T' * W          (upper trmm via `unified_rectrxm!`)
    B <- B - V * W       (`recgemm!`/`GEMM_SUB!` + unit-lower trmm)
"""
function _apply_block_qt!(V::AbstractMatrix, s::Float32, T::AbstractMatrix, B::AbstractMatrix)
    m, k = size(V)
    TW = eltype(T)

    # ---- W = Y' * B -------------------------------------------------------
    W = similar(T, k, size(B, 2))
    W .= view(B, 1:k, :)
    Vtri = _scaled_block(view(V, 1:k, 1:k), s, TW)
    unified_rectrxm!('L', 'L', 'T', 'U', 1.0f0, 'M', Vtri, W)          # W = Y11' * B1
    if m > k
        _gemm_tn_add!(W, view(V, k+1:m, :), view(B, k+1:m, :), s)      # W += s*Y21' * B2
    end

    # ---- W <- T' * W ------------------------------------------------------
    unified_rectrxm!('L', 'U', 'T', 'N', 1.0f0, 'M', T, W)

    # ---- B <- B - Y * W ---------------------------------------------------
    if m > k
        _gemm_nn_sub!(view(B, k+1:m, :), view(V, k+1:m, :), W, s)      # B2 -= s*Y21 * W
    end
    W2 = copy(W)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'M', Vtri, W2)         # B1 -= Y11 * W
    view(B, 1:k, :) .-= W2
    return B
end

# ==========================================================================
# Dense recursive QR (AbstractMatrix base-case version)
# ==========================================================================

function _geqrf_base!(A::AbstractMatrix)
    TW = _work_precision(eltype(A))
    if eltype(A) == Float16
        # CUSOLVER has no Float16 geqrf; mirror lu_recursive!'s base-case
        # promotion pattern.
        A_f32 = Float32.(A)
        _, tau = CUSOLVER.geqrf!(A_f32)
        T = _build_T(A_f32, tau, TW)
        A .= Float16.(A_f32)
        return T
    else
        _, tau = CUSOLVER.geqrf!(A)
        return _build_T(A, tau, TW)
    end
end

"""
    geqrf_recursive!(A::AbstractMatrix, block_size::Int=256) -> T

Flat, in-place, column-wise block-recursive QR factorization driver
(Elmroth-Gustavson). Partitions `A = [A1  A2]` and executes, in order:

1. `A1 -> Q1 R1`                       (recursive factorization of the left panel)
2. `A2 <- Q1' A2`                      (`recgemm!` / `unified_rectrxm!` block updates;
                                        rows `1:n1` of the result are `R12`)
3. `A2[n1+1:m, :] -> Q2 R2`            (recursive factorization of the trailing panel)
4. `Q = Q1 Q2`, `R = [R1 R12; 0 R2]`   (assembled implicitly in place; the
                                        aggregated block reflector `T` is returned,
                                        with `Q = I - Y*T*Y'`)

CUSOLVER `geqrf!` is used only at the recursion base case (`n <= block_size`).
"""
function geqrf_recursive!(A::AbstractMatrix, block_size::Int=256)
    m, n = size(A)
    @assert m >= n "geqrf_recursive! requires a tall or square panel (m >= n)"

    if n <= block_size
        return _geqrf_base!(A)
    end

    n1 = _split_dim(n)
    A1 = view(A, :, 1:n1)
    A2 = view(A, :, n1+1:n)

    # Step 1: factorize the left panel  A1 -> Q1 R1
    T1 = geqrf_recursive!(A1, block_size)

    # Step 2: apply the accumulated Householder transformations  A2 <- Q1' A2
    _apply_block_qt!(A1, 1.0f0, T1, A2)

    # Step 3: factorize the updated trailing panel  A2 -> Q2 R2
    T2 = geqrf_recursive!(view(A2, n1+1:m, :), block_size)

    # Step 4: construct the complete recursive factorization
    #   T = [T1  T3]        T3 = -T1 * (Y1' * Y2) * T2,    Q = Q1 Q2 = I - Y*T*Y'
    #       [ 0  T2]
    TW = promote_type(eltype(T1), eltype(T2))
    T = fill!(similar(T1, TW, n, n), zero(TW))
    view(T, 1:n1, 1:n1) .= T1
    view(T, n1+1:n, n1+1:n) .= T2
    _t_coupling!(view(T, 1:n1, n1+1:n),
                 view(A1, n1+1:m, :), 1.0f0,
                 view(A2, n1+1:m, :), 1.0f0,
                 T1, T2)
    return T
end

# ==========================================================================
# Mixed-precision recursive QR (PanelMixedPrec version)
# ==========================================================================

"""
    geqrf_recursive!(A::PanelMixedPrec, block_size::Int=2048) -> T

Performs an in-place, column-wise block-recursive QR factorization on a
`PanelMixedPrec` matrix structure, following the same four-step
Elmroth-Gustavson scheme as the dense driver. All trailing-panel block
updates are performed through `recgemm!`/`unified_rectrxm!` (and the
scale-aware `GEMM_ADD!`/`GEMM_SUB!` helpers they build on) so the precision
hierarchy and the `Float16` dynamic-quantization scales are respected
end-to-end. Falls back to the flat `geqrf_recursive!` driver (and thus to
CUSOLVER) at the leaf panels, mirroring `lu_recursive_mixed!`.

Returns the aggregated upper-triangular block reflector `T` (in `Float32`,
or `Float64` for all-double hierarchies), so that `Q = I - Y*T*Y'` with `Y`
stored below the diagonal inside the panel leaves and `R` on/above it.
"""
function geqrf_recursive!(A::PanelMixedPrec, block_size::Int=2048)
    return _geqrf_rec_mixed!(A, 0, block_size)
end

# r0 = number of already-eliminated rows above the active window; the panel
# leaves span all m rows, so the active part of any leaf is rows r0+1:m.
function _geqrf_rec_mixed!(A::PanelMixedPrec, r0::Int, block_size::Int)
    m, n = size(A)

    if A.BaseCase !== nothing
        P = A.BaseCase
        P_act = view(P, r0+1:m, :)

        if eltype(P) == Float16 && A.base_scale !== nothing
            # Dequantize the active window, factorize in Float32, requantize.
            # The stored scale is reused on the way back (struct is immutable),
            # matching the requantization convention of the existing drivers;
            # the Householder vectors are O(1) so they remain representable.
            s = A.base_scale
            P_f32 = Float32.(P_act) .* s
            T = geqrf_recursive!(P_f32, block_size)
            @. P_act = Float16(clamp(P_f32 / s, -65504.0f0, 65504.0f0))
            return T
        else
            return geqrf_recursive!(P_act, block_size)
        end
    end

    n1 = size(A.A1, 2)

    # Step 1: factorize the left panel  A1 -> Q1 R1
    T1 = _geqrf_rec_mixed!(A.A1, r0, block_size)

    # Step 2: apply the accumulated Householder transformations  A2 <- Q1' A2
    _apply_qt_mixed!(A.A1, r0, T1, A.A2)

    # Step 3: factorize the updated trailing panel  A2 -> Q2 R2
    T2 = _geqrf_rec_mixed!(A.A2, r0 + n1, block_size)

    # Step 4: construct the complete recursive factorization
    #   Q = Q1 Q2,   R = [R1 R12; 0 R2] (in place),   T3 = -T1*(Y1'*Y2)*T2
    TW = promote_type(eltype(T1), eltype(T2))
    T = fill!(similar(T1, TW, n, n), zero(TW))
    view(T, 1:n1, 1:n1) .= T1
    view(T, n1+1:n, n1+1:n) .= T2
    _t_coupling_mixed!(view(T, 1:n1, n1+1:n), A.A1, A.A2, r0, T1, T2)
    return T
end

# Apply Q1' (Householder panel stored in the mixed structure Y, active from
# row r0+1) to the trailing mixed structure B. Q' acts column-independently,
# so we recurse down B's panel tree and process each dense leaf.
function _apply_qt_mixed!(Y::PanelMixedPrec, r0::Int, T::AbstractMatrix, B::PanelMixedPrec)
    if B.BaseCase !== nothing
        m = size(B, 1)
        P = B.BaseCase
        B_act = view(P, r0+1:m, :)

        if eltype(P) == Float16 && B.base_scale !== nothing
            s = B.base_scale
            B_f32 = Float32.(B_act) .* s
            _apply_qt_leaves!(Y, r0, T, B_f32)
            @. B_act = Float16(clamp(B_f32 / s, -65504.0f0, 65504.0f0))
        else
            _apply_qt_leaves!(Y, r0, T, B_act)
        end
        return
    end

    _apply_qt_mixed!(Y, r0, T, B.A1)
    _apply_qt_mixed!(Y, r0, T, B.A2)
    return
end

# Monolithic compact-WY application  B <- (I - Y*T'*Y') B  where the unit
# lower trapezoidal Y spans the leaves of a mixed panel tree. Each leaf
# contributes a unit-triangular trmm (small, dequantized via `_scaled_block`)
# plus a bulk rectangular GEMM in its native precision with the quantization
# scale folded in. B is a dense matrix whose row 1 corresponds to global row
# r0+1.
function _apply_qt_leaves!(Y::PanelMixedPrec, r0::Int, T::AbstractMatrix, B::AbstractMatrix)
    m = size(Y, 1)
    k = size(Y, 2)
    ma = size(B, 1)                       # == m - r0
    TW = eltype(T)
    leaves = _leaf_panels(Y)

    # ---- W = Y' * B, assembled leaf by leaf ------------------------------
    W = similar(T, k, size(B, 2))
    for (P, s, c0, w) in leaves
        Wl = view(W, c0+1:c0+w, :)
        d2 = c0 + w                         # last diagonal row of this leaf (B-relative)
        Wl .= view(B, c0+1:d2, :)
        Ytri = _scaled_block(view(P, r0+c0+1:r0+d2, :), s, TW)
        unified_rectrxm!('L', 'L', 'T', 'U', 1.0f0, 'M', Ytri, Wl)
        if ma > d2
            _gemm_tn_add!(Wl, view(P, r0+d2+1:m, :), view(B, d2+1:ma, :), s)
        end
    end

    # ---- W <- T' * W ------------------------------------------------------
    unified_rectrxm!('L', 'U', 'T', 'N', 1.0f0, 'M', T, W)

    # ---- B <- B - Y * W, accumulated leaf by leaf ------------------------
    # The "-=" contributions of different leaves overlap in rows but only read
    # W (finalized above), so the accumulation order is immaterial.
    for (P, s, c0, w) in leaves
        Wl = view(W, c0+1:c0+w, :)
        d2 = c0 + w
        Ytri = _scaled_block(view(P, r0+c0+1:r0+d2, :), s, TW)
        W2 = copy(Wl)
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'M', Ytri, W2)
        view(B, c0+1:d2, :) .-= W2
        if ma > d2
            _gemm_nn_sub!(view(B, d2+1:ma, :), view(P, r0+d2+1:m, :), Wl, s)
        end
    end
    return B
end

# T12 = -T1 * (Y1' * Y2) * T2 across two mixed panel trees. Y1's rows below
# global row r0+n1 are pure Householder values (all of Y1's triangles end at
# or above that row), while Y2's leaves each contribute a unit-triangular
# block plus a rectangular block; quantization scales multiply through.
function _t_coupling_mixed!(T12::AbstractMatrix,
                            Y1::PanelMixedPrec, Y2::PanelMixedPrec,
                            r0::Int, T1::AbstractMatrix, T2::AbstractMatrix)
    m = size(Y1, 1)
    n1 = size(Y1, 2)
    TW = eltype(T12)
    fill!(T12, zero(TW))

    y1_leaves = _leaf_panels(Y1)

    for (P2, s2, c2, w2) in _leaf_panels(Y2)
        d1 = r0 + n1 + c2                  # global row just above this Y2 leaf's diagonal
        tri2 = _scaled_block(view(P2, d1+1:d1+w2, :), s2, TW)

        for (P1, s1, c1, w1) in y1_leaves
            Gb = view(T12, c1+1:c1+w1, c2+1:c2+w2)

            # Triangular contribution: (s1*Y1[d-rows])' * (I + s2*strict_lower(Y2))
            Gtmp = similar(T12, w1, w2)
            Gtmp .= transpose(view(P1, d1+1:d1+w2, :)) .* s1
            unified_rectrxm!('R', 'L', 'N', 'U', 1.0f0, 'M', tri2, Gtmp)
            Gb .+= Gtmp

            # Rectangular contribution below the Y2 triangle
            if m > d1 + w2
                _gemm_tn_add!(Gb, view(P1, d1+w2+1:m, :), view(P2, d1+w2+1:m, :), s1 * s2)
            end
        end
    end

    # T12 <- -T1 * (Y1'*Y2) * T2
    unified_rectrxm!('L', 'U', 'N', 'N', 1.0f0, 'M', T1, T12)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'M', T2, T12)
    T12 .*= -1
    return T12
end