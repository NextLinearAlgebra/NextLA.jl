using LinearAlgebra
using CUDA
using CUDA.CUSOLVER
using StochasticRounding

abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end

struct TransposedMixedPrec{T, M <: AbstractMixedPrec{T}} <: AbstractMixedPrec{T}
    parent::M
end

include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("trsm.jl")
include("trmm.jl")
include("matmul.jl")
include("rectrxm.jl")
include("recsyrk.jl")
# include("getrf.jl")
# include("wrappers.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# §1 – FullMixedPrec  (hierarchical mixed-precision structure for full matrices)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to full dense
square matrices. It partitions the matrix into two recursive diagonal blocks
(`A11`, `A22`) and two rectangular off-diagonal blocks (`A12` upper, `A21`
lower), structured symmetrically to `TriMixedPrec` but carrying both panels.

After an in-place `getrf_recursive!` call the structure holds the packed LU
factors in LAPACK convention:
  • lower triangle (unit diagonal implicit) → L factor panels
  • upper triangle (diagonal included)      → U factor panels
  • `A21` off-diagonal panels               → L21 sub-blocks
  • `A12` off-diagonal panels               → U12 sub-blocks
"""
struct FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}
    A11::Union{FullMixedPrec{T_Base}, Nothing}
    A22::Union{FullMixedPrec{T_Base}, Nothing}
    A12::Union{AbstractMatrix, Nothing}         # upper off-diagonal
    A21::Union{AbstractMatrix, Nothing}         # lower off-diagonal
    A12_scale::Union{Float32, Nothing}
    A21_scale::Union{Float32, Nothing}
    base_scale::Union{Float32, Nothing}
    BaseCase::Union{AbstractMatrix{T_Base}, Nothing}
    sz::Tuple{Int, Int}
end

"""
    FullMixedPrec(A::AbstractMatrix; precisions::Vector{DataType})

Constructs a `FullMixedPrec` representation of the general square matrix `A`.

Uses a base-2 recursive splitting scheme to partition the matrix into two
diagonal blocks (`A11`, `A22`) and two off-diagonal blocks (`A12`, `A21`).
The `precisions` vector maps levels to element types:
  • `precisions[1]`      → off-diagonal block precision at the current level
  • `precisions[2:end]`  → passed recursively to the diagonal blocks
  • `precisions[end]`    → `T_Base` (innermost dense block element type)

Applies the exact same `Float16` dynamic quantization as `SymmMixedPrec` and
`TriMixedPrec`: values exceeding `65504.0f0` are scaled and clamped per-block.
"""
function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    @assert n == size(A, 2) "FullMixedPrec requires a square matrix, got $(size(A))"

    # ── Base case ─────────────────────────────────────────────────────────────
    if length(precisions) == 1
        T_Base = precisions[1]
        local base_matrix, base_scale

        if T_Base == Float16
            alpha = maximum(abs, A)
            if alpha > FP16_MAX_VAL
                base_scale  = Float32(alpha / FP16_MAX_VAL)
                base_matrix = similar(A, Float16, size(A))
                @. base_matrix = Float16(round(clamp(A / base_scale, -FP16_MAX_VAL, FP16_MAX_VAL)))
            else
                base_scale  = nothing
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
            nothing, nothing, base_scale, base_matrix, (n, n)
        )
    end

    # ── Recursive case ─────────────────────────────────────────────────────────
    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag       = precisions[1]
    remaining_precs = precisions[2:end]
    T_Final_Base    = precisions[end]

    A11 = FullMixedPrec(view(A, 1:mid,   1:mid);   precisions=remaining_precs)
    A22 = FullMixedPrec(view(A, mid+1:n, mid+1:n); precisions=remaining_precs)

    # Quantize a rectangular off-diagonal view to T_OffDiag with overflow guard
    function _quantize(blk)
        local mat, sc
        if T_OffDiag == Float16
            alpha = maximum(abs, blk)
            if alpha > FP16_MAX_VAL
                sc  = Float32(alpha / FP16_MAX_VAL)
                mat = similar(blk, Float16, size(blk))
                @. mat = Float16(round(clamp(blk / sc, -FP16_MAX_VAL, FP16_MAX_VAL)))
            else
                sc  = nothing
                mat = similar(blk, Float16, size(blk))
                mat .= blk
            end
        else
            mat = (eltype(blk) == T_OffDiag) ? blk :
                  (tmp = similar(blk, T_OffDiag, size(blk)); tmp .= blk; tmp)
            sc = nothing
        end
        return mat, sc
    end

    A12_mat, A12_scale = _quantize(view(A, 1:mid,   mid+1:n))
    A21_mat, A21_scale = _quantize(view(A, mid+1:n, 1:mid))

    return FullMixedPrec{T_Final_Base}(
        A11, A22, A12_mat, A21_mat,
        A12_scale, A21_scale, nothing, nothing, (n, n)
    )
end

Base.size(A::FullMixedPrec) = A.sz

function Base.sizeof(A::FullMixedPrec)
    A.BaseCase !== nothing && return sizeof(A.BaseCase)
    return sizeof(A.A11) + sizeof(A.A22) + sizeof(A.A12) + sizeof(A.A21)
end

function Base.getindex(A::FullMixedPrec{T_Base}, i::Int, j::Int) where {T_Base}
    if A.BaseCase !== nothing
        return A.BaseCase[i, j]
    end
    mid = size(A.A11, 1)
    if i <= mid && j <= mid
        return A.A11[i, j]
    elseif i > mid && j > mid
        return A.A22[i - mid, j - mid]
    elseif i <= mid && j > mid          # upper off-diagonal → A12
        v = T_Base(A.A12[i, j - mid])
        return A.A12_scale !== nothing ? v * A.A12_scale : v
    else                                 # lower off-diagonal → A21
        v = T_Base(A.A21[i - mid, j])
        return A.A21_scale !== nothing ? v * A.A21_scale : v
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# §2 – Triangular extraction helpers
#
# After getrf_recursive!(A::FullMixedPrec), produce zero-copy TriMixedPrec views
# of the L and U factors packed inside A without allocating new storage.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    lower_tri_mixed(A::FullMixedPrec{T_Base}) -> TriMixedPrec{T_Base}

Zero-copy alias of the lower-triangular (L) factor in a packed LU `FullMixedPrec`.
Routes `A.A21` panels → `OffDiag` of the returned `TriMixedPrec` (uplo='L').
"""
function lower_tri_mixed(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            A.base_scale, A.BaseCase, 'L', A.sz
        )
    end
    return TriMixedPrec{T_Base}(
        lower_tri_mixed(A.A11),
        lower_tri_mixed(A.A22),
        A.A21,          # lower off-diagonal = L21 panel  →  OffDiag field
        A.A21_scale,    # forwarded as offDiag_scale
        nothing, nothing, 'L', A.sz
    )
end

"""
    upper_tri_mixed(A::FullMixedPrec{T_Base}) -> TriMixedPrec{T_Base}

Zero-copy alias of the upper-triangular (U) factor in a packed LU `FullMixedPrec`.
Routes `A.A12` panels → `OffDiag` of the returned `TriMixedPrec` (uplo='U').
"""
function upper_tri_mixed(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            A.base_scale, A.BaseCase, 'U', A.sz
        )
    end
    return TriMixedPrec{T_Base}(
        upper_tri_mixed(A.A11),
        upper_tri_mixed(A.A22),
        A.A12,          # upper off-diagonal = U12 panel  →  OffDiag field
        A.A12_scale,    # forwarded as offDiag_scale
        nothing, nothing, 'U', A.sz
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# §3 – _gemm_dispatch!
#
# Hardware-routing helper for general matrix multiplication C = alpha*A*B + beta*C.
# Mirrors _syrk_dispatch! in recsyrk.jl: same type-guard logic, same fallback path.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _gemm_dispatch!(alpha, A, B, beta, C)

Handles type-conversion and hardware routing for general matrix multiplications
`C ← alpha·A·B + beta·C`. Mirrors `_syrk_dispatch!` in its dispatch logic:
  • Native Float32/Float64 → `gemm!` (cuBLAS DGEMM/SGEMM)
  • Float16 operands       → `gemmEx!` (Tensor Core accelerated)
  • Mixed types            → upcast to Float32, call `gemm!`, write back
"""
function _gemm_dispatch!(
    alpha::Number, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray,
    beta::Number,  C::CUDA.StridedCuArray
)
    TA = eltype(A); TB = eltype(B); TC = eltype(C)

    if TA == TB == TC && TC in (Float32, Float64)
        gemm!('N', 'N', TC(alpha), A, B, TC(beta), C)

    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!('N', 'N', alpha, A, B, beta, C)

    else
        compute_type = Float32
        C_temp = (TC == compute_type) ? C : compute_type.(C)
        A_temp = (TA == compute_type) ? A : compute_type.(A)
        B_temp = (TB == compute_type) ? B : compute_type.(B)
        gemm!('N', 'N', compute_type(alpha), A_temp, B_temp, compute_type(beta), C_temp)
        C !== C_temp && copy!(C, C_temp)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# §4 – recgemm!  (recursive block GEMM for the Schur complement update)
#
# Computes C ← alpha·A·B + beta·C in-place for the hierarchical FullMixedPrec
# and dense AbstractMatrix targets. Replicates the exact @sync/@async parallel
# structure of recsyrk!: off-diagonal blocks are updated sequentially first,
# then the two diagonal recursive calls are launched in parallel when the
# sub-problem is large enough.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _recgemm_impl!(alpha, A, B, beta, C::AbstractMatrix, threshold; parallel)

Internal recursive GEMM for a dense `AbstractMatrix` target.
Mirrors `_recsyrk_impl!` for `AbstractMatrix`: splits `C` at a power-of-2
midpoint, updates the two off-diagonal quadrants with `_gemm_dispatch!`, then
recurses into the diagonal quadrants (in parallel when `parallel=true`).
"""
function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number,
    C::AbstractMatrix, threshold::Int; parallel::Bool
)
    n = size(C, 1)
    if n <= threshold
        _gemm_dispatch!(alpha, A, B, beta, C)
        return
    end

    n1  = 2^floor(Int, log2(n)) ÷ 2
    A1  = @view A[1:n1,     :]
    A2  = @view A[n1+1:end, :]
    B1  = @view B[:,         1:n1]
    B2  = @view B[:,         n1+1:end]
    C11 = @view C[1:n1,     1:n1]
    C12 = @view C[1:n1,     n1+1:end]
    C21 = @view C[n1+1:end, 1:n1]
    C22 = @view C[n1+1:end, n1+1:end]

    # Off-diagonal quadrants first (sequential), then diagonal in parallel
    _gemm_dispatch!(alpha, A1, B2, beta, C12)   # C12 ← alpha·A1·B2 + beta·C12
    _gemm_dispatch!(alpha, A2, B1, beta, C21)   # C21 ← alpha·A2·B1 + beta·C21

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C11, threshold, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C22, threshold, parallel=false)
    end
end

"""
    _recgemm_impl!(alpha, A, B, beta, C::FullMixedPrec; parallel)

Internal recursive GEMM for a `FullMixedPrec` target. Mirrors the
`_recsyrk_impl!` overload for `SymmMixedPrec`: descends into `C`'s own block
hierarchy, dispatching off-diagonal updates with `_gemm_dispatch!` and recursing
into the diagonal `FullMixedPrec` sub-blocks (in parallel when `parallel=true`).
"""
function _recgemm_impl!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number,
    C::FullMixedPrec; parallel::Bool
)
    if C.BaseCase !== nothing
        recgemm!(alpha, A, B, beta, C.BaseCase, 4096)
        return
    end

    n1 = size(C.A11, 1)
    A1 = @view A[1:n1,     :]
    A2 = @view A[n1+1:end, :]
    B1 = @view B[:,         1:n1]
    B2 = @view B[:,         n1+1:end]

    # Off-diagonal panels of C first (sequential)
    _gemm_dispatch!(alpha, A1, B2, beta, C.A12)  # C.A12 ← alpha·A1·B2 + beta·C.A12
    _gemm_dispatch!(alpha, A2, B1, beta, C.A21)  # C.A21 ← alpha·A2·B1 + beta·C.A21

    if parallel
        @sync begin
            @async _recgemm_impl!(alpha, A1, B1, beta, C.A11, parallel=false)
            @async _recgemm_impl!(alpha, A2, B2, beta, C.A22, parallel=false)
        end
    else
        _recgemm_impl!(alpha, A1, B1, beta, C.A11, parallel=false)
        _recgemm_impl!(alpha, A2, B2, beta, C.A22, parallel=false)
    end
end

"""
    recgemm!(alpha, A, B, beta, C::FullMixedPrec)

Performs an in-place, nested recursive block GEMM `C ← alpha·A·B + beta·C`
on a `FullMixedPrec` mixed-precision matrix structure. Falls back to the dense
`AbstractMatrix` path at the base case.
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number, C::FullMixedPrec
)
    if C.BaseCase !== nothing
        recgemm!(alpha, A, B, beta, C.BaseCase)
        return
    end
    n_subproblem    = size(C.A11, 1)
    should_parallel = n_subproblem > PARALLEL_THRESHOLD
    _recgemm_impl!(alpha, A, B, beta, C, parallel=should_parallel)
end

"""
    recgemm!(alpha, A, B, beta, C::AbstractMatrix, threshold=256)

Performs an in-place, nested recursive block GEMM `C ← alpha·A·B + beta·C`
on a dense GPU matrix `C`. Recurses until blocks reach `threshold`, at which
point `_gemm_dispatch!` fires the appropriate cuBLAS kernel.
"""
function recgemm!(
    alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number,
    C::AbstractMatrix, threshold::Int=256
)
    should_parallel = size(C, 1) > PARALLEL_THRESHOLD
    _recgemm_impl!(alpha, A, B, beta, C, threshold, parallel=should_parallel)
end

# ═══════════════════════════════════════════════════════════════════════════════
# §5 – getrf_recursive!  (non-pivoting, in-place, nested recursive block LU)
#
# Block LU without pivoting for A = [A11 A12; A21 A22]:
#
#   Step 1. A11 → L11·U11         getrf_recursive!(A11, …)
#   Step 2. A12 ← L11⁻¹·A12      unified_rectrxm! ('L','L','N','U', …)
#   Step 3. A21 ← A21·U11⁻¹      unified_rectrxm! ('R','U','N','N', …)
#   Step 4. A22 ← A22 − A21·A12  recgemm!(-1, A21, A12, 1, A22)
#   Step 5. A22 → L22·U22         getrf_recursive!(A22, …)
#
# Two dispatch targets mirror potrf_recursive! exactly:
#   • AbstractMatrix  – dense GPU matrix; CUSOLVER fires at the leaf.
#   • FullMixedPrec   – mixed-precision hierarchy; CUSOLVER fires at BaseCase.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    getrf_recursive!(A, block_size)

Performs a non-pivoting, in-place, nested recursive block LU factorization on
the dense GPU matrix `A`. Recursion continues until the sub-block fits within
`block_size`, at which point `getrf!` (cusolverDnSgetrf / cusolverDnDgetrf) is
called as the base case.

TRSM dispatches through `unified_rectrxm!` for Float16 blocks (mixed-precision
arithmetic via the existing recursive triangular solve infrastructure) and
through the direct `trsm!` wrapper for Float32/Float64 blocks.

The Schur complement update dispatches through `recgemm!` for Float16 blocks
and through a direct `gemm!` call otherwise.
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)
    if n <= block_size
        # getrf!(A)           # CUSOLVER base case (defined in getrf.jl)
        CUSOLVER.getrf!(A)
        return
    end

    n1  = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1,     1:n1]
    A12 = @view A[1:n1,     n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Step 1 – factor A11
    getrf_recursive!(A11, block_size)

    # Step 2 – A12 ← L11⁻¹ · A12  (left, lower-triangular, unit diagonal)
    if eltype(A11) == Float16
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
    end

    # Step 3 – A21 ← A21 · U11⁻¹  (right, upper-triangular, non-unit diagonal)
    if eltype(A11) == Float16
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    # Step 4 – Schur complement  A22 ← A22 − A21 · A12
    if eltype(A21) == Float16 || eltype(A12) == Float16
        recgemm!(-1.0, A21, A12, 1.0, A22)
    else
        gemm!('N', 'N', eltype(A22)(-1.0), A21, A12, eltype(A22)(1.0), A22)
    end

    # Step 5 – factor Schur complement
    getrf_recursive!(A22, block_size)
end

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base}) -> CuMatrix{T_Base}

Reconstructs a full dense GPU matrix from a `FullMixedPrec` block hierarchy.
Off-diagonal panels are dequantized using their stored scales before assembly.
Used primarily to compute residuals after `getrf_recursive!`.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end

    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    n1  = size(C11, 1)
    n2  = size(C22, 1)
    n   = n1 + n2

    C = CuArray{T_Base}(undef, n, n)
    C[1:n1,   1:n1]   .= C11
    C[n1+1:n, n1+1:n] .= C22

    A12_dense = A.A12_scale !== nothing ?
        T_Base.(A.A12) .* T_Base(A.A12_scale) : T_Base.(A.A12)
    A21_dense = A.A21_scale !== nothing ?
        T_Base.(A.A21) .* T_Base(A.A21_scale) : T_Base.(A.A21)

    C[1:n1,   n1+1:n] .= A12_dense
    C[n1+1:n, 1:n1]   .= A21_dense

    return C
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs a non-pivoting, in-place, nested recursive block LU factorization on a
`FullMixedPrec` mixed-precision matrix structure.

Follows the same five-step block LU as the `AbstractMatrix` overload. All
operations are expressed through the mixed-precision infrastructure so that
precision boundaries are respected at every recursive level:

  Step 1. `getrf_recursive!(A.A11)`                     — recurse (or BaseCase → CUSOLVER)
  Step 2. `unified_rectrxm!(…, lower_tri_mixed(A.A11), A.A12)` — solve for U12
  Step 3. `unified_rectrxm!(…, upper_tri_mixed(A.A11), A.A21)` — solve for L21
  Step 4. `recgemm!(-1, A.A21, A.A12, 1, A.A22)`       — Schur complement
  Step 5. `getrf_recursive!(A.A22)`                     — recurse on Schur complement

`lower_tri_mixed` / `upper_tri_mixed` produce zero-copy `TriMixedPrec` aliases of
the freshly factored `A.A11`, routing `A.A21` → L21 panel and `A.A12` → U12
panel into the `OffDiag` field that `unified_rec_mixed` already reads.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Step 1 – factor A11
    getrf_recursive!(A.A11)

    # Step 2 – A12 ← L11⁻¹ · A12  (left, lower-triangular, unit diagonal)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', lower_tri_mixed(A.A11), A.A12)

    # Step 3 – A21 ← A21 · U11⁻¹  (right, upper-triangular, non-unit diagonal)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', upper_tri_mixed(A.A11), A.A21)

    # Step 4 – Schur complement  A22 ← A22 − A21 · A12
    recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)

    # Step 5 – factor Schur complement
    getrf_recursive!(A.A22)
end