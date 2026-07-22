using LinearAlgebra
using CUDA
using CUDA.CUSOLVER
using StochasticRounding

abstract type AbstractMixedPrec{T} <: AbstractMatrix{T} end

struct TransposedMixedPrec{T, M <: AbstractMixedPrec{T}} <: AbstractMixedPrec{T}
    parent::M
end

include("fullmixedprec.jl")   # FullMixedPrec struct + constructor (this file, §1)
include("recgemm.jl")         # recgemm_sub! (this file, §2)
include("rectrxm.jl")
include("recsyrk.jl")
# include("wrappers.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# §0 – AbstractMixedPrec parent (assumed defined upstream; listed for reference)
# ═══════════════════════════════════════════════════════════════════════════════
# abstract type AbstractMixedPrec{T_Base} end

# ═══════════════════════════════════════════════════════════════════════════════
# §1 – FullMixedPrec  (full dense square matrices)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FullMixedPrec{T_Base} <: AbstractMixedPrec{T_Base}

A hierarchical, recursive mixed-precision data structure that maps to general
(full) dense square matrices. It partitions the matrix into two recursive
diagonal blocks (`A11`, `A22`) and two rectangular off-diagonal blocks (`A12`
upper, `A21` lower), each stored at an independently chosen precision.

After an in-place LU factorization with `getrf_recursive!`, the structure
stores the packed L/U factors:
  • The lower triangle of `BaseCase` / `A11` / `A22` holds L (unit diagonal implicit).
  • The upper triangle holds U (diagonal included).
  • `A21` holds the L off-diagonal panels; `A12` holds the U off-diagonal panels.
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

Uses a base-2 recursive splitting scheme:
  • Off-diagonal blocks (`A12`, `A21`) at the current level are stored at
    `precisions[1]`.
  • Diagonal blocks recurse with `precisions[2:end]`.
  • The innermost base-case block is stored at `precisions[end]` (= `T_Base`).

To prevent overflow during `Float16` conversion, dynamic per-block quantization
detects values exceeding `65504.0f0`, computes a scaling factor, and applies
clamping — identical to the logic in `SymmMixedPrec` and `TriMixedPrec`.
"""
function FullMixedPrec(
    A::AbstractMatrix;
    precisions::Vector{DataType}
)
    FP16_MAX_VAL = 65504.0f0
    n = size(A, 1)
    @assert n == size(A, 2) "FullMixedPrec requires a square matrix, got $(size(A))"

    # ── Base case: single precision level ─────────────────────────────────────
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
            base_matrix = (eltype(A) == T_Base) ? A : (tmp = similar(A, T_Base, size(A)); tmp .= A; tmp)
            base_scale  = nothing
        end

        return FullMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            nothing, nothing, base_scale, base_matrix, (n, n)
        )
    end

    # ── Recursive case ─────────────────────────────────────────────────────────
    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    T_OffDiag          = precisions[1]
    remaining_precs    = precisions[2:end]
    T_Final_Base       = precisions[end]

    # Recurse on diagonal blocks
    A11 = FullMixedPrec(view(A, 1:mid,     1:mid);     precisions=remaining_precs)
    A22 = FullMixedPrec(view(A, mid+1:n,   mid+1:n);   precisions=remaining_precs)

    # Helper: quantize a rectangular view to T_OffDiag with Float16 overflow guard
    function _quantize_offdiag(blk)
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

    A12_mat, A12_scale = _quantize_offdiag(view(A, 1:mid,     mid+1:n))
    A21_mat, A21_scale = _quantize_offdiag(view(A, mid+1:n,   1:mid))

    return FullMixedPrec{T_Final_Base}(
        A11, A22, A12_mat, A21_mat,
        A12_scale, A21_scale, nothing, nothing, (n, n)
    )
end

# ── Base interface ─────────────────────────────────────────────────────────────

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
    elseif i <= mid && j > mid           # upper off-diagonal → A12
        v = T_Base(A.A12[i, j - mid])
        return A.A12_scale !== nothing ? v * A.A12_scale : v
    else                                  # lower off-diagonal → A21
        v = T_Base(A.A21[i - mid, j])
        return A.A21_scale !== nothing ? v * A.A21_scale : v
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# §2 – Triangular extraction helpers
#       After getrf_recursive!(A::FullMixedPrec), produce TriMixedPrec views
#       of the packed L and U factors without any data copy.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    lower_tri_mixed(A::FullMixedPrec{T_Base}) -> TriMixedPrec{T_Base}

Returns a `TriMixedPrec` (uplo='L') that aliases the lower-triangular (L) factor
stored inside the packed LU factorization of `A`. Zero-copy: all sub-blocks are
shared views into `A`.
"""
function lower_tri_mixed(A::FullMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        # BaseCase holds packed L+U; TriMixedPrec with uplo='L' exposes L part.
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing, nothing,
            A.base_scale, A.BaseCase, 'L', A.sz
        )
    end
    return TriMixedPrec{T_Base}(
        lower_tri_mixed(A.A11),
        lower_tri_mixed(A.A22),
        A.A21,          # lower off-diagonal = L21 panel
        A.A21_scale,
        nothing, nothing, 'L', A.sz
    )
end

"""
    upper_tri_mixed(A::FullMixedPrec{T_Base}) -> TriMixedPrec{T_Base}

Returns a `TriMixedPrec` (uplo='U') that aliases the upper-triangular (U) factor
stored inside the packed LU factorization of `A`. Zero-copy.
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
        A.A12,          # upper off-diagonal = U12 panel
        A.A12_scale,
        nothing, nothing, 'U', A.sz
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# §3 – recgemm_sub!  (recursive Schur-complement update: C -= A * B)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    recgemm_sub!(C::FullMixedPrec, A::AbstractMatrix, B::AbstractMatrix)

In-place recursive block rank-k update  `C ← C − A · B`  targeting the
`FullMixedPrec` structure `C`. The recursion follows `C`'s own block hierarchy:

    [C11  C12]   [C11  C12]   [A1]               [A1·B1   A1·B2]
    [C21  C22] ← [C21  C22] − [A2] · [B1  B2] = [A2·B1   A2·B2]

Diagonal blocks (`C11`, `C22`) are handled by recursive calls; off-diagonal
blocks (`C12`, `C21`) are updated with a single `GEMM_SUB!` call, which
dispatches to the appropriate hardware kernel (cuBLAS GEMM or gemmEx for
Float16 inputs).

This function is the counterpart of `recsyrk!` for the LU Schur complement,
replacing the symmetric rank-k update used in Cholesky.
"""
function recgemm_sub!(C::FullMixedPrec, A::AbstractMatrix, B::AbstractMatrix)
    if C.BaseCase !== nothing
        # Dense base: single GEMM_SUB! call, handles Float16 via gemmEx internally
        GEMM_SUB!(C.BaseCase, A, B)
        return
    end

    n1  = size(C.A11, 1)
    # Row-split A; column-split B to match C's block structure
    A1  = @view A[1:n1,      :]
    A2  = @view A[n1+1:end,  :]
    B1  = @view B[:,          1:n1]
    B2  = @view B[:,          n1+1:end]

    # Off-diagonal updates (AbstractMatrix targets → single kernel call each)
    GEMM_SUB!(C.A12, A1, B2)      # C12 -= A1 * B2
    GEMM_SUB!(C.A21, A2, B1)      # C21 -= A2 * B1

    # Diagonal updates (recurse into FullMixedPrec)
    recgemm_sub!(C.A11, A1, B1)   # C11 -= A1 * B1
    recgemm_sub!(C.A22, A2, B2)   # C22 -= A2 * B2
end

# ═══════════════════════════════════════════════════════════════════════════════
# §4 – getrf_recursive!  (non-pivoting, in-place, nested recursive block LU)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Recursive block LU decomposition without pivoting.
#
# For A partitioned at midpoint n1:
#
#   [A11  A12]   =   [L11   0 ] [U11  U12]
#   [A21  A22]       [L21  L22] [ 0   U22]
#
# which yields the five in-place steps:
#
#   1.  getrf_recursive!(A11)                   — factor top-left block
#   2.  L11 · U12 = A12  →  TRSM left-lower-unit    →  A12 ← U12
#   3.  L21 · U11 = A21  →  TRSM right-upper-nonunit →  A21 ← L21
#   4.  A22 ← A22 − L21 · U12                   — Schur complement (recgemm_sub!)
#   5.  getrf_recursive!(A22)                   — factor Schur complement
#
# Two dispatch targets:
#   • AbstractMatrix  – uniform-precision GPU matrices; CUSOLVER is the base case.
#   • FullMixedPrec   – hierarchical mixed-precision structure; CUSOLVER fires
#                       when A.BaseCase is reached (at the innermost level).
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dense AbstractMatrix path ─────────────────────────────────────────────────

"""
    getrf_recursive!(A::AbstractMatrix, block_size::Int=256)

Non-pivoting, in-place, nested recursive block LU factorization for a uniform-
precision GPU matrix `A`.

**Base case** (`n ≤ block_size`): delegates to CUSOLVER (`cusolverDnSgetrf` /
`cusolverDnDgetrf`). For element types other than Float32/Float64 (e.g., Float16),
the block is first cast to Float32, factored in single precision, then cast back —
matching the mixed-precision spirit of the overall framework.

**Recursive case**: performs the standard five-step block LU (see module header),
dispatching TRSM through `unified_rectrxm!` for Float16 blocks and through the
direct `trsm!` wrapper otherwise.
"""
function getrf_recursive!(A::AbstractMatrix, block_size::Int=256)
    n = size(A, 1)
    @assert size(A, 2) == n "getrf_recursive! requires a square matrix"

    # ── Base case: CUSOLVER ────────────────────────────────────────────────────
    if n <= block_size
        T = eltype(A)
        if T in (Float32, Float64)
            _, _, info = CUSOLVER.getrf!(A)
            CUDA.@allowscalar info[1] != 0 &&
                @warn "getrf_recursive! base case: singular block (CUSOLVER info=$(CUDA.@allowscalar info[1]))"
        else
            # Mixed-precision: downcast to Float32, factor, upcast back
            A32 = CuMatrix{Float32}(A)
            _, _, info = CUSOLVER.getrf!(A32)
            CUDA.@allowscalar info[1] != 0 &&
                @warn "getrf_recursive! base case: singular block (CUSOLVER info=$(CUDA.@allowscalar info[1]))"
            copyto!(A, CuMatrix{T}(A32))
        end
        return
    end

    # ── Recursive case ─────────────────────────────────────────────────────────
    n1  = 2^floor(Int, log2(n)) ÷ 2     # power-of-2 split (matches SymmMixedPrec/TriMixedPrec)

    A11 = @view A[1:n1,     1:n1]
    A12 = @view A[1:n1,     n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Step 1 – factor A11
    getrf_recursive!(A11, block_size)

    # Step 2 – solve L11 · U12 = A12  (left, lower-triangular, unit diagonal)
    if eltype(A11) == Float16
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
    end

    # Step 3 – solve L21 · U11 = A21, i.e. X · U11 = A21  (right, upper, non-unit)
    if eltype(A11) == Float16
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    # Step 4 – Schur complement  A22 ← A22 − L21 · U12
    if eltype(A21) == Float16 || eltype(A12) == Float16
        GEMM_SUB!(A22, A21, A12)
    else
        gemm!('N', 'N', eltype(A22)(-1.0), A21, A12, eltype(A22)(1.0), A22)
    end

    # Step 5 – factor Schur complement
    getrf_recursive!(A22, block_size)
end

# ── FullMixedPrec path ────────────────────────────────────────────────────────

"""
    getrf_recursive!(A::FullMixedPrec)

Non-pivoting, in-place, nested recursive block LU factorization on a
`FullMixedPrec` mixed-precision matrix structure.

The five-step block LU algorithm follows exactly the same structure as the dense
`AbstractMatrix` overload, but all operations are expressed through the mixed-
precision infrastructure so that precision boundaries are respected at every
recursive level:

  1. `getrf_recursive!(A.A11)` — recurse on the top-left block (or hit CUSOLVER
     at the `BaseCase`).
  2. `unified_rectrxm!('L','L','N','U', …, lower_tri_mixed(A.A11), A.A12)` —
     forward substitution for U12 using the (freshly factored) mixed-precision
     lower-triangular L11.
  3. `unified_rectrxm!('R','U','N','N', …, upper_tri_mixed(A.A11), A.A21)` —
     backward substitution for L21 using the upper-triangular U11.
  4. `recgemm_sub!(A.A22, A.A21, A.A12)` — recursive Schur complement update.
  5. `getrf_recursive!(A.A22)` — recurse on the Schur complement.

`lower_tri_mixed` / `upper_tri_mixed` produce zero-copy `TriMixedPrec` wrappers
of A.A11's packed factors, routing the correct off-diagonal panels (`A.A21` →
L21, `A.A12` → U12) to `unified_rectrxm!` without allocating any new storage.
"""
function getrf_recursive!(A::FullMixedPrec)
    # ── Base case: dense block, hand to dense recursive path ──────────────────
    if A.BaseCase !== nothing
        # block_size=4096: at this depth we want one more level of recursion
        # before CUSOLVER fires, consistent with potrf_recursive!(A::SymmMixedPrec)
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # ── Step 1: factor top-left diagonal block ────────────────────────────────
    getrf_recursive!(A.A11)

    # ── Step 2: solve  L11 · U12 = A12
    #   L11 extracted as TriMixedPrec (uplo='L', unit diagonal)
    #   Result written in-place to A.A12 (U12 panel).
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', lower_tri_mixed(A.A11), A.A12)

    # ── Step 3: solve  L21 · U11 = A21  ⟺  X · U11 = A21
    #   U11 extracted as TriMixedPrec (uplo='U', non-unit diagonal)
    #   Result written in-place to A.A21 (L21 panel).
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', upper_tri_mixed(A.A11), A.A21)

    # ── Step 4: Schur complement  A22 ← A22 − L21 · U12
    #   A.A21 now holds L21; A.A12 now holds U12.
    recgemm_sub!(A.A22, A.A21, A.A12)

    # ── Step 5: factor Schur complement ──────────────────────────────────────
    getrf_recursive!(A.A22)
end

# ═══════════════════════════════════════════════════════════════════════════════
# §5 – Reconstruction helper  (validation / dense-format recovery)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    reconstruct_matrix(A::FullMixedPrec{T_Base}) -> CuMatrix{T_Base}

Reconstructs a full dense matrix from a `FullMixedPrec` structure. Used
primarily for validation (e.g., computing the LU residual after factorization).
Off-diagonal blocks are dequantized using their stored scales before assembly.
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
    C[1:n1,     1:n1]     .= C11
    C[n1+1:n,   n1+1:n]   .= C22

    # Dequantize off-diagonal panels
    A12_dense = A.A12_scale !== nothing ?
        T_Base.(A.A12) .* T_Base(A.A12_scale) : T_Base.(A.A12)
    A21_dense = A.A21_scale !== nothing ?
        T_Base.(A.A21) .* T_Base(A.A21_scale) : T_Base.(A.A21)

    C[1:n1,     n1+1:n]   .= A12_dense
    C[n1+1:n,   1:n1]     .= A21_dense

    return C
end

# ═══════════════════════════════════════════════════════════════════════════════
# §6 – Standalone test driver
# ═══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    using Printf
    using LinearAlgebra: I, norm, tril, triu

    println("CUDA device : ", CUDA.name(CUDA.device()))
    println()

    # ── Dense AbstractMatrix path ──────────────────────────────────────────────
    println("── Dense AbstractMatrix path ─────────────────────────────────────")
    for n in [256, 1024, 4096]
        A_cpu = randn(Float64, n, n); A_cpu .+= n .* I(n)   # diagonal dominant → stable
        A_gpu = CuMatrix{Float64}(A_cpu)
        n == 256 && getrf_recursive!(copy(A_gpu))            # JIT warm-up

        t = @elapsed (getrf_recursive!(A_gpu); CUDA.synchronize())

        F   = Array(A_gpu)
        L   = tril(F, -1) + Matrix{Float64}(I, n, n)
        U   = triu(F)
        res = norm(L * U - A_cpu) / norm(A_cpu)
        @printf "n=%4d | t=%7.3f s | ‖LU−A‖/‖A‖ = %.3e\n" n t res
    end
    println()

    # ── FullMixedPrec path ─────────────────────────────────────────────────────
    println("── FullMixedPrec path ────────────────────────────────────────────")
    n          = 2048
    A_cpu      = randn(Float32, n, n); A_cpu .+= n .* I(n)
    # Three-level hierarchy: Float16 outermost panels → Float32 inner panels → Float32 base
    precisions = [Float16, Float32, Float32]
    A_mp       = FullMixedPrec(CuMatrix{Float32}(A_cpu); precisions=precisions)

    getrf_recursive!(A_mp)           # in-place LU on the mixed-precision structure

    # Reconstruct packed LU to dense and verify
    F_mp  = Array(reconstruct_matrix(A_mp))
    L_mp  = tril(F_mp, -1) + Matrix{Float32}(I, n, n)
    U_mp  = triu(F_mp)
    res_mp = norm(L_mp * U_mp - A_cpu) / norm(A_cpu)
    @printf "FullMixedPrec n=%d | precisions=%s | ‖LU−A‖/‖A‖ = %.3e\n" n string(precisions) res_mp
end