# =============================================================================
# recursive_qr.jl
#
# In-place, column-wise block-recursive QR factorization on the GPU
# (Elmroth–Gustavson "RGEQR3" recursion scheme), with CUSOLVER `geqrf!`
# as the base case and CUSOLVER `ormqr!`/`unmqr` for trailing-matrix updates.
#
# Storage is fully LAPACK-compatible. On exit:
#   * the upper triangle of A holds R,
#   * the strict lower triangle holds the Householder vectors V
#     (implicit unit diagonal),
#   * τ holds the Householder scalar factors,
# so the packed result can be consumed directly by `orgqr!` / `ormqr!`
# (or their complex `ungqr` / `unmqr` counterparts, which CUDA.jl
# dispatches to automatically by element type).
#
# Why recursion helps on GPU:
#   * The column split turns most of the flops into wide trailing-matrix
#     applications (Qᴴ · A₂), which CUSOLVER's ormqr executes as
#     compact-WY (larft + GEMM-rich larfb) kernels — high arithmetic
#     intensity, excellent SM occupancy.
#   * Panel factorizations shrink geometrically, so the low-intensity,
#     latency-bound part of the algorithm is a vanishing fraction of work.
#   * Everything stays on-device on a single stream: no host round trips,
#     no explicit synchronization needed between recursion steps.
#
# Tested pattern: CUDA.jl ≥ 4.x. The `applicable`-based shims below make
# the code robust to whether your CUDA.jl version accepts contiguous
# GPU-array views in the CUSOLVER wrappers (newer versions do; older
# versions get a cheap copy fallback).
# =============================================================================

module RecursiveQR

using LinearAlgebra
using CUDA
using CUDA.CUSOLVER

export rgeqrf!, lmulQ!, lmulQt!, explicitQ!

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

# Adjoint code for ormqr/unmqr: real → 'T', complex → 'C'.
_adj(::Type{T}) where {T<:Real}    = 'T'
_adj(::Type{T}) where {T<:Complex} = 'C'

"""
    _split(n, nb) -> n1

Width of the left recursion branch: `n/2` rounded **up** to a multiple of
`nb`, clamped to `[nb, n-1]`. Keeping panel widths at multiples of the base
block size keeps the CUSOLVER base-case calls uniformly sized and keeps
column offsets aligned, which is friendly to coalesced access.
Only called when `n > nb`, so `n1 < n` is guaranteed.
"""
@inline function _split(n::Int, nb::Int)
    half = n >> 1
    n1   = nb * cld(half, nb)      # round half up to a multiple of nb
    return max(n1, nb)
end

# --- CUSOLVER shims ----------------------------------------------------------
# The recursion hands CUSOLVER *views* (strided sub-matrices, contiguous
# sub-vectors of τ). Recent CUDA.jl accepts these directly (StridedCuArray
# unions); the fallbacks below materialize a copy only if dispatch fails,
# so correctness never depends on the CUDA.jl version.

@inline function _geqrf!(A, τv)
    if applicable(CUSOLVER.geqrf!, A, τv)
        CUSOLVER.geqrf!(A, τv)                 # writes τ in place
    else
        _, t = CUSOLVER.geqrf!(A)              # allocating fallback (tiny)
        copyto!(τv, t)
    end
    return nothing
end

@inline function _ormqr!(side::Char, trans::Char, V, τv, C)
    if applicable(CUSOLVER.ormqr!, side, trans, V, τv, C)
        CUSOLVER.ormqr!(side, trans, V, τv, C)
    else
        # Copy fallback for older wrappers with narrow signatures.
        CUSOLVER.ormqr!(side, trans, CuMatrix(V), CuVector(τv), C)
    end
    return C
end

# -----------------------------------------------------------------------------
# Core recursion (Elmroth–Gustavson RGEQR3)
# -----------------------------------------------------------------------------

"""
    _rgeqr3!(A, τ, nb)

Recursively factor `A` (a strided device view, `m ≥ n` guaranteed by the
caller) in place. Column-wise split:

    [A₁ A₂] :  1. QR(A₁)  recursively            (left n₁ columns)
               2. A₂ ← Q₁ᴴ A₂  via ormqr         (compact-WY, GEMM-rich)
               3. QR(A₂[n₁+1:m, :]) recursively  (trailing submatrix)

Householder vectors land in the strict lower triangle of the *global* A
automatically: within the trailing view, local column k starts at local
row k, i.e. global row n₁+k — exactly the LAPACK packed layout.
"""
function _rgeqr3!(A::AbstractMatrix{T}, τ::AbstractVector{T}, nb::Int) where {T}
    m, n = size(A)

    # Base case: hand the panel to CUSOLVER's Householder QR.
    if n <= nb
        _geqrf!(A, τ)
        return nothing
    end

    n1 = _split(n, nb)

    # 1. Factor the left panel recursively.
    A1 = view(A, :, 1:n1)
    τ1 = view(τ, 1:n1)
    _rgeqr3!(A1, τ1, nb)

    # 2. Apply Q₁ᴴ to the trailing columns. CUSOLVER builds the block
    #    reflector T internally and applies it as (I - V T Vᴴ)ᴴ · A₂,
    #    i.e. two large GEMM-shaped operations. This is where ~all the
    #    flops (and the speed) live.
    A2 = view(A, :, n1+1:n)
    _ormqr!('L', _adj(T), A1, τ1, A2)

    # 3. Factor the trailing submatrix (rows n1+1:m only — rows 1:n1 of
    #    A₂ are already finished entries of R).
    A22 = view(A, n1+1:m, n1+1:n)
    τ2  = view(τ, n1+1:n)
    _rgeqr3!(A22, τ2, nb)

    return nothing
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    rgeqrf!(A::StridedCuMatrix{T} [, τ::CuVector{T}]; nb=128) -> (A, τ)

In-place, column-wise block-recursive QR factorization of the `m × n`
device matrix `A`, with CUSOLVER `geqrf!` as the base case for panels of
width ≤ `nb` and CUSOLVER `ormqr!` for trailing updates.

On exit `A` and `τ` hold the factorization in LAPACK packed form
(R in the upper triangle, Householder vectors below the diagonal,
scalar factors in `τ`, `length(τ) == min(m, n)`).

Wide matrices (`n > m`) are handled by factoring the leading `m` columns
and applying `Qᴴ` to the remainder (which then holds the right block of R).

Tuning: `nb` in 128–512 is a good range on modern GPUs; larger `nb` means
fewer, fatter CUSOLVER calls (less launch latency) but a less GEMM-rich
mix. Benchmark on your card/sizes — `demo()` below gives a template.

All work is enqueued on the current CUDA stream; calls are ordered, so no
explicit synchronization is required inside the recursion.
"""
function rgeqrf!(A::StridedCuMatrix{T},
                 τ::CuVector{T} = CuVector{T}(undef, min(size(A)...));
                 nb::Integer = 128) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    m, n = size(A)
    k    = min(m, n)
    nb   = Int(nb)
    nb ≥ 1 || throw(ArgumentError("nb must be ≥ 1, got $nb"))
    length(τ) == k ||
        throw(DimensionMismatch("τ has length $(length(τ)), needs min(m,n) = $k"))

    if n <= m
        _rgeqr3!(A, τ, nb)
    else
        # Wide case: QR of the leading square block, then finish R.
        A1 = view(A, :, 1:m)
        _rgeqr3!(A1, τ, nb)
        _ormqr!('L', _adj(T), A1, τ, view(A, :, m+1:n))
    end
    return A, τ
end

"""
    lmulQt!(C, A, τ) -> C

Overwrite `C` with `Qᴴ * C`, where `Q` is held implicitly in `(A, τ)`
as produced by [`rgeqrf!`](@ref). `C` must have `size(C, 1) == size(A, 1)`.
"""
lmulQt!(C::StridedCuVecOrMat{T}, A::StridedCuMatrix{T}, τ::CuVector{T}) where {T} =
    _ormqr!('L', _adj(T), A, τ, C)

"""
    lmulQ!(C, A, τ) -> C

Overwrite `C` with `Q * C` (no adjoint), `Q` implicit in `(A, τ)`.
"""
lmulQ!(C::StridedCuVecOrMat{T}, A::StridedCuMatrix{T}, τ::CuVector{T}) where {T} =
    _ormqr!('L', 'N', A, τ, C)

"""
    explicitQ!(A, τ) -> A

Overwrite the packed factorization in `A` with the explicit thin `Q`
(`m × min(m,n)` orthonormal columns) via CUSOLVER `orgqr!`/`ungqr!`.
Destructive: R and the Householder vectors are lost — call on a copy if
you still need the packed form.
"""
explicitQ!(A::StridedCuMatrix{T}, τ::CuVector{T}) where {T} = CUSOLVER.orgqr!(A, τ)

end # module