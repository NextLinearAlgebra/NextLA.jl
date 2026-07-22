module RecursiveLU

using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra: I, norm

export lu_recursive!, lu_residual

# ── tunables ──────────────────────────────────────────────────────────────────

"""Default tile size at which we hand off to the CUSOLVER single-precision kernel."""
const DEFAULT_BASE_SIZE = 512

# ── public API ────────────────────────────────────────────────────────────────

"""
    lu_recursive!(A::CuMatrix{Float64}; base_size::Int = $DEFAULT_BASE_SIZE) -> A

Non-pivoting, in-place, nested recursive, mixed-precision LU factorization.

# Arguments
- `A`          : square `CuMatrix{Float64}` — overwritten with the packed L/U factors.
- `base_size`  : block size at which recursion bottoms out to the CUSOLVER Float32 kernel.

# Returns
`A` itself (mutated in-place).

# Warnings
- No pivoting: the routine diverges or produces inaccurate results if any leading
  principal submatrix is singular or near-singular.
- Singularity in the base case emits a `@warn` but does not throw.
"""
function lu_recursive!(A::CuMatrix{Float64}; base_size::Int = DEFAULT_BASE_SIZE)
    m, n = size(A)
    m == n || throw(DimensionMismatch("lu_recursive! requires a square matrix, got $m×$n"))
    _lu_rec!(A, base_size)
    return A
end

# ── internal recursive kernel ─────────────────────────────────────────────────

@inline function _lu_rec!(A::CuMatrix{Float64}, bs::Int)
    n = size(A, 1)

    # ── Base case: mixed-precision CUSOLVER ────────────────────────────────
    if n <= bs
        _lu_base_cusolver!(A)
        return
    end

    # ── Recursive case ─────────────────────────────────────────────────────
    m = n >>> 1          # split point (bit-shift: fast integer halving)

    #  Subblock views — zero-copy, all on-device
    A11 = view(A,   1:m,   1:m)
    A12 = view(A,   1:m, m+1:n)
    A21 = view(A, m+1:n,   1:m)
    A22 = view(A, m+1:n, m+1:n)

    # Step 1: factor top-left block
    _lu_rec!(A11, bs)

    # Step 2: U12 ← L11⁻¹ A12
    #   Solve  L11 · X = A12,  X stored in A12
    #   side='L', uplo='L', trans='N', diag='U' (unit diagonal)
    CUBLAS.trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)

    # Step 3: L21 ← A21 · U11⁻¹
    #   Solve  X · U11 = A21,  X stored in A21
    #   side='R', uplo='U', trans='N', diag='N' (non-unit diagonal)
    CUBLAS.trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)

    # Step 4: Schur complement  A22 ← A22 − L21 · U12  (Float64 DGEMM)
    CUBLAS.gemm!('N', 'N', -1.0, A21, A12, 1.0, A22)

    # Step 5: factor bottom-right Schur complement
    _lu_rec!(A22, bs)

    return
end

# ── base case: Float32 CUSOLVER ───────────────────────────────────────────────

"""
    _lu_base_cusolver!(A::CuMatrix{Float64})

Mixed-precision base case.

1. Allocate a Float32 shadow of `A` on the GPU.
2. Call `CUSOLVER.getrf!` (cusolverDnSgetrf) — single-precision, in-place.
3. Cast the factored Float32 result back into `A` (Float64).
Pivot indices from getrf! are discarded (non-pivoting contract).
"""
function _lu_base_cusolver!(A::CuMatrix{Float64})
    # ── downcast ──────────────────────────────────────────────────────────
    A32 = CUDA.CuMatrix{Float32}(A)          # synchronous copy + convert on GPU

    # ── CUSOLVER single-precision LU ──────────────────────────────────────
    #   getrf!(A) → (A, ipiv, info)  — modifies A32 in-place
    _, _, info = CUSOLVER.getrf!(A32)

    # ── singularity check (async-friendly: info lives on GPU) ─────────────
    info_host = Array(info)[]                # scalar — pull to CPU for branch
    if info_host != 0
        @warn "Singular block in CUSOLVER base case (info=$info_host); results may be inaccurate."
    end

    # ── upcast back to Float64 ────────────────────────────────────────────
    copyto!(A, CUDA.CuMatrix{Float64}(A32))

    return
end

# ── utility: reconstruction residual (for testing) ───────────────────────────

"""
    lu_residual(A_factored::CuMatrix{Float64}, A_orig::Matrix{Float64}) -> Float64

Compute the relative Frobenius residual ‖LU − A_orig‖_F / ‖A_orig‖_F
from the packed in-place factorization stored in `A_factored`.
"""
function lu_residual(A_factored::CuMatrix{Float64}, A_orig::Matrix{Float64})
    F = Array(A_factored)
    n = size(F, 1)
    L = tril(F, -1) + Matrix{Float64}(I, n, n)   # restore implicit unit diagonal
    U = triu(F)
    return norm(L * U - A_orig) / norm(A_orig)
end

end # module RecursiveLU

# ── standalone test driver ─────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    using .RecursiveLU
    using CUDA
    using LinearAlgebra: norm

    println("CUDA device: ", CUDA.name(CUDA.device()))

    for n in [256, 512, 1024, 2048, 4096]
        # Build a diagonally dominant random matrix (ensures non-pivoting stability)
        A_cpu = randn(Float64, n, n)
        A_cpu .+= n .* I(n)                  # diagonal dominance
        A_gpu = CuMatrix{Float64}(A_cpu)

        # Warm-up on small size, then benchmark
        if n == 256
            lu_recursive!(copy(A_gpu))       # warm-up JIT
        end

        t = @elapsed begin
            lu_recursive!(A_gpu)
            CUDA.synchronize()
        end

        resid = lu_residual(A_gpu, A_cpu)
        @printf "n=%4d | time=%7.3f s | relative residual=%.3e\n" n t resid
    end
end