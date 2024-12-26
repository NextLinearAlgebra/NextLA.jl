using LinearAlgebra
include("trsm.jl")
"""
    rectrsm!(side, A, n, B, k=1; uplo="L", transpose="N", threshold=128)

Recursive triangular solve matrix function `TRSM`. Solves one of the matrix equations:

    op(A) * X = B  (when `side = 'L'`)
    X * op(A) = B  (when `side = 'R'`)

where:
- `A` is an `n x n` upper or lower triangular matrix (triangular form specified by `uplo`).
- `B` is the `n x k` right-hand side matrix (defaulting to `n x 1`).
- `op(A)` can be `A` itself or its transpose, based on the `transpose` parameter.
- The matrix `X`, which holds the solution, overwrites `B`.

# Parameters
- `side` (Character): `'L'` for left-side multiplication (`op(A) * X = B`) or `'R'` for right-side (`X * op(A) = B`).
- `A` (AbstractMatrix{T}): Triangular matrix of size `n x nither upper or lower, controlled by `uplo`).
- `n` (Integer): Size of `A`.
- `B` (AbstractMatrix{T}): Right-hand side matrix of size `n x k` (defaults to `n x 1`).
- `k` (Integer, optional): Number of columns in `B` (i.e., different systems to solve in parallel); defaults to 1.
- `uplo` (Character, optional): Specifies if `A` is `'U'` (upper) or `'L'` (lower triangular); default is `'L'`.
- `transpose` (Character, optional): Specifies if `A` should be transposed; default is `'N'` (no transpose).
- `threshold` (Integer, optional): Minimum block size for recursive division; below this size, it uses a direct `BLAS.trsm!` call.

# Details
For matrix sizes `n` larger than `threshold`, this function partitions `A` and `B`, solving smaller triangular systems recursively
and updating `B` in-place using `GEMM` operations. This recursive structure leverages matrix blocking to improve parallelism and data reuse.

# Example
    rectrsm!('L', A, n, B; uplo="L", transpose="N", threshold=128)
"""
function rectrsm!(A::AbstractMatrix{T}, n::Int, B::AbstractMatrix{T}, side::AbstractChar = 'L', k::Int=1;
                  uplo::AbstractChar='L', transpose::AbstractChar='N', threshold::Int=16) where T
    # Base case: Small matrix sizes use the non-recursive `trsm` (no BLAS).
    if n <= threshold
        trsm!(side, uplo, transpose, A, B)
        return
    end

    # Partition `A` and `B`
    mid = div(n, 2)
    A11 = view(A, 1:mid, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)
    B1 = view(B, 1:mid, :)
    B2 = view(B, mid+1:n, :)
    A21 = view(A, mid+1:n, 1:mid)  # For lower triangular
    A12 = view(A, 1:mid, mid+1:n)  # For upper triangular

    # Current focus: Left side, lower triangular
    if uplo == 'L'
        if side == 'L'
            # Step 1: Solve A11 * X1 = B1 recursively
            rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

            # Step 2: GEMM update B2 = B2 - A21 * B1
            B2 .-= A21 * B1

            # Step 3: Solve A22 * X2 = B2 recursively
            rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        elseif side == 'R'  # Right side
            #NOT YET CORRECTLY IMPLEMENTED
            # # Step 1: Solve A11 * X1 = B1 recursively
            # rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

            # # Step 2: GEMM update B1 = B1 - B2*A21
            # B2 .-= B1 * A21'

            # # Step 3: Solve A22 * X2 = B2 recursively
            # rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)
        end

    elseif uplo == 'U'  # Upper triangular cases
        if side == 'L'  # Left side
            #NOT YET CORRECTLY IMPLEMENTED
            # # Step 1: Solve A11 * X1 = B1 recursively
            # rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

            # # Step 2: GEMM update B1 = B1 - A12*B2
            # B2 .-= A12 * B1

            # # Step 3: Solve A22 * X2 = B2 recursively
            # rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        elseif side == 'R'  # Right side
            #NOT YET CORRECTLY IMPLEMENTED
            # # Step 1: Solve A11 * X1 = B1 recursively
            # rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

            # # Step 2: GEMM update B2 = B2 - B1 * A12
            # B2 .-= B1 * A12'

            # # Step 3: Solve A22 * X2 = B2 recursively
            # rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)
        end
    end
end