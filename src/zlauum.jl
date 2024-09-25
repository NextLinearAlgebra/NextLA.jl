using LinearAlgebra

include("zlauu2.jl") 

"""
    zlauum(uplo::Char, n::Int, a::Array{Complex{Float64}, 2}, lda::Int)

Computes the product of a triangular matrix U (or L) with its conjugate transpose.
 
# Arguments
- `uplo`: A character indicating whether the matrix is upper ('U') or lower ('L').
- `n`: The order of the triangular matrix.
- `a`: The input/output array storing the triangular matrix.
- `lda`: The leading dimension of the array `a`.

# Returns
None; modifies `a` in place with the result of the computation.
"""
function zlauum(uplo::Char, n::Int, a::Array{Complex{Float64}, 2}, lda::Int)
    # Validate inputs
    if !(uplo in ['U', 'L'])
        throw(ArgumentError("UPLO must be 'U' (upper) or 'L' (lower)."))
    end
    if n < 0
        throw(ArgumentError("Matrix dimension N must be non-negative."))
    end
    if lda < max(1, n)
        throw(ArgumentError("Leading dimension LDA must be at least max(1, N)."))
    end
    
    if n == 0
        return
    end

    block_size = matrix_size(uplo, n, lda)

    if block_size <= 1 || block_size >= n
        zlauu2(uplo, n, a, lda)
        return
    end

    if uplo == 'U'
        compute_upper(n, block_size, a, lda)
    else
        compute_lower(n, block_size, a, lda)
    end
end

"""
    compute_upper(n::Int, block_size::Int, a::Array{Complex{Float64}, 2}, lda::Int)

Computes the product U * U' for an upper triangular matrix stored in `a`.

# Arguments
- `n`: The order of the triangular matrix.
- `block_size`: The size of the blocks for blocked matrix multiplication.
- `a`: The input/output array storing the triangular matrix.
- `lda`: The leading dimension of the array `a`.

# Returns
None; modifies `a` in place with the result of the computation.
"""
function compute_upper(n::Int, block_size::Int, a::Array{Complex{Float64}, 2}, lda::Int)
    cone = Complex(1.0, 0.0)
    for i in 1:block_size:n
        ib = min(block_size, n - i + 1)
        LinearAlgebra.BLAS.trmm!('R', 'U', 'C', 'N', i - 1, ib, cone, a[i:i + ib - 1, i:i + ib - 1], lda, a[1:i, i], lda)
        zlauu2('U', ib, a[i:i + ib - 1, i:i + ib - 1], lda)

        if i + ib <= n
            LinearAlgebra.BLAS.gemm!('N', 'C', i - 1, ib, n - i - ib + 1, cone, a[1:i + ib - 1, i + ib:n], lda, a[i:i + ib - 1, i + ib:n], lda, cone, a[1:i, i], lda)
            LinearAlgebra.BLAS.herk!('U', 'N', ib, n - i - ib + 1, 1.0, a[i:i + ib - 1, i + ib:n], lda, 1.0, a[i:i + ib - 1, i:i + ib - 1], lda)
        end
    end
end

"""
    compute_lower(n::Int, block_size::Int, a::Array{Complex{Float64}, 2}, lda::Int)

Computes the product L' * L for a lower triangular matrix stored in `a`.

# Arguments
- `n`: The order of the triangular matrix.
- `block_size`: The size of the blocks for blocked matrix multiplication.
- `a`: The input/output array storing the triangular matrix.
- `lda`: The leading dimension of the array `a`.

# Returns
None; modifies `a` in place with the result of the computation.
"""
function compute_lower(n::Int, block_size::Int, a::Array{Complex{Float64}, 2}, lda::Int)
    cone = Complex(1.0, 0.0)
    for i in 1:block_size:n
        ib = min(block_size, n - i + 1)
        LinearAlgebra.BLAS.trmm!('L', 'L', 'C', 'N', ib, i - 1, cone, a[i:i + ib - 1, i:i + ib - 1], lda, a[i, 1:i - 1], lda)
        zlauu2('L', ib, a[i:i + ib - 1, i:i + ib - 1], lda)

        if i + ib <= n
            LinearAlgebra.BLAS.gemm!('C', 'N', ib, i - 1, n - i - ib + 1, cone, a[i + ib:n, i], lda, a[i + ib, 1:i - 1], lda, cone, a[i, 1:i - 1], lda)
            LinearAlgebra.BLAS.herk!('L', 'C', ib, n - i - ib + 1, 1.0, a[i + ib:n, i], lda, 1.0, a[i:i + ib - 1, i:i + ib - 1], lda)
        end
    end
end

"""
    matrix_size(uplo::Char, n::Int, lda::Int)::Int

Determines the size of the matrix for the blocked algorithm.

# Arguments
- `uplo`: A character indicating whether the matrix is upper ('U') or lower ('L').
- `n`: The order of the triangular matrix.
- `lda`: The leading dimension of the array.

# Returns
An integer indicating the size of the blocks used for computation.
"""
function matrix_size(uplo::Char, n::Int, lda::Int)::Int
    return (n < 10) ? 64 : 128
end
