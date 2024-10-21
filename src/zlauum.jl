using LinearAlgebra

include("zlauu2.jl")  # Ensure this file correctly defines the zlauu2 function

"""
    zlauum(uplo::Char, n::Int, a::AbstractMatrix{T}, lda::Int, block_size::Int)

Computes the product U * U' or L' * L, where the triangular factor U or L is 
stored in the upper or lower triangular part of the array A. 

If UPLO = 'U' or 'u', then the upper triangle of the result is stored, 
overwriting the factor U in A. If UPLO = 'L' or 'l', then the lower triangle 
of the result is stored, overwriting the factor L in A.

Arguments:
- `uplo`: Specifies whether the triangular factor stored in A is upper or lower triangular.
           Must be 'U' for upper triangular or 'L' for lower triangular.
- `n`: The order of the triangular factor U or L. Must be non-negative.
- `a`: On entry, the triangular factor U or L stored in a matrix. 
       On exit, the upper or lower triangle of A is overwritten with the result.
- `lda`: The leading dimension of the array A. Must be at least max(1, N).
- `block_size`: The block size to be used in the blocked algorithm.

Returns:
- `info`: An integer value indicating the success or failure of the operation. 
          Returns 0 if successful, or a negative integer indicating the argument that caused the error.
"""
function zlauum(uplo::Char, n::Int, a::AbstractMatrix{T}, lda::Int, block_size::Int) where T
    # Validate inputs
    if !(uplo in ['U', 'L'])
        return -1  # INFO = -1: illegal value for UPLO
    end
    if n < 0
        return -2  # INFO = -2: illegal value for N
    end
    if lda < max(1, n)
        return -4  # INFO = -4: illegal value for LDA
    end

    if n == 0
        return 0  # Quick return if n is zero
    end

    # Determine the block size
    block_size = min(block_size, n)

    if block_size <= 1 || block_size >= n
        zlauu2(uplo, n, a, lda)  # Call unblocked version
        return 0
    end

    if uplo == 'U'
        compute_upper(n, block_size, a, lda)
    else
        compute_lower(n, block_size, a, lda)
    end
    return 0
end


function compute_upper(n::Int, block_size::Int, a::AbstractMatrix{T}, lda::Int) where T
    # Iterate over the blocks, similar to the Fortran loop
    for i in 1:block_size:n
        ib = min(block_size, n - i + 1)  # Current block size

        # DTRMM equivalent: A(1:i-1, i) += A(1:i-1, i:i + ib - 1) * U
        A[1:i-1, i:i+ib-1] = A[1:i-1, i:i+ib-1] * A[i:i+ib-1, i:i+ib-1]'

        # Step 1: Compute the product U * U'
        zlauu2('U', ib, view(a, i:i + ib - 1, i:i + ib - 1), lda)

        # Step 2: Update the upper triangular part
        if i + ib <= n  # Check if there's a block to update
            # DGEMM equivalent: Update A(1:i-1, i:i + ib - 1)
            a[1:i - 1, i:i + ib - 1] .+= a[1:i - 1, i + ib:n] * a[i:i + ib - 1, i + ib:n]'

            # DSYRK equivalent: Update A(i:i + ib - 1, i:i + ib - 1)
            # a[i:i + ib - 1, i:i + ib - 1] .+= a[i:i + ib - 1, i + ib:n] * a[i:i + ib - 1, i + ib:n]'
            product_matrix = a[i:i + ib - 1, i + ib:n] * a[i:i + ib - 1, i + ib:n]'

            # Update only the upper triangular part
            for j in 1:ib
                for k in j:ib  # Only update the upper triangular part
                    a[i + j - 1, i + k - 1] += product_matrix[j, k]
                end
            end
        end
    end
end




function compute_lower(n::Int, block_size::Int, a::AbstractMatrix{T}, lda::Int) where T
    # Iterate over the blocks, similar to the Fortran loop but reverse the order
    for i in 1:block_size:n
        ib = min(block_size, n - i + 1)  # Current block size

        # DTRMM equivalent: A(i:i+ib-1, 1:i-1) += A(i:i+ib-1, i:i+ib-1)' * A(i:i+ib-1, 1:i-1)
        a[i:i + ib - 1, 1:i - 1] = a[i:i + ib - 1, i:i + ib - 1]' * a[i:i + ib - 1, 1:i - 1]



        # Step 1: Compute the product L' * L for the current block (L' is transposed)
        zlauu2('L', ib, view(a, i:i + ib - 1, i:i + ib - 1), lda)

        # Step 2: Update the lower triangular part for the remaining blocks
        if i + ib <= n  # Check if there's a block below to update
            # DGEMM equivalent: A(i+ib:n, 1:i-1) += A(i+ib:n, i:i+ib-1)' * A(i:i+ib-1, 1:i-1)
            a[i:i + ib - 1, 1:i - 1] .+= a[i + ib:n, i:i + ib - 1]' * a[i + ib:n, 1:i - 1]

            # DSYRK equivalent: Update A(i+ib:n, i:i+ib-1) to store the result of L' * L
            product_matrix = a[i + ib:n, i:i + ib - 1]' * a[i + ib:n, i:i + ib - 1]

            # Update only the lower triangular part for A(i:i+ib-1, i:i+ib-1)
            for j in 1:ib
                for k in 1:j  # Only update the lower triangular part (j >= k)
                    a[i + j - 1, i + k - 1] += product_matrix[k, j]
                end
            end
        end
    end
end
