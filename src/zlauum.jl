using LinearAlgebra

include("zlauu2.jl")  # Import the unblocked version of the matrix multiplication function (zlauu2) to use later in this computation.

"""
    zlauum(uplo::Char, n::Int, a::AbstractMatrix{T}, lda::Int, block_size::Int)

This function computes the product of a triangular matrix with its conjugate transpose. Specifically, it computes:

- `U * U'` if the triangular matrix `U` is stored in the upper part of matrix `a`.
- `L' * L` if the triangular matrix `L` is stored in the lower part of matrix `a`.

Where:
- `U'` represents the conjugate transpose of the upper triangular matrix `U`.
- `L'` represents the conjugate transpose of the lower triangular matrix `L`.

### Parameters:
- `uplo`: A character (`'U'` or `'L'`) that specifies whether the triangular matrix is stored in the upper or lower part of `a`.
  - `'U'`: Indicates that the upper triangle contains the triangular matrix `U`. The result of `U * U'` will overwrite the corresponding entries in the upper triangle of matrix `a`.
  - `'L'`: Indicates that the lower triangle contains the triangular matrix `L`. The result of `L' * L` will overwrite the corresponding entries in the lower triangle of matrix `a`.
  
- `n`: The order (size) of the triangular matrix. This must be a non-negative integer, representing the dimensions of the square matrix `a`, which is `n x n`.

- `a`: The matrix where the triangular factor `U` or `L` is stored, and where the result will be stored after computation. This matrix is modified in place, meaning its contents will change as a result of the computation.

- `lda`: The leading dimension of the array `a`. This should be at least `max(1, n)`. This parameter is important for accessing the elements of the matrix in memory correctly, particularly in scenarios where matrices may be stored in a non-contiguous fashion for performance reasons.

- `block_size`: This specifies the block size for the blocked algorithm. A blocked algorithm processes the matrix in submatrices (or blocks), improving performance on large matrices by making better use of CPU cache and reducing memory bandwidth demands.

### Returns:
- `info`: An integer indicating the success or failure of the function execution:
  - `0`: Indicates successful execution.
  - A negative integer indicates that an invalid argument was provided:
    - `-1`: Invalid value for `uplo`.
    - `-2`: Invalid value for `n`.
    - `-4`: Invalid value for `lda`.
"""
function zlauum(uplo::Char, n::Int, a::AbstractMatrix{T}, lda::Int, block_size::Int) where T
    # Validate the 'uplo' parameter to ensure it is either 'U' or 'L'
    if !(uplo in ['U', 'L'])
        return -1  # Return an error code for invalid 'uplo'
    end

    # Check if 'n' is non-negative
    if n < 0
        return -2  # Return an error code for invalid 'n'
    end

    # Validate 'lda' to ensure it meets the minimum requirement
    if lda < max(1, n)
        return -4  # Return an error code for invalid 'lda'
    end

    # If 'n' is zero, no computation is needed, so return success
    if n == 0
        return 0  # Early exit with success code
    end

    # Adjust block_size to ensure it does not exceed the size of the matrix
    block_size = min(block_size, n)

    # If block_size is less than or equal to 1, or greater than or equal to n, use the unblocked version
    if block_size <= 1 || block_size >= n
        zlauu2(uplo, n, a, lda)  # Call the unblocked computation
        return 0  # Return success code
    end

    # Call the appropriate computation based on whether the upper or lower triangular matrix is specified
    if uplo == 'U'
        compute_upper(n, block_size, a, lda)  # Compute for upper triangular matrix
    else
        compute_lower(n, block_size, a, lda)  # Compute for lower triangular matrix
    end

    return 0  # Return success code after completing the computation
end

"""
    compute_upper(n, block_size, a, lda)

This function performs the blocked computation of U * U' for an upper triangular matrix `U`.
The computation is carried out in parallel to improve performance on large matrices.

### Parameters:
- `n`: The size of the matrix, which is also the order of the triangular matrix `U`.
- `block_size`: The size of the blocks to be processed in each iteration. This allows for better cache usage and performance.
- `a`: The matrix that contains the upper triangular part `U`, and where the results will be stored.
- `lda`: The leading dimension of the matrix `a`.

This function modifies the matrix `a` in place.
"""
function compute_upper(n::Int, block_size::Int, a::AbstractMatrix{T}, lda::Int) where T
    Threads.@threads for i in 1:block_size:n  # Parallelize the outer loop over blocks
        ib = min(block_size, n - i + 1)  # Determine the actual block size for this iteration

        # Perform a triangular matrix multiplication (equivalent to DTRMM)
        # Update the upper triangle of the matrix using the current block
        view(a, 1:i-1, i:i+ib-1) .= view(a, 1:i-1, i:i+ib-1) * view(a, i:i+ib-1, i:i+ib-1)'

        # Compute the product U * U' for the current block using the zlauu2 function
        # zlauu2('U', ib, view(a, i:i+ib-1, i:i+ib-1), lda)
        U = view(a, i:i+ib-1, i:i+ib-1)  # Extract the block U
        U_Ut = U * adjoint(U)  # Use adjoint for complex matrices
        # Only update the upper triangular part of the matrix
        for j in 1:ib
            for k in j:ib
                a[i + j - 1, i + k - 1] = U_Ut[j, k]
            end
        end
        
        
        # Check if there are additional blocks to process
        if i + ib <= n
            # Perform matrix-matrix multiplication (equivalent to DGEMM)
            view(a, 1:i-1, i:i+ib-1) .+= view(a, 1:i-1, i+ib:n) * view(a, i:i+ib-1, i+ib:n)'

            # Perform symmetric rank-k update (equivalent to DSYRK)
            product_matrix = view(a, i:i+ib-1, i+ib:n) * view(a, i:i+ib-1, i+ib:n)'
            for j in 1:ib  # Iterate over the rows of the current block
                for k in j:ib  # Iterate over the columns of the current block
                    @inbounds a[i + j - 1, i + k - 1] += product_matrix[j, k]  # Update the result matrix
                end
            end
        end
    end
end

"""
    compute_lower(n, block_size, a, lda)

This function performs the blocked computation of L' * L for a lower triangular matrix `L`.
The computation is carried out in parallel to improve performance on large matrices.

### Parameters:
- `n`: The size of the matrix, which is also the order of the triangular matrix `L`.
- `block_size`: The size of the blocks to be processed in each iteration. This allows for better cache usage and performance.
- `a`: The matrix that contains the lower triangular part `L`, and where the results will be stored.
- `lda`: The leading dimension of the matrix `a`.

This function modifies the matrix `a` in place.
"""
function compute_lower(n::Int, block_size::Int, a::AbstractMatrix{T}, lda::Int) where T
    Threads.@threads for i in 1:block_size:n  # Parallelize the outer loop over blocks
        ib = min(block_size, n - i + 1)  # Determine the actual block size for this iteration

        # Perform a triangular matrix multiplication for lower triangular matrix
        view(a, i:i+ib-1, 1:i-1) .= adjoint(view(a, i:i+ib-1, i:i+ib-1)) * view(a, i:i+ib-1, 1:i-1)

        # Compute the product L' * L for the current block using adjoint for complex matrices
        L = view(a, i:i+ib-1, i:i+ib-1)  # Extract the block L
        Lt_L = adjoint(L) * L

        # Store the result back in the lower triangular part only
        for j in 1:ib
            for k in 1:j
                @inbounds a[i + j - 1, i + k - 1] = Lt_L[j, k]
            end
        end

        # Check if there are additional blocks to process
        if i + ib <= n
            # Perform matrix-matrix multiplication (equivalent to DGEMM) with proper adjoint
            view(a, i:i+ib-1, 1:i-1) .+= adjoint(view(a, i+ib:n, i:i+ib-1)) * view(a, i+ib:n, 1:i-1)

            # Perform symmetric rank-k update
            product_matrix = adjoint(view(a, i+ib:n, i:i+ib-1)) * view(a, i+ib:n, i:i+ib-1)
            for j in 1:ib
                for k in 1:j
                    a[i + j - 1, i + k - 1] += product_matrix[j, k]
                end
            end
        end
    end
end
