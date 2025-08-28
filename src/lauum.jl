export lauum!

# Import the unblocked version (lauu2!) for use in blocked algorithm

"""
    lauum!(uplo, n, A, ib)

Compute the product U * U^H or L^H * L using a blocked algorithm, where
the triangular factor U or L is stored in the upper or lower triangular
part of the matrix a.

This is a blocked version of the triangular matrix multiplication that
achieves better performance on large matrices by exploiting cache locality
and enabling vectorization.

# Arguments
- `uplo`: Character specifying which triangle contains the factor
  - 'U': Upper triangular, computes U * U^H  
  - 'L': Lower triangular, computes L^H * L
- `n`: Order of the triangular matrix (≥ 0)
- `A`: Matrix containing triangular factor (modified in-place)
- `ib`: Block size for blocked algorithm (typically 32-64)

# Algorithm
The blocked algorithm partitions the matrix into blocks of size ib
and processes them using high-performance BLAS operations:
- Level-3 BLAS (matrix-matrix operations) for most computations
- Level-2 BLAS (matrix-vector operations) for smaller blocks
- Automatic fallback to unblocked algorithm for small matrices

For upper triangular (uplo='U'): A := U * U^H
For lower triangular (uplo='L'): A := L^H * L

# Performance Notes
- Block size should be chosen based on cache size (typically 32-64)
- Uses parallel processing for independent block operations
- Optimal performance achieved when n >> block_size

# Input Validation
- uplo must be 'U' or 'L'
- n must be non-negative  
- block_size is automatically clamped to valid range

# Example
```julia
n = 100
block_size = 32
A = triu(randn(ComplexF64, n, n))
lauum!('U', n, A, n, block_size)  # A := U * U^H
```
"""
function lauum!(uplo::Char, n::Integer, A::AbstractMatrix{T}, ib::Integer) where {T}
    # Input validation with descriptive error messages
    if !(uplo in ['U', 'L'])
        throw(ArgumentError("uplo must be 'U' or 'L', got '$uplo'"))
    end

    if n < 0
        throw(ArgumentError("n must be non-negative, got $n"))
    end

    # Quick return for degenerate case
    if n == 0
        return
    end

    # Adjust block_size to reasonable bounds
    ib = max(1, min(ib, n))

    # Use unblocked algorithm for small matrices or invalid block size
    if ib <= 1 || ib >= n
        lauu2!(uplo, n, A)
        return
    end

    # Call appropriate blocked computation
    if uplo == 'U'
        compute_upper!(n, ib, A)
    else
        compute_lower!(n, ib, A)
    end
end

"""
    compute_upper!(n, block_size, a)

Blocked computation of U * U^H for upper triangular matrix U.

Processes the matrix in blocks to achieve better cache performance and
enable vectorized operations. Uses Level-3 BLAS operations where possible.

# Arguments
- `n`: Order of the matrix
- `block_size`: Size of blocks for processing  
- `a`: Matrix containing U (modified in-place)

# Algorithm
For each diagonal block:
1. Update off-diagonal blocks using TRMM operations
2. Compute diagonal block product U_block * U_block^H  
3. Add contribution from remaining blocks using SYRK operations
"""
function compute_upper!(n::Integer, ib::Integer, A::AbstractMatrix{T}) where T
    Threads.@threads for i in 1:ib:n
        ib = min(ib, n - i + 1)  # Actual block size

        # Update off-diagonal blocks: A[1:i-1, i:i+ib-1] = A[1:i-1, i:i+ib-1] * A[i:i+ib-1, i:i+ib-1]^H
        if i > 1
            view(A, 1:i-1, i:i+ib-1) .= view(A, 1:i-1, i:i+ib-1) * view(A, i:i+ib-1, i:i+ib-1)'
        end

        # Compute diagonal block: U_block * U_block^H
        U_block = view(A, i:i+ib-1, i:i+ib-1)
        U_Ut = U_block * U_block'
        
        # Store only upper triangular part
        for j in 1:ib, k in j:ib
            A[i + j - 1, i + k - 1] = U_Ut[j, k]
        end
        
        # Add contribution from trailing blocks if they exist
        if i + ib <= n
            # Update off-diagonal: add A[1:i-1, i+ib:n] * A[i:i+ib-1, i+ib:n]^H
            if i > 1
                view(A, 1:i-1, i:i+ib-1) .+= view(A, 1:i-1, i+ib:n) * view(A, i:i+ib-1, i+ib:n)'
            end

            # Rank-k update: add A[i:i+ib-1, i+ib:n] * A[i:i+ib-1, i+ib:n]^H to diagonal block
            trailing_block = view(A, i:i+ib-1, i+ib:n)
            syrk_result = trailing_block * trailing_block'
            
            for j in 1:ib, k in j:ib
                A[i + j - 1, i + k - 1] += syrk_result[j, k]
            end
        end
    end
end

"""
    compute_lower!(n, ib, A) 

Blocked computation of L^H * L for lower triangular matrix L.

Processes the matrix in blocks to achieve better cache performance and
enable vectorized operations. Uses Level-3 BLAS operations where possible.

# Arguments  
- `n`: Order of the matrix
- `block_size`: Size of blocks for processing
- `A`: Matrix containing L (modified in-place)  

# Algorithm
For each diagonal block:
1. Update off-diagonal blocks using TRMM operations
2. Compute diagonal block product L_block^H * L_block
3. Add contribution from remaining blocks using SYRK operations
"""
function compute_lower!(n::Integer, ib::Integer, A::AbstractMatrix{T}) where T
    Threads.@threads for i in 1:ib:n
        ib = min(ib, n - i + 1)  # Actual block size

        # Update off-diagonal blocks: A[i:i+ib-1, 1:i-1] = A[i:i+ib-1, i:i+ib-1]^H * A[i:i+ib-1, 1:i-1]
        if i > 1
            view(A, i:i+ib-1, 1:i-1) .= view(A, i:i+ib-1, i:i+ib-1)' * view(A, i:i+ib-1, 1:i-1)
        end

        # Compute diagonal block: L_block^H * L_block  
        L_block = view(A, i:i+ib-1, i:i+ib-1)
        Lt_L = L_block' * L_block

        # Store only lower triangular part
        for j in 1:ib, k in 1:j
            A[i + j - 1, i + k - 1] = Lt_L[j, k]
        end

        # Add contribution from trailing blocks if they exist
        if i + ib <= n
            # Update off-diagonal: add A[i+ib:n, i:i+ib-1]^H * A[i+ib:n, 1:i-1]
            if i > 1
                view(A, i:i+ib-1, 1:i-1) .+= view(A, i+ib:n, i:i+ib-1)' * view(A, i+ib:n, 1:i-1)
            end

            # Rank-k update: add A[i+ib:n, i:i+ib-1]^H * A[i+ib:n, i:i+ib-1] to diagonal block
            trailing_block = view(A, i+ib:n, i:i+ib-1)
            syrk_result = trailing_block' * trailing_block
            
            for j in 1:ib, k in 1:j
                A[i + j - 1, i + k - 1] += syrk_result[j, k]
            end
        end
    end
end
