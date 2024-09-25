using LinearAlgebra

"""
    zlauu2(uplo::Char, n::Int, A::Matrix{Complex{Float64}}, lda::Int) -> Int

Computes the product U * U' or L' * L, where the triangular factor U or L is stored in the upper or lower triangular part of the array A.

# Arguments
- `uplo::Char`: Specifies whether the triangular factor stored in A is upper or lower triangular:
    - 'U' or 'u': Upper triangular
    - 'L' or 'l': Lower triangular
- `n::Int`: The order of the triangular factor U or L. Must be `n >= 0`.
- `A::Matrix{Complex{Float64}}`: The input/output matrix where the triangular factor is stored. It will be updated with the product.
- `lda::Int`: The leading dimension of the array A. Must satisfy `lda >= max(1, n)`.

# Returns
- `info::Int`: 
    - 0 if successful.
    - < 0 if `info = -k`, the k-th argument had an illegal value.
"""
function zlauu2(uplo::Char, n::Int, A::Matrix{Complex{Float64}}, lda::Int)
    # Initialize the INFO variable
    info = 0

    # Validate input parameters
    if !(uplo in ['U', 'u', 'L', 'l'])
        return -1  # Invalid value for UPLO
    end
    if n < 0
        return -2  # Invalid value for N
    end
    if lda < max(1, n)
        return -4  # Invalid value for LDA
    end

    # Quick return if possible
    if n == 0
        return info  # Nothing to do
    end

    # Perform the computation based on the value of UPLO
    if uplo in ['U', 'u']
        for i in 1:n
            aii = A[i, i]
            if i < n
                # Update A[i, i] using the dot product
                A[i, i] += aii * aii + real(dot(A[i, i+1:n], conj(A[i, i+1:n])))
                
                # Conjugate the entries in the upper triangular part
                A[i, i+1:n] .= conj(A[i, i+1:n])
                
                # Perform matrix-vector multiplication
                A[1:i-1, i] .+= A[1:i-1, i+1:n] * aii
            else
                # Scale the vector if at the last diagonal element
                A[1:i, i] .= aii * A[1:i, i]
            end
        end
    else
        for i in 1:n
            aii = A[i, i]
            if i < n
                # Update A[i, i] using the dot product
                A[i, i] += aii * aii + real(dot(A[i+1:n, i], conj(A[i+1:n, i])))
                
                # Conjugate the entries in the lower triangular part
                A[1:i-1, i] .= conj(A[1:i-1, i])
                
                # Perform matrix-vector multiplication
                A[i+1:n, 1:i-1] .+= A[i+1:n, i] * aii
            else
                # Scale the vector if at the last diagonal element
                A[1:i, 1:i] .= aii * A[1:i, 1:i]
            end
        end
    end

    return info  # Successful exit
end
