# File: zlauu2.jl

using LinearAlgebra

function zlauu2(uplo::Char, n::Int, A::Matrix{T}, lda::Int) where T
    # Initialize the INFO variable
    info = 0
    
    # Check if UPLO is valid
    if !(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l')
        info = -1
        return info
    end
    
    # Check other parameter constraints
    if n < 0
        info = -2
        return info
    end
    if lda < max(1, n)
        info = -4
        return info
    end
    
    # Quick return if possible
    if n == 0
        return info
    end
    
    if uplo == 'U' || uplo == 'u'
        # Compute the product U * U'
        for i in 1:n
            aii = A[i, i]
            if i < n
                # Update A[i, i] based on the dot product
                A[i, i] = aii * aii + real(dot(A[i, i+1:n], conj(A[i, i+1:n])))
                
                # Conjugate the next entries
                A[i, i+1:n] .= conj(A[i, i+1:n])
                
                # Perform the matrix-vector multiplication
                # @show size(A[1:i-1, i+1:n]), size(A[i, i+1:n]), size(aii)
                A[1:i-1, i] = A[1:i-1, i+1:n]*A[i, i+1:n] + aii * A[1:i-1, i] # Ax
              #  A[1:i-1, i] = A[1:i-1, i+1:n-i]*A[i+1:n-i, i]
                #display(A[1:i-1, i+1:n])
                #display(A[i, i+1:n])
                display(A[1:i-1, i])
                #A[1:lda, i] .+= A[:, i+1:n] * aii 
                # A[1, i:lda] += A[:, i+1:n] * aii
                # y := alpha*A*x + beta*y beta = aii
                # y = A*x + aii * y
                # A is A( 1, I+1 ) --> A[1, i+1:n]
                # x is A( I, I+1 ) --> A[i, 1+i:n]
                # y is A( 1, I )
                
            else
                # Scale the vector
                A[1:i, i] .= aii * A[1:i, i]
            end
        end
    else
        # Compute the product L' * L
        for i in 1:n
            aii = A[i, i]
            if i < n
                # Update A[i, i] based on the dot product
                A[i, i] = aii * aii + real(dot(A[i+1:n, i], conj(A[i+1:n, i])))
                
                # Conjugate the previous entries
                A[1:i-1, i] .= conj(A[1:i-1, i])
                
                # Perform the matrix-vector multiplication
                A[1:i-1, i] = A[1:i-1, i+1:n]*A[i, i+1:n] + aii * A[1:i-1, i]
                # A[i+1:n, 1:i-1] .+= A[i+1:n, i] * aii
            else
                # Scale the vector
                A[1:i, 1:i] .= aii * A[1:i, 1:i]
            end
        end
    end
    
    return info
end
