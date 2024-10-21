using LinearAlgebra

function zlauu2(uplo::Char, n::Int, A::AbstractMatrix{T}, lda::Int) where T
    """
    Purpose:
    =======
    ZLAUU2 computes the product U * U' or L' * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array A.

    If UPLO = 'U' or 'u', the upper triangle of the result is stored,
    overwriting the factor U in A.
    If UPLO = 'L' or 'l', the lower triangle of the result is stored,
    overwriting the factor L in A.

    Arguments:
    ==========
    UPLO    (input) CHARACTER*1
            Specifies whether the triangular factor stored in the array A
            is upper or lower triangular:
            = 'U':  Upper triangular
            = 'L':  Lower triangular
    
    N       (input) INTEGER
            The order of the triangular factor U or L.  N >= 0.
    
    A       (input/output) COMPLEX{T} array, dimension (LDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = 'U', the upper triangle of A is
            overwritten with the upper triangle of the product U * U';
            if UPLO = 'L', the lower triangle of A is overwritten with
            the lower triangle of the product L' * L.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
    
    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
    """
    
    # Initialize the INFO variable
    info = 0

    # Validate the input for 'uplo'
    if !(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l')
        info = -1
        return info
    end

    # Check for valid matrix order
    if n < 0
        info = -2
        return info
    end

    # Validate the leading dimension of A
    if lda < max(1, n)
        info = -4
        return info
    end

    # Quick return if possible (nothing to do if n is zero)
    if n == 0
        return info
    end

    if uplo == 'U' || uplo == 'u'
        # Upper triangular case: Compute U * U'
        for i in 1:n
            aii = A[i, i]  # Diagonal element of U

            if i < n
                # Update the diagonal element
                A[i, i] = aii^2 + real(dot(A[i, i+1:n], A[i, i+1:n]))

                # Update the remaining upper triangle elements
                if i > 1
                    A[1:i-1, i] .= A[1:i-1, i+1:n] * A[i, i+1:n] + A[1:i-1, i] * aii
                end
            else
                # Scale diagonal entries when i == n
                A[1:i, i] .= aii * A[1:i, i]
            end
        end
    else
        # Lower triangular case: Compute L' * L
        for i in 1:n
            aii = A[i, i]  # Diagonal element of L

            if i < n
                # Update the diagonal element
                A[i, i] = aii^2 + real(dot(A[i+1:n, i], A[i+1:n, i]))

                # Update the remaining lower triangle elements
                if i > 1
                    A[i, 1:i-1] .= A[i+1:n, 1:i-1]' * A[i+1:n, i] + A[i, 1:i-1] * aii
                end
            else
                # Scale diagonal entries when i == n
                A[i, 1:i] .= aii * A[i, 1:i]
            end
        end
    end

    return info
end
