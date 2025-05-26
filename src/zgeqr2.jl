using LinearAlgebra

function zgeqr2(m,n, A, lda, tau, work)
    if m < 0
        throw(ArgumentError("illegal value of m"))
        return -1
    end
    
    if n < 0
        throw(ArgumentError("illegal value of n"))
        return -2
    end

    if lda < max(1,m)
        throw(ArgumentError("illegal value of lda"))
        return -4
    end

    k = min(m,n)
    one = oneunit(eltype(A))

    #av = parent(A)
    #a1, a2 = parentindices(A)
    #a1 = a1.start-1
    #a2 = a2.start-1

    for i in 1:k
        # generate elementary reflector H(i) to anniliate A(i+1:m, i)
        A[i,i], tau[i] = zlarfg(m-i+1, A[i, i], (@view A[min(i+1,m):m, i]), 1, tau[i])
        
        if i < n
            # apply H(i)^H to A(i:m, i+1:n) from left
            alpha = A[i,i]
            A[i,i] = one

            #LinearAlgebra.LAPACK.larf!('L', (@view A[i:m, i]), conj(tau[i]), (@view A[i:m, i+1:n]), work)
            zlarf('L', m-i+1, n-i, (@view A[i:m, i]), 1, conj(tau[i]), (@view A[i:m, i+1:n]), lda, work)
            #zlarf('L', m-i+1, n-i, (@view av[i+a1:m+a1, i+a2]), 1, conj(tau[i]), (@view av[i+a1:m+a1, i+1+a2:n+a2]), lda, work)

            A[i,i] = alpha
        end
    end

    return   
end
