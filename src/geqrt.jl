function geqrt(m,n,ib, A, lda, T, ldt, tau, work)
    if m < 0
        throw(ArgumentError("illegal value of m"))
        return -1
    end

    if n < 0
        throw(ArgumentError("illegal value of n"))
        return -2
    end

    if (ib < 0) || ((ib == 0) && (m > 0) && (n > 0))
        throw(ArgumentError("illegal value of ib"))
        return -3
    end

    if lda < max(1,m) && m > 0
        throw(ArgumentError("illegal value of lda"))
        return -5
    end

    if ldt < max(1,ib) && ib > 0
        throw(ArgumentError("illegal value of ldt"))
        return -7
    end

    if m == 0 || n == 0 || ib == 0
        return 
    end

    k = min(m,n)

    for i in 1:ib:k
        sb = min(ib, k-i+1)

        av = @view A[i:m, i:i+sb-1]
        tv = @view T[1:sb,i:i+sb-1]
        tauv = @view tau[i:i+sb-1]

        # compute qr for A[i:m, i:i+sb-1]
        
        geqr2(m-i+1, sb, av, lda, tauv, work)
        larft('F', 'C', m-i+1, sb, av, lda, tauv, tv, ldt)

        if n >= i + sb
            # update by apply H^H to A[i:m, i+sb:n] from left

            #wwork = @view work[1: (n-i-sb+1)*sb]
            #ww = reshape(wwork, n-i-sb+1, sb)
            ww = reshape((@view work[1: (n-i-sb+1)*sb]), n-i-sb+1, sb)

            larfb('L', 'C', 'F', 'C', m-i+1, n-i-sb+1, sb, av, 
                m-i+1, tv, sb, (@view A[i:m, i+sb:n]), lda, ww, n-i-sb+1)
        end
    end

    return
end
