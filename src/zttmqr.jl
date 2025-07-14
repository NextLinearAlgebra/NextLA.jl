function zttmqr(side, trans, m1, n1, m2, n2, k, ib, A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)
    # check input arguments
    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
        return -1
    end

    if trans != 'N' && trans != 'C' 
        throw(ArgumentError("illegal value of trans"))
        return -2
    end

    if m1 < 0 
        throw(ArgumentError("illegal value of m1"))
        return -3
    end

    if n1 < 0
        throw(ArgumentError("illegal value of n1"))
        return -4
    end

    if (m2 < 0) || (m2 != m1 && side == 'R')
        throw(ArgumentError("illegal value of m2"))
        return -5
    end

    if (n2 < 0) || (n2 != n1 && side == 'L')
        throw(ArgumentError("illegal value of n2"))
        return -6
    end

    if (k < 0) || (side == 'L' && k > m1) || (side == 'R' && k > n1)
        throw(ArgumentError("illegal value of k"))
        return -7
    end

    if ib < 0
        throw(ArgumentError("illegal value of ib"))
        return -8
    end

    if lda1 < max(1,m1)
        throw(ArgumentError("illegal value of lda1"))
        return -10
    end

    if lda2 < max(1,m2)
        throw(ArgumentError("illegal value of lda2"))
        return -12
    end

    if ldv < max(1, side == 'L' ? m2 : n2)
        throw(ArgumentError("illegal value of ldv"))
        return -14
    end

    if ldt < max(1,ib)
        throw(ArgumentError("illegal of ldt"))
        return -16
    end

    if ldwork < max(1, side == 'L' ? ib : m1)
        throw(ArgumentError("illegal value of ldwork"))
        return -18
    end

    # quick return
    if m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0 || k == 0 || ib == 0
        return 0
    end

    if (side == 'L' && trans != 'N') || (side == 'R' && trans == 'N')
        i1 = 1
        i3 = ib
        istop = k
    else
        i1 = (div(k-1,ib))*ib + 1
        i3 = -ib
        istop = 1
    end

    ic = 1
    jc = 1
    mi = m1
    ni = n1
    mi2 = m2
    ni2 = n2    

    for i in i1:i3:istop
        kb = min(ib, k-i+1)

        if side == 'L'
            # H or H^HY applied to C[i:m, 1:n]
            mi = kb # m1 - i + 1
            mi2 = min(i+kb - 1, m2)
            ic = i
            l = min(kb, max(0, m2-i+1))
            ldvv = mi2
        else 
            ni = kb
            ni2 = min(i + kb - 1, n2)
            jc = i
            l = min(kb, max(0, n2-i+1))
            ldvv = ni2
        end

        
        # apply H or H^H 
        zparfb(side, trans, 'F', 'C', mi, ni, mi2, ni2, kb, l,
        (@view A1[ic:ic+mi-1, jc:jc+ni-1]), lda1, (@view A2[1:mi2, 1:ni2]), lda2, 
        (@view V[1:ldvv, i:i+kb-1]), ldvv, (@view T[1:kb, i:i+kb-1]), kb, work, ldwork)
    end
end
