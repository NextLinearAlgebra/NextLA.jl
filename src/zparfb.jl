export zparfb

function zparfb(side, trans, direct, storev, m1, n1, m2, n2, k, l, 
                A1, lda1, A2, lda2, V,  ldv, T, ldt, work, ldwork)

    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
        return -1
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
        throw(ArgumentError("illegal value of trans"))
        return -2
    end

    if direct != 'F' && direct != 'B'
        throw(ArgumentError("illegal value of direct"))
        return -3
    end

    if storev != 'C' && storev != 'R'
        throw(ArgumentError("illegal value of storev"))
        return -4
    end

    if m1 < 0
        throw(ArgumentError("illegal value of m1"))
        return -5
    end

    if n1 < 0
        throw(ArgumentError("illegal value of n1"))
        return -6
    end

    if m2 < 0 || (side == 'R' && m1 != m2)
        throw(ArgumentError("illegal value of m2"))
        return -7
    end

    if n2 < 0 || (side == 'L' && n1 != n2)
        throw(ArgumentError("illegal value of n2"))
        return -8
    end

    if k < 0
        throw(ArgumentError("illegal value of k"))
        return -9
    end

    if l < 0 || l > k
        throw(ArgumentError("illegal value of l"))
        return -10
    end

    if lda1 < 0
        throw(ArgumentError("illegal value of lda1"))
        return -12
    end

    if lda2 < 0
        throw(ArgumentError("illegal value of lda2"))
        return -14
    end

    if ldv < 0 
        throw(ArgumentError("illegal value of ldv"))
        return -16
    end

    if ldt < 0
        throw(ArgumentError("illegal value of ldt"))
        return -18
    end

    if ldwork < 0
        throw(ArgumentError("illegal value of ldwork"))
        return -20
    end

    # quick return 

    if m1 == 0 || n1 == 0 || n2 == 0 || k == 0
        return 
    end

    one0 = oneunit(eltype(A1))
    zero0 = zero(eltype(A1))

    if trans == 'N'
        tfun = identity
    else
        tfun = adjoint
    end

    if direct == 'F'
        forward = true
    else
        forward = false
    end

    if side == 'L'
        left = true
    else
        left = false
    end

    if storev == 'C'
        colmajor = true
    else
        colmajor = false
    end
    zpamm('W', side, storev, direct, m2, n2, k, l, A1, lda1, A2, lda2, V, ldv, work, ldwork)

    if colmajor && forward && left # colmajor, forward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'U', 'N', tfun, (@view T[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one0, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if colmajor && forward && !left # colmajor, forward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'U', 'N', tfun, (@view work[1:m2, 1:k]), (@view T[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one0, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    if colmajor && !forward && left # colmajor, backward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'L', 'N', tfun, (@view T[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one0, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if colmajor && !forward && !left # colmajor, backward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'L', 'N', tfun, (@view work[1:m2, 1:k]), (@view T[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one0, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    if !colmajor && forward && left # rowmajor, forward, left

        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'U', 'N', tfun, (@view T[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!((-one0), (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if !colmajor && forward && !left # rowmajor, forward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'U', 'N', tfun, (@view work[1:m2, 1:k]), (@view T[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!(-one0, (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end

    if !colmajor && !forward && left # rowmajor, backward, left
        LinearAlgebra.generic_trimatmul!((@view work[1:k, 1:n2]), 'L', 'N', tfun, (@view T[1:k, 1:k]), (@view work[1:k, 1:n2]))

        for i in 1:k
            LinearAlgebra.axpy!(-one0, (@view work[i, 1:n2]), (@view A1[i, 1:n2]))
        end
    end

    if !colmajor && !forward && !left # rowmajor, backward, right
        LinearAlgebra.generic_mattrimul!((@view work[1:m2, 1:k]), 'L', 'N', tfun, (@view work[1:m2, 1:k]), (@view T[1:k, 1:k]))

        for j in 1:k
            LinearAlgebra.axpy!((-one0), (@view work[1:m2, j]), (@view A1[1:m2, j]))
        end
    end


    zpamm('A', side, storev, direct, m2, n2, k, l, A1, lda1, A2, lda2, V, ldv, work, ldwork)

    return
end
