function larf(side, m, n, v, incv, tau, c, work)
    lastv = 0
    lastc = 0
    one = oneunit(eltype(c))
    zero0 = zero(eltype(c))
    
    if tau != 0
        # set up variables for scanning v, lastv beigns pointing to end of V

        if side == 'L' 
            lastv = m
        else
            lastv = n
        end
        
        if incv > 0
            i = 1 + (lastv-1)*incv
        else
            i = 1
        end

        while lastv > 0 && v[i] == 0
            lastv -= 1
            i -= incv
        end

        if side == 'L'
            # scan for last non-zero column in C[1:lastv, :]
            lastc = ilazlc(lastv, n, c)
        else
            # scan for last non-zero row in C[:, 1:lastv]
            lastc = ilazlr(m, lastv, c)
        end
    end

    if side == 'L'
        #form H*C

        if lastv > 0
            vv = @view v[1:lastv, 1]
            cv = @view c[1:lastv, 1:lastc]
            wv = @view work[1:lastc]
            # w[1:lastc,1] = c[1:lastv, 1:lastc]^H * v[1:lastv, 1]

            #LinearAlgebra.BLAS.gemv!('C', one, cv, vv, zero0, wv)
            LinearAlgebra.generic_matvecmul!(wv, 'C', cv, vv, LinearAlgebra.MulAddMul(one, zero0))
            #LinearAlgebra.generic_matvecmul!((@view work[1:lastc, 1]), 'C', (@view c[1:lastv, 1:lastc]), 
            #(@view v[1:lastv, 1]), LinearAlgebra.MulAddMul(one, zero0))

            #c[1:lastv,1:lastc] -= tau*v[1:lastv, 1]*w[1:lastc,1]^H
            #LinearAlgebra.BLAS.gemm!('N', 'C', -tau, vv, wv, one, cv)
           gerc!(-tau, vv, wv, cv)

            #LinearAlgebra.generic_matmul!((@view c[1:lastv, 1:lastc]), 'N', (@view v[1:lastv, 1]), 
            #(@view work[1:lastc, 1]), LinearAlgebra.MulAddMul(-tau, one))
        end
    else
        #form C*H

        if lastv > 0
            # w[1:lastc,1] = c[1:lastc, 1:lastv] * v[1:lastv, 1]
            LinearAlgebra.generic_matvecmul!((@view work[1:lastc, 1]), 'N', (@view c[1:lastc, 1:lastv]),
            (@view v[1:lastv, 1]), LinearAlgebra.MulAddMul(one, zero0))

            #c[1:lastc,1:lastv] -= tau(?)*w[1:lastc,1]*v[1:lastv, 1]^H
            
            #LinearAlgebra.BLAS.ger!(-tau, wv, vv, cv)
            gerc!(-tau, (@view work[1:lastc, 1]), (@view v[1:lastv, 1]), (@view c[1:lastc, 1:lastv]))
        end
    end
end

function ilazlc(m,n,a)
    if n == 0
        return n
    end

    if a[1,n] != 0 || a[m,n] != 0 
        return n
    end

    for j in n:-1:1
        for i in 1:m
            if a[i, j] != 0
                return j
            end
        end
    end
end

function ilazlr(m,n,a)
    
    if m == 0
        return m
    end

    if a[m,1] != 0 || a[m,n] != 0 
        return m
    end

    ila = 0

    for j in 1:n
        i = m
        while (a[max(i,1), j] == 0) && (i > 1)
            i -= 1
        end

        ila = max(ila, i)
    end

    return ila
end
