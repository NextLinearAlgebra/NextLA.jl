function zttqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
    begin
        if m < 0
            throw(ArgumentError("illegal value of m"))
            return -1
        end

        if n < 0
            throw(ArgumentError("illegal value of n"))
            return -2
        end

        if ib < 0
            throw(ArgumentError("illegal value of ib"))
            return -3
        end

        if lda1 < max(1, n) && n > 0
            throw(ArgumentError("illegal value of lda1"))
            return -5
        end

        if lda2 < max(1, m) && m > 0
            throw(ArgumentError("illegal value of lda2"))
            return -7
        end

        if ldt < max(1, ib) && ib > 0
            throw(ArgumentError("illegal value of ldt"))
            return -9
        end

        # quick return
        if m == 0 || n == 0 || ib == 0
            return
        end

        #   original function had this todo:
        #   todo: Need to check why some cases require this to avoid
        #   uninitialized values
        #   core_zlaset(CoreBlasGeneral, ib, n, 0.0, 0.0, T, ldt);

        one = oneunit(eltype(A1))

        for ii in 1:ib:n
            sb = min(n - ii + 1, ib)

            for i in 1:sb
                j = ii + i - 1 # index
                mi = min(j, m) # length
                ni = sb - i  # length

                A1[j, j], tau[j] = zlarfg(mi + 1, A1[j, j], (@view A2[1:mi, j]), 1, tau[j])

                if ni > 0
                    work[1:ni] .= (@view A1[j, j+1:j+ni])
                    conj!((@view work[1:ni]))

                    LinearAlgebra.generic_matvecmul!((@view work[1:ni]), 'C', (@view A2[1:mi, j+1:j+ni]),
                        (@view A2[1:mi, j]), LinearAlgebra.MulAddMul(one, one))
                    conj!((@view work[1:ni]))

                    alpha = -conj(tau[j])
                    axpy!(alpha, (@view work[1:ni]), (@view A1[j, j+1:j+ni]))
                    conj!((@view work[1:ni]))
                    gerc!(alpha, (@view A2[1:mi, j]), (@view work[1:ni]), (@view A2[1:mi, j+1:j+ni]))
                end

                # calculate T
                if i > 1
                    l = min(i - 1, max(0, m - ii + 1)) # length
                    alpha = -tau[j]

                    zpemv('C', 'C', min(j - 1, m), i - 1, l, alpha, (@view A2[1:m, ii:ii+i-2]), lda2,
                        (@view A2[1:m, j]), 0, (@view T[1:i-1, j]), work)
                    LinearAlgebra.generic_trimatmul!((@view T[1:i-1, j]), 'U', 'N', identity, (@view T[1:i-1, ii:ii+i-2]), (@view T[1:i-1, j]))
                end

                T[i, j] = tau[j]
            end

            if (n >= ii + sb)
                mi = min(ii + sb - 1, m)
                ni = n - (ii + sb - 1)
                l = min(sb, max(0, mi - ii + 1))
                ww = reshape(@view(work[1:sb*ni]), sb, ni) # k by n1 -- sb by ni

                zparfb('L', 'C', 'F', 'C', ib, ni, mi, ni, sb, l, (@view A1[ii:ii+ib-1, ii+sb:ii+sb+ni-1]),
                    lda1, (@view A2[1:mi, ii+sb:ii+sb+ni-1]), lda2, (@view A2[1:mi, ii:ii+sb-1]), lda2,
                    (@view T[1:sb, ii:ii+sb-1]), ldt, ww, sb)

            end
        end

        return
    end
end
