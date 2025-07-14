function zpemv(trans, storev, m, n, l, alpha, A, lda, X, beta, Y, work)
    begin
        if trans != 'N' && trans != 'T' && trans != 'C'
            throw(ArgumentError("illegal value of trans"))
            return -1
        end

        if storev != 'C' && storev != 'R'
            throw(ArgumentError("illegal value of storev"))
            return -2
        end

        if !((storev == 'C' && trans != 'N') || (storev == 'R' && trans == 'N'))
            throw(ArgumentError("illegal values of trans/storev"))
            return -2
        end

        if m < 0
            throw(ArgumentError("illegal value of m"))
            return -3
        end

        if n < 0
            throw(ArgumentError("illegal value of n"))
            return -4
        end

        if l > min(m, n)
            throw(ArgumentError("illegal value of l"))
            return -5
        end

        if lda < max(1, m)
            throw(ArgumentError("illegal value of lda"))
            return -8
        end

        # quick return 
        if m == 0 || n == 0
            return
        end

        if alpha == 0 && beta == 0
            return
        end

        if l == 1
            l = 0
        end

        if storev == 'C'
            x1 = (@view X[1:m-l])
            x2 = (@view X[m-l+1:m])
            xf = (@view X[1:m])
        else # assume incx = ldaX
            x1 = (@view X[1:n-l])
            x2 = (@view X[n-l+1:n])
            xf = (@view X[1:n])
            # columnwise 
        end

        if storev != 'C'
            y1 = (@view Y[1:l])
            y2 = (@view Y[l+1:m])
        else # assume incy = ldaY
            y1 = (@view Y[1:l])
            y2 = (@view Y[l+1:n])
            # columnwise 
        end


        if storev == 'C'
            if trans == 'N'
                throw(ErrorException("not implemented"))
                return -1
            else
                if l > 0
                    (@view work[1:l]) .= (@view X[m-l+1:m])

                    if trans == 'C'
                        LinearAlgebra.generic_trimatmul!((@view work[1:l]), 'U', 'N', adjoint,
                            (@view A[m-l+1:m, 1:l]), (@view work[1:l]))
                    else
                        LinearAlgebra.generic_trimatmul!((@view work[1:l]), 'U', 'N', transpose,
                            (@view A[m-l+1:m, 1:l]), (@view work[1:l]))
                    end

                    if m > l
                        LinearAlgebra.generic_matvecmul!((@view Y[1:l]), trans, (@view A[1:m-l, 1:l]),
                            (@view X[1:m-l]), LinearAlgebra.MulAddMul(alpha, beta))
                        LinearAlgebra.axpy!(alpha, (@view work[1:l]), (@view Y[1:l]))
                    else
                        if beta == 0
                            (@view work[1:l]) .*= alpha
                            (@view Y[1:l]) .= (@view work[1:l])
                        else
                            (@view Y[1:l]) .*= beta
                            LinearAlgebra.axpy!(alpha, (@view work[1:l]), (@view Y[1:l]))
                        end

                    end
                end

                if n > l
                    k = n - l
                    LinearAlgebra.generic_matvecmul!((@view Y[l+1:n]), trans, (@view A[1:m, l+1:n]),
                        (@view X[1:m]), LinearAlgebra.MulAddMul(alpha, beta))
                end
            end
        else
            if trans == 'N'
                if l > 0
                    work[1:l] .= x2
                    LinearAlgebra.generic_trimatmul!((@view work[1:l]), 'L', 'N', identity,
                        (@view A[1:l, n-l+1:n]), (@view work[1:l]))

                    if n > l
                        LinearAlgebra.generic_matvecmul!(y1, 'N', (@view A[1:l, 1:n-l]),
                            x1, LinearAlgebra.MulAddMul(alpha, beta))
                        LinearAlgebra.axpy!(alpha, (@view work[1:l]), y1)
                    else
                        if beta == 0
                            y1 .= alpha * (@view work[1:l])
                        else
                            y1 .*= beta
                            LinearAlgebra.axpy!(alpha, (@view work[1:l]), y1)
                        end
                    end
                end

                if m > l
                    LinearAlgebra.generic_matvecmul!(y2, 'N', (@view A[l+1:m, 1:n]),
                        xf, LinearAlgebra.MulAddMul(alpha, beta))
                end
            else
                throw(ErrorException("not implemented"))
                return -1
            end
        end
    end
end