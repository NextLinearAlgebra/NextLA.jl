function zlarft(direct, storev, n, k, v, ldv, tau, t, ldt)
    if n == 0
        return
    end

    zero0 = zero(eltype(v))
    one = oneunit(eltype(v))

    if direct == 'F'
        prevlastv = n

        for i in 1:k

            prevlastv = max(prevlastv, i)

            if tau[i] == 0
                # H(i) = i

                for j in 1:i
                    t[j,i] = zero0
                end

            else
                # general case

                if storev == 'C'
                    lastv = n
                    
                    #for lastv in n:-1:i+1
                    while lastv >= i+1

                        if v[lastv, i] != 0
                            break
                        end

                        lastv -= 1
                    end

                    for j in 1:i-1
                        t[j,i] = -tau[i] * conj(v[i,j])
                    end

                    j = min(lastv, prevlastv)

                    # t[1:i-1, i] = -tau[i] * v[i:j, 1:i-1]^H * v[i:j, i] 
                    LinearAlgebra.generic_matvecmul!((@view t[1:i-1, i]), 'C', (@view v[i+1:j, 1:i-1]), 
                    (@view v[i+1:j,i]), LinearAlgebra.MulAddMul(-tau[i], one))

                else
                    lastv = n
                    #for lastv in n:-1:i+1
                    while lastv >= i+1
                        if v[i, lastv] != 0
                            break
                        end

                        lastv -= 1
                    end

                    for j in 1:i-1
                        t[j,i] = -tau[i] * v[j,i]
                    end

                    j = min(lastv, prevlastv)

                    # t[1:i-1, i] = -tau[i] * v[1:i-1, i:j] * v[i,i:j]^H
                    if i-1 > 0
                        LinearAlgebra.generic_matmatmul!((@view t[1:i-1, i]), 'N', 'C', (@view v[1:i-1, i:j]), 
                        (@view v[i:i, i:j]), LinearAlgebra.MulAddMul(-tau[i], one))
                    end
                end

                #t[1:i-1,i] = t[1:i-1, 1:i-1] * t[1:i-1,i]
                LinearAlgebra.generic_trimatmul!((@view t[1:i-1,i]), 'U', 'N', identity, 
                (@view t[1:i-1, 1:i-1]), (@view t[1:i-1, i]))

                t[i,i] = tau[i]

                if i > 1
                    prevlastv = max(prevlastv, lastv)
                else
                    prevlastv = lastv
                end

            end
        end
    else
        prevlastv = 1
        for i in k:-1:1
            if tau[i] == 0

                #H(i) = I

                for j in i:k
                    t[j,i] = zero0
                end

            else
                if i < k
                    if storev == 'C'
                        lastv = 1

                        #for lastv in 1:i-1
                        while lastv <= i-1
                            if v[lastv,i] != 0
                                break
                            end
                            lastv += 1
                        end

                        for j in i+1:k
                            t[j,i] = -tau[i] * conj(v[n-k+i, j])
                        end
                        
                        j = max(lastv, prevlastv)


                        #t[i+1:k, i] = -tau[i] * v[j:n-k+i, i+1:k]^H * v[j:n-k+i, i]

                        LinearAlgebra.generic_matvecmul!((@view t[i+1:k, i]), 'C', (@view v[j:n-k+i, i+1:k]), 
                        (@view v[j:n-k+i, k]), LinearAlgebra.MulAddMul(-tau[i], one))
                    else
                        lastv = 1
                        #for lastv in 1:i-1
                        while lastv <= i-1
                            if v[lastv,i] != 0
                                break
                            end
                            lastv += 1
                        end

                        for j in i+1:k
                            t[j,i] = -tau[i] * v[j, n-k+i]
                        end
                        
                        j = max(lastv, prevlastv)

                        #t[i+1:k, i] = -tau[i] * v[i+1:k , j:n-k+i] * v[i, j:n-k+i]^H
                        LinearAlgebra.generic_matmatmul!((@view t[i+1:k, i]), 'N', 'C', (@view v[i+1:k, j:n-k+i-1]), 
                        (@view v[i:i, j:n-k+i-1]), LinearAlgebra.MulAddMul(-tau[i], one))
                    end

                    # t[i+1:k, i] = t[i+1:k, i+1:k] * t[i+1:k, i]

                    LinearAlgebra.generic_trimatmul!((@view t[i+1:k, i]), 'L', 'N', identity, 
                    (@view t[i+1:k, i+1:k]), (@view t[i+1:k, i]))

                    if i > 1
                        prevlastv = min(prevlastv, lastv)
                    else
                        prevlastv = lastv
                    end

                end

                t[i,i] = tau[i]
            end
        end
    end
end