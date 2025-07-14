"""
    zlarfb(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)

Applies complex block reflector H or its transpose H^H to m-by-n matrix C from either the left or the right
Implemented with Julia internal functions for matrix multiplication

# Arguments
- 'side': 
    - 'L' : apply H or H^H from the left
    - 'R' : apply H or H^H from the right
- 'trans': 
    - 'N' : apply H
    - 'C' : apply H^H
- 'direct':  indicates how H is formed from product of elementary reflectors
    - 'F' : H = H(1) H(2) ... H(k) (Forward)
    - 'B' : H = H(k) ... H(2) H(1) (Backward)
- 'storev': indcicates how the vectors which define the elementary reflectors are stored
    - 'C' : columnwise
    - 'R' : rowwise
- 'm': the number of rows of matrix c
- 'n': the number of columns of matrix c
- 'k': the order of marix t (= the number of elementary reflectors whose roduct defines the block reflector)
- 'v': dimension 
        - (ldv, k) if storev = 'C'
        - (ldv, m) if storev = 'R' and side = 'L'
        - (ldv, n) if storev = 'R' and side = 'R'
- 'ldv': the leading dimension of array v
    - if storev = 'C' and side = 'L', ldv >= max(1,m)
    - if storev = 'C' and side = 'R', ldv >= max(1,n)
    - if storev = 'R', ldv >= k
- 't': dimension (ldv, k), the triangular k-by-k matrix t in representation of the block reflector
- 'ldt': the leading dimension of array t, ldt >= k
- 'c': 
    - on entry m-by-n matrix
    - on exit, overwritten by H*C or H^H*C or C*H or C*H^H
- 'ldc': the leading dimension of c. ldc >= max(1,m)
- 'work': dimension (ldwork, k)
- 'ldwork': 
    - if side = 'L', ldwork >= max(1,n)
    - if side = 'R', ldwork >= max(1,m)
"""
function zlarfb(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)
    
    if m <= 0 || n <= 0
        return
    end

    one = oneunit(eltype(c))
    plus = LinearAlgebra.MulAddMul(one, one)
    minus = LinearAlgebra.MulAddMul(one*(-1),one)

    if storev == 'C'
        if direct == 'F'
            """
            V = (V1) (first k rows)
                (V2)
            where V1 is unit lower triangular
            """
            if side == 'L'
                """
                Form H*C or H^H * C where C = (C1)
                                              (C2)
                """

                c1 = @view c[1:k,:] 
                c2 = @view c[k+1:m,:]
                v1 = @view v[1:k,:]
                v2 = @view v[k+1:m,:]
            
                work .= c1'

                # W = W*V1           
                LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v1)

                if m > k
                    # W = W + C2^H * V2
                    LinearAlgebra.generic_matmatmul!(work, 'C', 'N', c2, v2, plus)
                end
                
                # W = W * T^H or W*T

                if trans == 'N' # W = W*T^H
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, t)
                else
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, t)
                end

                if m > k 
                    #C2 = C2 - V2*W^H
                    LinearAlgebra.generic_matmatmul!(c2, 'N', 'C', v2, work, minus)
                end

                # w = w*v1^H
                LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v1)

                c1 .-= (work)'

            else 
                if side == 'R'
                    """
                    Form C*H or C*H^H where C = (c1 c2)
                    """
                    c1 = @view c[:, 1:k]
                    c2 = @view c[:, k+1:n]
                    v1 = @view v[1:k,:]
                    v2 = @view v[k+1:n,:]

                    work .= c1

                    # w = w*v1
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v1)


                    if n > k
                        # w = w + c2*V2
                        LinearAlgebra.generic_matmatmul!(work, 'N', 'N', c2, v2, plus)
                    end
                    
                    #w = w*t or w*t^H

                    if trans == 'C' # W = W*T^H
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, t)
                    else
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, t)
                    end

                    if n > k
                        # c2 = c2 - w*v2^h
                        LinearAlgebra.generic_matmatmul!(c2, 'N', 'C', work, v2, minus)
                    end

                    #work = work*(v1')
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v1)

                    c1 .-= work
                end
            end
        else
            """
            V = (v1)
                (v2) (last k rows)
            where v2 is unit upper triangular
            """
            if side == 'L'
                """
                Form H*C or H^H*C where C = (c1)
                                            (c2)
                """
                c1 = @view c[1:m-k,:]
                c2 = @view c[m-k+1:m,:]
                v1 = @view v[1:ldv-k,:]
                v2 = @view v[ldv-k+1:ldv,:]
                
                work .= c2'

                #work = work*v2
                LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v2)

                if m > k
                    #work = work + (c1')*V1
                    LinearAlgebra.generic_matmatmul!(work, 'C', 'N', c1, v1, plus)
                end

                if trans == 'N'
                    #work = work*(t')
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, t)
                else
                    #work = work*t
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, t)
                end

                #c1 = c1 - v1*w^H
                if m > k
                    LinearAlgebra.generic_matmatmul!(c1, 'N', 'C', v1, work, minus)                    
                end

                #work = work*(v2')
                LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v2)

                #c2 = c2 - w^H
                for j in 1:k
                    for i in 1:n
                        c[m-k+j,i] = c[m-k+j,i] - conj(work[i,j])
                    end
                end
            else 
                if side == 'R'
                    """
                    Form C*H or C*H^H where C = (c1 c2)
                    """
                    c1 = @view c[:,1:n-k]
                    c2 = @view c[:,n-k+1:n]
                    v1 = @view v[1:ldv-k,:]
                    v2 = @view v[ldv-k+1:ldv,:]

                    work .= c2

                    #work = work*v2
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v2)

                    if n > k
                        #work = work + c1*V1
                        LinearAlgebra.generic_matmatmul!(work, 'N', 'N', c1, v1, plus)
                    end

                    if trans == 'C'
                        #work = work*(t')
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, t)
                    else
                        #work = work*t
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, t)
                    end
                    
                    #c1 = c1 - w*v1^H
                    if n > k
                        LinearAlgebra.generic_matmatmul!(c1, 'N', 'C', work, v1, minus)
                    end

                    #work = work*(v2')
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v2)

                    c2 .-= work
                end
            end
        end
    else 
        if storev == 'R'
            if direct == 'F'
                """
                Let V = (V1 V2) (v1: first k columns)
                where v1 is unit upper triangular
                """

                if side == 'L'
                    """
                    Form H*C or H^H*C where C = (c1)
                                                (c2)
                    """

                    v1 = @view v[:, 1:k]
                    v2 = @view v[:, k+1:m]
                    c1 = @view c[1:k, :]
                    c2 = @view c[k+1:m, :]

                    work .= c1'

                    #work = work*(v1')
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v1)

                    if m > k
                        #work = work + (c2')*(v2')
                        LinearAlgebra.generic_matmatmul!(work, 'C', 'C', c2, v2, plus)
                    end

                    if trans == 'N'
                        #work = work*(t')
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, t)
                    else
                        #work = work*t
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, t)
                    end

                    #c2 = c2 - v2^h*w^h
                    if m > k
                        LinearAlgebra.generic_matmatmul!(c2, 'C', 'C', v2, work, minus)
                    end

                    #work = work*v1
                    LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v1)

                    c1 .-= work'

                else 
                    if side == 'R' || side == 'r'
                        """
                        Form C*H or C*H^H where C = (c1 c2)
                        """
                        
                        v1 = @view v[:, 1:k]
                        v2 = @view v[:, k+1:n]
                        c1 = @view c[:, 1:k]
                        c2 = @view c[:, k+1:n]

                        work .= c1

                        #work = work*(v1')
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'U', adjoint, work, v1)

                        if n > k
                            #work = work + c2*(v2')
                            LinearAlgebra.generic_matmatmul!(work, 'N', 'C', c2, v2, plus)
                        end

                        if trans == 'C'
                            #work = work*(t')
                            LinearAlgebra.generic_mattrimul!(work, 'U', 'N', adjoint, work, t)
                        else
                            #work = work*t
                            LinearAlgebra.generic_mattrimul!(work, 'U', 'N', identity, work, t)
                        end

                        #c2 = c2 - w*v2
                        if n > k
                            LinearAlgebra.generic_matmatmul!(c2, 'N', 'N', work, v2, minus)
                        end

                        #work = work*v1
                        LinearAlgebra.generic_mattrimul!(work, 'U', 'U', identity, work, v1)
                        
                        c1 .-= work
                    end
                end
            else # direct = B
                """
                Let V = (v1  v2) (v2: last k columns)
                where v2 is unit lower triangular
                """
                if side == 'L' || side == 'l'
                    """
                    Form H*C or H^H*C where C = (c1)
                                                (c2)
                    """
                    v1 = @view v[:, 1:m-k]
                    v2 = @view v[:, m-k+1:m]
                    c1 = @view c[1:m-k,:]
                    c2 = @view c[m-k+1:m,:]

                    work .= c2'

                    #work = work * (v2')
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v2)
                    
                    if m > k
                        #work = work + (c1')*(v1')
                        LinearAlgebra.generic_matmatmul!(work, 'C', 'C', c1, v1, plus)
                    end

                    if trans == 'N'
                        #work = work*(t')
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, t)
                    else
                        #work = work*t
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, t)
                    end

                    #c1 = c1 - v1^h * w^h
                    if m > k
                        LinearAlgebra.generic_matmatmul!(c1, 'C', 'C', v1, work, minus)
                    end

                    #work = work*v2
                    LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v2)
                
                    c2 .-= work'

                else 
                    if side == 'R'
                        """
                        Form C*H or C*H^H where C = (c1 c2)
                        """
                        v1 = @view v[:, 1:n-k]
                        v2 = @view v[:, n-k+1:n]
                        c1 = @view c[:, 1:n-k]
                        c2 = @view c[:,n-k+1:n]

                        work .= c2
                        
                        #work = work * (v2')
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'U', adjoint, work, v2)

                        if n > k
                            #work = work + c1*(v1')
                            LinearAlgebra.generic_matmatmul!(work, 'N', 'C', c1, v1, plus)
                        end

                        if trans == 'C'
                            #work = work*(t')
                            LinearAlgebra.generic_mattrimul!(work, 'L', 'N', adjoint, work, t)
                        else
                            #work = work*t
                            LinearAlgebra.generic_mattrimul!(work, 'L', 'N', identity, work, t)
                        end

                        #c1 = c1 - w*v1
                        if n > k
                            LinearAlgebra.generic_matmatmul!(c1, 'N', 'N', work, v1, minus)
                        end

                        #work = work*v2
                        LinearAlgebra.generic_mattrimul!(work, 'L', 'U', identity, work, v2)

                        c2 .-= work
                    end
                end
            end
        end
    end

    return
end 