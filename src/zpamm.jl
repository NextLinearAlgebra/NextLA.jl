function zpamm(op, side, storev, direct, m, n, k, l, A1, lda1, A2, lda2, V, ldv, W, ldw)
    # Input validation
    op ∉ ('W', 'A') && throw(ArgumentError("illegal value of op"))
    side ∉ ('L', 'R') && throw(ArgumentError("illegal value of side"))
    storev ∉ ('C', 'R') && throw(ArgumentError("illegal value of storev"))
    direct ∉ ('F', 'B') && throw(ArgumentError("illegal value of direct"))
    
    # Dimension validation
    m < 0 && throw(ArgumentError("illegal value of m"))
    n < 0 && throw(ArgumentError("illegal value of n"))
    k < 0 && throw(ArgumentError("illegal value of k"))
    l < 0 && throw(ArgumentError("illegal value of l"))
    
    # Leading dimension validation
    lda1 < 0 && throw(ArgumentError("illegal value of lda1"))
    lda2 < 0 && throw(ArgumentError("illegal value of lda2"))
    ldv < 0 && throw(ArgumentError("illegal value of ldv"))
    ldw < 0 && throw(ArgumentError("illegal value of ldw"))
    
    # Quick return for degenerate cases
    (m == 0 || n == 0 || k == 0) && return nothing

    if direct == 'F'
        forward = true
    else
        forward = false
    end

    if storev == 'C'
        colmajor = true
    else
        colmajor = false
    end

    if side == 'L'
        left = true
    else
        left = false
    end
    

    if op == 'W'
        zpamm_w(left, colmajor, forward, m,n,k,l, A1, A2, V, W)
    else
        zpamm_a(left, colmajor, forward, m,n,k,l, A2, V, W)
    end
    
    return 
end

function zpamm_w(left, colmajor, forward, m, n, k, l, A1, A2, V, W)
    # W = A1 + op(V) * A2 or W = A1 + A2 * op(V)
    one0 = oneunit(eltype(A1))
    zero0 = zero(eltype(A1))
    plus = LinearAlgebra.MulAddMul(one0, one0)
    eqa = LinearAlgebra.MulAddMul(one0, zero0)

    if colmajor && forward && left # colmajor, forward, left
        mp = min(m-l+1, m)
        kp = min(l+1, k)

        for i in 1:l
            copyto!((@view W[i, 1:n]), (@view A2[m-l + i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'U', 'N', adjoint, (@view V[mp:mp+l-1, 1:l]), (@view W[1:l, 1:n]))
    
        LinearAlgebra.generic_matmatmul!((@view W[1:l, 1:n]), 'C', 'N', (@view V[1:m-l, 1:l]), (@view A2[1:m-l, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+k-l-1, 1:n]), 'C', 'N', (@view V[1:m, kp:kp+k-l-1]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end

    end
    if !colmajor && forward && left # rowmajor, forward, left
        mp = min(m-l+1, m)
        kp = min(l+1, k)

        for i in 1:l
            copyto!((@view W[i, 1:n]), (@view A2[m-l+i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'L', 'N', identity, (@view V[1:l, mp:mp+l-1]), (@view W[1:l, 1:n]))

        LinearAlgebra.generic_matmatmul!((@view W[1:l, 1:n]), 'N', 'N', (@view V[1:l, 1:m-l]), (@view A2[1:m-l, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+k-l-1, 1:n]), 'N', 'N', (@view V[kp:kp+k-l-1, 1:m]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end
    end
    if colmajor && forward && !left # colmajor, forward, right
        np = min(n-l+1, n)
        kp = min(l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, j]), (@view A2[1:m, n-l+j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'U', 'N', identity, (@view W[1:m, 1:l]), (@view V[np:np+l-1, 1:l]))

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:l]), 'N', 'N', (@view A2[1:m, 1:n-l]), (@view V[1:n-l, 1:l]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+k-l-1]), 'N', 'N', (@view A2[1:m, 1:n]), (@view V[1:n, kp:kp+k-l-1]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
    if !colmajor && forward && !left # rowmajor, forward, right
        np = min(n-l+1, n)
        kp = min(l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, j]), (@view A2[1:m, n-l+j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'L', 'N', adjoint, (@view W[1:m, 1:l]), (@view V[1:l,  np:np+l-1]))

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:l]), 'N', 'C', (@view A2[1:m, 1:n-l]), (@view V[1:l, 1:n-l]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+k-l-1]), 'N', 'C', (@view A2[1:m, 1:n]), (@view V[kp:kp+k-l-1, 1:n]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
    if colmajor && !forward && left # colmajor, backward, left
        mp = min(l+1, m)
        kp = min(k-l+1, k)

        for i in 1:l
            copyto!((@view W[k-l + i, 1:n]), (@view A2[i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'L', 'N', adjoint, (@view V[1:l, kp:kp+l-1]), (@view W[kp:kp+l-1, 1:n]))

        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+l-1, 1:n]), 'C', 'N', (@view V[mp:mp+m-l-1, kp:kp+l-1]), (@view A2[mp:mp+m-l-1, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:k-l, 1:n]), 'C', 'N', (@view V[1:m, 1:k-l]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end
    end

    if !colmajor && !forward && left # rowmajor, backward, left       
        mp = min(l+1, m)
        kp = min(k-l+1, k)

        for i in 1:l 
            copyto!((@view W[k-l + i, 1:n]), (@view A2[i, 1:n]))
        end

        LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'L', 'N', identity, (@view V[kp:kp+l-1, 1:l]), (@view W[kp:kp+l-1, 1:n]))
        
        LinearAlgebra.generic_matmatmul!((@view W[kp:kp+l-1, 1:n]), 'N', 'N', (@view V[kp:kp+l-1, mp:mp+m-l-1]), (@view A2[mp:mp+m-l-1, 1:n]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:k-l, 1:n]), 'N', 'N', (@view V[1:k-l, 1:m]), (@view A2[1:m, 1:n]), eqa)

        for i in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[i, 1:n]), (@view W[i, 1:n]))
        end      
    end
    if !colmajor && !forward && !left # rowmajor, backward, right
        np = min(l+1, n)
        kp = min(k-l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, k-l+j]), (@view A2[1:m, j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'U', 'N', adjoint, (@view W[1:m, kp:kp+l-1]), (@view V[kp:kp+l-1, 1:l]))
        
        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+l-1]), 'N', 'C', (@view A2[1:m, np:np+n-l-1]), (@view V[kp:kp+l-1, np:np+n-l-1]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:k-l]), 'N', 'C', (@view A2[1:m, 1:n]), (@view V[1:k-l, 1:n]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
    if colmajor && !forward && !left # colmajor, backward, right
        np = min(l+1, n) 
        kp = min(k-l+1, k)

        for j in 1:l
            copyto!((@view W[1:m, k-l+j]), (@view A2[1:m, j]))
        end

        LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'L', 'N', identity, (@view W[1:m, kp:kp+l-1]), (@view V[1:l, kp:kp+l-1]))
        
        LinearAlgebra.generic_matmatmul!((@view W[1:m, kp:kp+l-1]), 'N', 'N', (@view A2[1:m, np:np+n-l-1]), (@view V[np:np+n-l-1, kp:kp+l-1]), plus)

        LinearAlgebra.generic_matmatmul!((@view W[1:m, 1:k-l]), 'N', 'N', (@view A2[1:m, 1:n]), (@view V[1:n, 1:k-l]), eqa)

        for j in 1:k
            LinearAlgebra.axpy!(one0, (@view A1[1:m, j]), (@view W[1:m, j]))
        end
    end
   
    return
end


function zpamm_a(left, colmajor, forward, m, n, k, l, A2, V, W)
        # A2 = A2 + op(V) * W or A2 = A2 + W * op(V)
        one0 = oneunit(eltype(A2))
        minus = LinearAlgebra.MulAddMul(one0*(-1),one0)

        if colmajor && forward && left # colmajor, forward, left
            mp = min( m-l+1, m )
            kp = min( l+1, k )

            LinearAlgebra.generic_matmatmul!((@view A2[1:m-l, 1:n]), 'N', 'N', (@view V[1:m-l, 1:k]), (@view W[1:k, 1:n]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+l-1, 1:n]), 'N', 'N', (@view V[mp:mp+l-1, kp:kp+k-l-1]), (@view W[kp:kp+k-l-1, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'U', 'N', identity, (@view V[mp:mp+k-l-1, 1:k-l]), (@view W[1:l, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[i, 1:n]), (@view A2[m-l+i, 1:n]))
            end
        end

        if !colmajor && forward && left # rowmajor, forward, left
            mp = min(m-l+1, m)
            kp = min(l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m-l, 1:n]), 'C', 'N', (@view V[1:k, 1:m-l]), (@view W[1:k, 1:n]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+l-1, 1:n]), 'C', 'N', (@view V[kp:kp+k-l-1, mp:mp+l-1]), (@view W[kp:kp+k-l-1, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[1:l, 1:n]), 'L', 'N', adjoint, (@view V[1:l, mp:mp+l-1]), (@view W[1:l, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!((-one0), (@view W[i, 1:n]), (@view A2[m-l+i, 1:n])) 
            end
        end


        if colmajor && forward && !left # colmajor, forward, right
            np = min(n-l+1, n)
            kp = min(l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n-l]), 'N', 'C', (@view W[1:m, 1:k]), (@view V[1:n-l, 1:k]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+l-1]), 'N', 'C', (@view W[1:m, kp:kp+k-l-1]), (@view V[np:np+l-1, kp:kp+k-l-1]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'U', 'N', adjoint, (@view W[1:m, 1:l]), (@view V[np:np+l-1, 1:l]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[1:m, j]), (@view A2[1:m, n-l+j]))
            end
        end

        if !colmajor && forward && !left # rowmajor, forward, right
            np = min(n-l+1, n)
            kp = min(l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:n-l]), 'N', 'N', (@view W[1:m, 1:k]), (@view V[1:k, 1:n-l]), minus)
 
            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+l-1]), 'N', 'N', (@view W[1:m, kp:kp+k-l-1]), (@view V[kp:kp+k-l-1, np:np+l-1]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, 1:l]), 'L', 'N', identity, (@view W[1:m, 1:l]), (@view V[1:l, np:np+l-1]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[1:m, j]), (@view A2[1:m, n-l+j]))
            end
        end

        if colmajor && !forward && left # colmajor, backward, left
            mp = min(l+1, m)
            kp = min(k-l+1, k)
            
            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+m-l-1, 1:n]), 'N', 'N', (@view V[mp:mp+m-l-1, 1:k]), (@view W[1:k, 1:n]), minus)
            
            LinearAlgebra.generic_matmatmul!((@view A2[1:l, 1:n]), 'N', 'N', (@view V[1:l, 1:k-l]), (@view W[1:k-l, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'L', 'N', identity, (@view V[1:l, kp:kp+l-1]), (@view W[kp:kp+l-1, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[k-l+i, 1:n]), (@view A2[i, 1:n]))
            end
        end
        
        if !colmajor && !forward && left # rowmajor, backward, left
            mp = min(l+1, m)
            kp = min(k-l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[mp:mp+m-l-1, 1:n]), 'C', 'N', (@view V[1:k, mp:mp+m-l-1]), (@view W[1:k, 1:n]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:l, 1:n]), 'C', 'N', (@view V[1:k-l, 1:l]), (@view W[1:k-l, 1:n]), minus)

            LinearAlgebra.generic_trimatmul!((@view W[kp:kp+l-1, 1:n]), 'L', 'N', adjoint, (@view V[kp:kp+l-1, 1:l]), (@view W[kp:kp+l-1, 1:n]))

            for i in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[k-l+i, 1:n]), (@view A2[i, 1:n]))
            end
        end
        
        if !colmajor && !forward && !left # rowmajor, backward, right
            np = min(l+1, n)
            kp = min(k-l+1, k)
            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+n-l-1]), 'N', 'N', (@view W[1:m, 1:k]), (@view V[1:k, np:np+n-l-1]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:l]), 'N', 'N', (@view W[1:m, 1:k-l]), (@view V[1:k-l, 1:l]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'U', 'N', identity, (@view W[1:m, kp:kp:kp+l-1]), (@view V[kp:kp+l-1, 1:l]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[1:m, k-l+j]), (@view A2[1:m, j]))
            end
        end
        
        if colmajor && !forward && !left # colmajor, backward, right
            np = min(l+1, n)
            kp = min(k-l+1, k)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, np:np+n-l-1]), 'N', 'C', (@view W[1:m, 1:k]), (@view V[np:np+n-l-1, 1:k]), minus)

            LinearAlgebra.generic_matmatmul!((@view A2[1:m, 1:l]), 'N', 'C', (@view W[1:m, 1:k-l]), (@view V[1:l, 1:k-l]), minus)

            LinearAlgebra.generic_mattrimul!((@view W[1:m, kp:kp+l-1]), 'L', 'N', adjoint, (@view W[1:m, kp:kp:kp+l-1]), (@view V[1:l, kp:kp+l-1]))

            for j in 1:l 
                LinearAlgebra.axpy!(-one0, (@view W[1:m, k-l+j]), (@view A2[1:m, j]))
            end
        end

        return
end
  

