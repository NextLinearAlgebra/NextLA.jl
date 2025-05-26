using LinearAlgebra 
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools


function lapack_tpmqrt!(::Type{T}, side::Char, trans::Char, l::Int64, V::AbstractMatrix{T}, 
    Tau::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<: Number}

    m, n = size(B)
    nb, k = size(Tau)
    minmn = min(m, n)

    if nb > minmn
        throw(ArgumentError("block size $nb > $minmn too large"))
    end

    ldv = max(1, stride(V,2))
    ldt = max(1, stride(Tau,2))
    lda = max(1, stride(A,2))
    ldb = max(1, stride(B,2))

    if side == 'L'
        work = Vector{T}(undef, n*nb)
    else
        work = Vector{T}(undef, m*nb)
    end
    
    info = Ref{BlasInt}()
  
    if n > 0
        if T == ComplexF64
            ccall((@blasfunc(ztpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])
        elseif T == Float64
                
            ccall((@blasfunc(dtpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])

        elseif T == ComplexF32
            ccall((@blasfunc(ctpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])
        else # T = Float32
            ccall((@blasfunc(stpmqrt_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, 
            Ref{BlasInt}, Ref{BlasInt},  Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ptr{BlasInt}),
            side, trans, m, n, k, l, nb, V, ldv, Tau, ldt, A, lda,
            B, ldb, work, info)

            chklapackerror(info[])
        end
    end
end


function ztsmqr(side, trans, m1, n1, m2, n2, k, ib, 
    A1, lda1, A2, lda2, V, ldv, T, ldt, work, ldwork)

    #check input arguments
    if side != 'L' && side != 'R'
        throw(ArgumentError("illegal value of side"))
        return -1
    end

    if trans != 'N' && trans != 'C' && trans != 'T'
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

    if m1 < 0 || (m2 != m1 && side == 'R')
        throw(ArgumentError("illegal value of m2"))
        return -5
    end

    if n2 < 0 || (n2 != n1 && side == 'L')
        throw(ArgumentError("illegal value of n2"))
        return -6
    end

    if k < 0 || (side == 'L' && k > m1) || (side == 'R' && k > n1)
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

    if (side == 'L' && ldv < max(1,m2)) || (side == 'R' && ldv < max(1,n2))
        throw(ArgumentError("illegal value of ldv"))
        return -14
    end

    if ldt < max(1,ib)
        throw(ArgumentError("illegal value of ldt"))
        return -16
    end

    if (side == 'L' && ldwork < max(1,ib)) || (side == 'R' && ldwork < max(1,m1))
        throw(ArgumentError("illegal value of ldwork"))
        return -18
    end

    # quick return
    if m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0  || k == 0 || ib == 0
        return 
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

    for i in i1:i3:istop
        kb = min(ib, k-i+1)

        if side == 'L'
            # H  or H^H is applied to C[i:m, 1:n]
            mi = m1 - i + 1
            ic = i
            ldvv = m2
        else
            # H or H^H is applied to C[1:m, i:n]
            ni = n1- i + 1
            jc = i
            ldvv = n2
        end

        # apply H or H^H 
        zparfb(side, trans, 'F', 'C', mi, ni, m2, n2, kb, 0,
        (@view A1[ic:ic+mi-1, jc:jc+ni-1]), lda1, (@view A2[1:m2, 1:n2]), lda2, 
        (@view V[1:ldvv, i:i+kb-1]), ldvv, (@view T[1:kb, i:i+kb-1]), kb, work, ldwork)
    end
end
