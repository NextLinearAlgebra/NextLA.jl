using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools


function lapack_getc2!(A::AbstractMatrix{T},  ipiv::AbstractVector{Int}, jpiv::AbstractVector{Int}) where {T<: Number}
    lda, n = size(A)
    info = Ref{BlasInt}(0)

    if T == ComplexF64
        ccall((@blasfunc(zgetc2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt},Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            n,A,lda, ipiv, jpiv, info)

    elseif T == Float64
        ccall((@blasfunc(dgetc2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt},Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            n,A,lda, ipiv, jpiv, info)

    elseif T == ComplexF32
        ccall((@blasfunc(cgetc2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt},Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            n,A,lda, ipiv, jpiv, info)

    else #T  = Float32
        ccall((@blasfunc(sgetc2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt},Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            n,A,lda, ipiv, jpiv, info)
    end
end

"""
    getc2!(A::AbstractMatrix{T}, ipiv::AbstractVector{Int}, jpiv::AbstractVector{Int}, info::Ref{Int})

Computes an LU factorization with complete pivoting of the n-by-n matrix A. The factorization has the form A = P * L * U * Q,
where P and Q are permutation matrices, L is lower triangular with unit diagonal elements and U is upper triangular.

# Arguments
- 'A' : matrix, dimension (n,n)
    - on entry, the n-by-n matrix A to be factored
    - on exit, the factors L and U from the factorization A = P * L * U * Q; the unit diagonal elements of L are not stored. 
    - If U(k,k) appears to be less than smin, U(k,k) is given the value smin, i.e, given a nonsingular pertubed system

- 'ipiv': dimension (n)
    - the pivot indicies; for 1 <= i <= n, row i of the matrix has been interchanged with row ipiv[i]

- 'jpiv': dimension (n)
    - the pivot indicies; for 1 <= j <= n, column j of the matrix has been interchanged with column jpiv[i]

- 'info': 
    - =0: successful exit
    - >0: if info = k, U(k,k) is likely to produce overflow if we try to solve for x in Ax = b, so U is pertubed to avoid overflow
"""
function getc2!(A::AbstractMatrix{T}, ipiv::AbstractVector{Int}, jpiv::AbstractVector{Int}, info::Ref{Int}) where T
    lda, n = size(A)
    info[] = 0
    
    realt = typeof(real(A[1,1]))
    if n == 0
        return
    end

    ep = lamch(realt, 'P')
    smlnum  = lamch(realt, 'S') / ep
    bignum = one(realt) / smlnum

    if log10(bignum) > realt(2000)
        smlnum = sqrt(smlnum)
        bignum = sqrt(bignum)
    end

    if n == 1
        ipiv[1] = 1
        jpiv[1] = 1

        if abs(A[1,1]) < smlnum
            info[] = 1
            A[1,1] = T(smlnum)
        end

        return
    end

    # factorize A using complete pivoting
    # set pivots less than SMIN to SMIN
    smin = zero(realt)
    
    for i in 1:n-1
        # find the max element in matrix A

        xmax = zero(realt)
        ipv = i
        jpv = i

        for ip in i:n
            for jp in i:n
                if abs(A[ip, jp]) >= xmax
                    xmax = abs(A[ip,jp])
                    ipv = ip
                    jpv = jp
                end
            end
        end

        if i == 1
            smin = max(ep*xmax, smlnum)
        end

        #swap rows

        if ipv != i
            for j in 1:n
                temp = A[ipv, j]
                A[ipv, j] = A[i,j]
                A[i,j] = temp
            end
        end

        ipiv[i] = ipv

        #swap columns

        if jpv != i
            for j in 1:n
                temp = A[j, jpv]
                A[j,jpv] = A[j,i]
                A[j,i] = temp
            end
            
        end

        jpiv[i] = jpv

        #check for singularity

        if abs(A[i,i]) < smin
            info[] = i
            A[i,i] = T(smin)
        end

        (@view A[i+1:n, i]) ./= A[i,i]

        geru!(-one(T), (@view A[i+1: n, i]), (@view A[i, i+1:n]),  (@view A[i+1:n, i+1:n]))
    end

    if abs(A[n,n]) < smin
        info[] = n
        A[n,n] = T(smin)
    end

    ipiv[n] = n
    jpiv[n] = n

    return
end


function blas_geru!(alpha::T, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where T
    m,n = size(A)
    
    px, stx = LinearAlgebra.BLAS.vec_pointer_stride(x, ArgumentError("input vector with 0 stride is not allowed"))
    py, sty = LinearAlgebra.BLAS.vec_pointer_stride(y, ArgumentError("input vector with 0 stride is not allowed"))

    if T == ComplexF64
        GC.@preserve x y ccall((@blasfunc(zgeru_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m, n, alpha, px, stx, py, sty, A, max(1,stride(A,2)))

    elseif T == Float64
        GC.@preserve x y ccall((@blasfunc(dger_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
        m, n, alpha, px, stx, py, sty, A, max(1,stride(A,2)))

    elseif T == ComplexF32
        GC.@preserve x y ccall((@blasfunc(cgeru_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
        m, n, alpha, px, stx, py, sty, A, max(1,stride(A,2)))

    else #T  = Float32
        GC.@preserve x y ccall((@blasfunc(sger_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
        m, n, alpha, px, stx, py, sty, A, max(1,stride(A,2)))
    end

end
