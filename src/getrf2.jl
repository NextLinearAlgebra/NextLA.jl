using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools


function lapack_getrf2!(::Type{T}, A::AbstractMatrix{T}, ipiv::AbstractVector{Int}) where {T<:Number}
    m,n = size(A)
    lda = max(1,m)
    info = Ref{BlasInt}()

    if T == ComplexF64
        ccall((@blasfunc(zgetrf2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            m, n, A, lda, ipiv, info)

    elseif T == Float64
        ccall((@blasfunc(dgetrf2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            m, n, A, lda, ipiv, info)

    elseif T == ComplexF32
        ccall((@blasfunc(cgetrf2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            m, n, A, lda, ipiv, info)

    else #T  = Float32
        ccall((@blasfunc(sgetrf2_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            m, n, A, lda, ipiv, info)
    end
end

"""
    getrf2!(A::AbstractMatrix{T}, ipiv::AbstractVector{Int}, info::Ref{Int})

Computes an LU factorization of a general M-by-N matrix A using partial pivoting with row exchanges.
The factorization has form 
    A = P * L * U
where P is a permutation matrix, L is lower triangula with unit diagonal elements (lower trapezoidal if m > n),
and U is upper triangular (upper trapezoidal if m < n)

This is the recursive form of the algorithm. 

# Arguments
- 'A' : matrix, dimension (m,n)
    - On entry, the m-by-n matrix to be factored
    - On exit, the factors L and U from the factorization A = P * L * U; the unit diagonal elements of L are not stored

- 'ipiv' : dimension (min(m,n))
    - the pivot indicies; for 1 <= i <= min(m,n), row i of the matrix was interchanged with row ipiv[i]

- 'info' : 
    - =0: successful exit
    - <0: if info = -i, the i-th argument had an illegal value
    - >0: if info = i, U[i,i] is exactly zero. The factorization has been completed, but the factor U is exactly singular and division by zero will occur if it is used to solve a system of equations
"""
function getrf2!(A::AbstractMatrix{T}, ipiv::AbstractVector{Int}, info::Ref{Int}) where T
    m, n = size(A)

    lda = m
    info[] = 0

    if m < 0
        info[] = -1
        return
    end
    if n < 0
        info[] = -2
        return
    end
    if lda < max(1,m)
        info[] = -4
        return
    end

    # quick return
    if m == 0 || n == 0
        return
    end

    if m == 1
        
        ipiv[1] = 1

        if A[1,1] == zero(T)
            info[] = 1
            return
        end

    elseif n == 1
        sfmin = lamch(eltype(real(A[1,1])), 'S')

        dmax = abs(real(A[1,1])) + abs(imag(A[1,1]))
        idamax = 1

        for i = 2:m
            if abs(real(A[i,1])) + abs(imag(A[i,1])) > dmax
                idamax = i
                dmax = abs(real(A[i,1])) + abs(imag(A[i,1]))
            end
        end

        ipiv[1] = idamax

        if A[idamax,1] != zero(T)
            #Apply the interchange
            if idamax != 1
                temp = A[1,1]
                A[1,1] = A[idamax,1]
                A[idamax,1] = temp
            end

            #Compute element 2:m of the column
            if abs(A[1, 1]) >= sfmin
                BLAS.scal!(m-1, one(T)/A[1,1], view(A, 2:m, 1), 1)
            else
                view(A, 2:m, 1) ./= A[1,1]
            end
        else
            info[] = 1
            return
        end
    else
        #use recursive code
        n1 = div(min(m,n), 2)
        n2 = n - n1

        #          [A11]
        #Factor    [---]
        #          [A12]

        iinfo = Ref{Int}(0)
        Aleft = @view A[:, 1:n1]

        getrf2!(Aleft, ipiv, iinfo)

        if info[] == 0 && iinfo[] > 0
            info[] = iinfo[]
        end

        # Apply interchanges to [A12]
        #                       [---]
        #                       [A22]
        laswp(view(A, :, n1+1:n), Int(1), Int(n1), ipiv, Int(1))

        # Solve A12
        LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'U', one(T), (@view A[1:n1, 1:n1]), (@view A[1:n1, n1+1:n]))

        # Update A22
        LinearAlgebra.BLAS.gemm!('N', 'N', -one(T), view(A, n1+1:m, 1:n1), view(A, 1:n1, n1+1:n), one(T), view(A, n1+1:m, n1+1:n))

        #Factor A22
        iinfo = Ref{Int}(0)
        getrf2!(view(A, n1+1:m, n1+1:n), view(ipiv, n1+1:min(m,n)), iinfo)

        #Adjust INFO and pivot indicies
        if info[] == 0 && iinfo[] > 0
            info[] = iinfo[] + n1
        end
        
        for i = n1+1: min(m,n)
            ipiv[i] += n1
        end

        #Apply interchanges to A21
        laswp(view(A, :, 1:n1), Int(n1+1), Int(min(m,n)), ipiv, Int(1))
    end

    return A, ipiv, info[]
end

function laswp(A::AbstractMatrix{T}, first::Integer, last::Integer, ipiv::AbstractVector{Int}, incx::Integer) where T
    m, n = size(A)
    k1 = first
    k2 = last
    if incx > 0
        ix0 = k1
        i1 = k1
        i2 = k2
        inc = Int(1)
    elseif incx < 0
        ix0 = 1+ (1 - k2) * incx
        i1 = k2
        i2 = k1
        inc = Int(-1)
    else
        return
    end

    n32 = (n ÷ 32) * 32

    if n32 != 0
        for j in 1:32:n32
            ix = ix0
            for i in i1:inc:i2
                ip = ipiv[ix]
                if ip != i
                    for k in j:j+31
                        temp = A[i, k]
                        A[i, k] = A[ip, k]
                        A[ip, k] = temp
                    end
                end
                ix += inc
            end
        end
    end
    if n32 != n
        n32 +=1
        ix = ix0
        for i in i1:inc:i2
            ip = ipiv[ix]
            if ip != i
                for k in n32:n
                    temp = A[i, k]
                    A[i, k] = A[ip, k]
                    A[ip, k] = temp
                end
            end
            ix += inc
        end
    end
    return
end
