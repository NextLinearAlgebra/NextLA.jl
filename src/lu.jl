export CompletePivoting, RowMaximum, RowNonZero, NoPivot, lupivottype, lu!, getrf2!, laswp, geru!, getc2!

struct CompletePivoting <: PivotingStrategy end

struct LU{T,S<:AbstractMatrix{T},P<:AbstractVector{<:Integer}}
    factors::S
    ipiv::P
    jpiv::P
    info::LinearAlgebra.BlasInt # Can be negative to indicate failed unpivoted factorization

    function LU{T,S,P}(factors, ipiv, jpiv, info) where {T, S<:AbstractMatrix{T}, P<:AbstractVector{<:Integer}}
        new{T,S,P}(factors, ipiv, jpiv, info)
    end
end

LU(factors::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, jpiv::AbstractVector{<:Integer}, info::BlasInt) where {T} =
    LU{T,typeof(factors),typeof(ipiv)}(factors, ipiv, info)

LU{T}(factors::AbstractMatrix, ipiv::AbstractVector{<:Integer},  jpiv::AbstractVector{<:Integer}, info::Integer) where {T} =
    LU(convert(AbstractMatrix{T}, factors), ipiv, BlasInt(info))

checknozeropivot(info) = info == 0 || throw(LinearAlgebra.ZeroPivotException(info))

checknonsingular(info) = info == 0 || throw(LinearAlgebra.SingularException(info))
    
function _check_lu_success(info, allowsingular)
    if info < 0
        checknozeropivot(-info)
    else
        allowsingular || checknonsingular(info)
    end
end


# the following method is meant to catch calls to DLA lu! implmenation which is the getrf LAPACK code written in Julia
LinearAlgebra.lu!(A::DLAMatrix{<:BlasFloat}; check::Bool = true, allowsingular::Bool = false) = LinearAlgebra.lu!(A, RowMaximum(); check, allowsingular)

"""
Computes an LU factorization of a general M-by-N matrix A using partial pivoting with row exchanges.
The factorization has form 
    A = P * L * U
where P is a permutation matrix, L is lower triangula with unit diagonal elements (lower trapezoidal if m > n),
and U is upper triangular (upper trapezoidal if m < n)

This is the recursive form of the algorithm. 
"""

function LinearAlgebra.lu!(A::DLAMatrix{T}, ::RowMaximum; check::Bool = true, allowsingular::Bool = false) where {T<:BlasFloat}
    A = A.data
    m, n = size(A)
    minmn = min(m,n)

    # Initialize variables
    info = 0
    ipiv = Vector{Int}(undef, minmn)
    A, ipiv, info = getrf2!(A, ipiv::AbstractVector{Int}, info::Int)

    check && _check_lu_success(info, allowsingular)
    return LinearAlgebra.LU{T,typeof(A),typeof(ipiv)}(A, ipiv, convert(BlasInt, info))

end


LinearAlgebra.lu!(A::DLAMatrix,  pivot::Union{CompletePivoting, RowMaximum, RowNonZero, NoPivot}= lupivottype(eltype(A.data));
    check::Bool = true, allowsingular::Bool = false)  = LinearAlgebra.generic_lufact!(A, pivot; check, allowsingular) 


function LinearAlgebra.generic_lufact!(A::DLAMatrix{T}, pivot::Union{CompletePivoting, RowMaximum, RowNonZero, NoPivot} = lupivottype(T); 
    check::Bool = true, allowsingular::Bool = false)  where {T}
    
    A = A.data
    check && LAPACK.chkfinite(A)
    # Extract values

    m, n = size(A)
    minmn = min(m,n)

    # Initialize variables
    info = 0
    ipiv = Vector{Int}(undef, minmn)
    jpiv = Vector{Int}(undef, minmn)
    @inbounds begin
        for k = 1:minmn
            # Find maximum element in the submatrix A[k:m, k:n]
            kp = k
            jp = k
            if pivot === RowMaximum() && k < m
                amax = abs(A[k, k])
                for i = k+1:m
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i 
                        amax = absi
                    end
                end
            elseif pivot === RowNonZero()
                for i = k:m
                    if !iszero(A[i,k])
                        kp = i
                        break
                    end
                end
            elseif pivot === CompletePivoting()
                amax = abs(A[k, k])
                for i = k+1:m
                    for j = k+1:n
                        absi = abs(A[i,j])
                        if absi > amax
                            kp = i
                            jp = j
                            amax = absi
                        end
                    end
                end
            end

            ipiv[k] = kp
            jpiv[k] = jp
            if !iszero(A[kp,jp])
                if k != kp
                    # Interchange rows
                    A[k,:], A[kp,:] = A[kp,:], A[k,:]
                end
                if k != jp
                    # Interchange columns
                    A[:,k], A[:,jp] = A[:,jp], A[:,k]
                end

                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:m
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
            end

            # Update the rest of the matrix
            for j = k+1:n
                for i = k+1:m
                    A[i,j] -= A[i,k] * A[k,j]
                end
            end
        end
    end

    if pivot === NoPivot()
        # Use a negative value to distinguish a failed factorization (zero in pivot
        # position during unpivoted LU) from a valid but rank-deficient factorization
        info = -info
    end

    check && _check_lu_success(info, allowsingular)

    # Return LU object with row and column pivots
    return LU{T,typeof(A),typeof(ipiv)}(A, ipiv, jpiv, convert(LinearAlgebra.BlasInt, info))
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

function getrf2!(A::AbstractMatrix{T}, ipiv::AbstractVector{Int}, info::Int)  where {T<:BlasFloat}


    m, n = size(A)

    lda = m
    info = 0

    if m < 0
        info = -1
        return
    end
    if n < 0
        info = -2
        return
    end
    if lda < max(1,m)
        info = -4
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

        iinfo = 0
        Aleft = @view A[:, 1:n1]

        getrf2!(Aleft, ipiv, iinfo)

        if info == 0 && iinfo > 0
            info = iinfo
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
        iinfo = 0
        getrf2!(view(A, n1+1:m, n1+1:n), view(ipiv, n1+1:min(m,n)), iinfo)

        #Adjust INFO and pivot indicies
        if info == 0 && iinfo > 0
            info = iinfo + n1
        end
        
        for i = n1+1: min(m,n)
            ipiv[i] += n1
        end

        #Apply interchanges to A21
        laswp(view(A, :, 1:n1), Int(n1+1), Int(min(m,n)), ipiv, Int(1))
    end

    return A, ipiv, info
end

"""
lu!(A::DLAArray{T}, ::CompletePivoting; check::Bool = true,  allowsingular::Bool = false) where {T<:BlasFloat}

Computes an LU factorization with complete pivoting of the n-by-n matrix A. The factorization has the form A = P * L * U * Q,
where P and Q are permutation matrices, L is lower triangular with unit diagonal elements and U is upper triangular.

# Arguments
- 'A' : matrix, dimension (n,n)
    - on entry, the n-by-n matrix A to be factored
    - on exit, the factors L and U from the factorization A = P * L * U * Q; the unit diagonal elements of L are not stored. 
    - If U(k,k) appears to be less than smin, U(k,k) is given the value smin, i.e, given a nonsingular pertubed system

"""

function getc2!(A::DLAMatrix{T}, ::CompletePivoting; check::Bool = true,  allowsingular::Bool = false) where {T<:BlasFloat}
    A = A.data
    check && LAPACK.chkfinite(A)

    lda, n = size(A)
    m = lda
    info = 0
    minmn = min(m,n)

    ipiv = Vector{Int}(undef, minmn)
    jpiv = Vector{Int}(undef, minmn)
    
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
            info = i
            A[i,i] = T(smin)
        end

        (@view A[i+1:n, i]) ./= A[i,i]

        geru!(-one(T), (@view A[i+1: n, i]), (@view A[i, i+1:n]),  (@view A[i+1:n, i+1:n]))
    end

    if abs(A[n,n]) < smin
        info = n
        A[n,n] = T(smin)
    end

    ipiv[n] = n
    jpiv[n] = n


    check && _check_lu_success(info, allowsingular)

    # Return LU object with row and column pivots
    return LU{T,typeof(A),typeof(ipiv)}(A, ipiv, jpiv, convert(LinearAlgebra.BlasInt, info))
end

"""
    geru!(alpha::T, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T})

Performs operation A = alpha * x * y^T + A, 
where alpha is a scalar, x is an m element vector, y is an n element vector, and A is an m-by-n matrix
"""

function geru!(alpha::T, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where T
    m,n = size(A)
    # assume incy = incx = 1

    if m < 0
        return 1
    end 

    if n < 0
        return 2
    end

    if m == 0 || n == 0 || alpha == zero(T)
        return
    end

    jy = 1

    for j in 1:n
        if y[jy] != zero(T)
            temp = alpha * y[jy]
            for i in 1:m
                A[i,j] += x[i] * temp
            end
        end

        jy += 1
    end

    return 
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