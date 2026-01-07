
"""
Function Docstring from lapack

!>
!> SLASET initializes an m-by-n matrix A to BETA on the diagonal and
!> ALPHA on the offdiagonals.
!> 


#Arguments
UPLO
    !>          UPLO is CHARACTER*1
    !>          Specifies the part of the matrix A to be set.
    !>          = 'U':      Upper triangular part is set; the strictly lower
    !>                      triangular part of A is not changed.
    !>          = 'L':      Lower triangular part is set; the strictly upper
    !>                      triangular part of A is not changed.
    !>          Otherwise:  All of the matrix A is set.

M
    !>          M is INTEGER
    !>          The number of rows of the matrix A.  M >= 0.
    !> 

N
    !>          N is INTEGER
    !>          The number of columns of the matrix A.  N >= 0.
    !> 

ALPHA
    !>          ALPHA is REAL
    !>          The constant to which the offdiagonal elements are to be set.
    !> 

BETA
    !>          BETA is REAL
    !>          The constant to which the diagonal elements are to be set.
    !>
    
A
    !>          A is REAL array, dimension (LDA,N)
    !>          On exit, the leading m-by-n submatrix of A is set as follows:
    !>
    !>          if UPLO = 'U', A(i,j) = ALPHA, 1<=i<=j-1, 1<=j<=n,
    !>          if UPLO = 'L', A(i,j) = ALPHA, j+1<=i<=m, 1<=j<=n,
    !>          otherwise,     A(i,j) = ALPHA, 1<=i<=m, 1<=j<=n, i.ne.j,
    !>
    !>          and, for all UPLO, A(i,i) = BETA, 1<=i<=min(m,n).
    !> 

LDA
    !>          LDA is INTEGER
    !>          The leading dimension of the array A.  LDA >= max(1,M).
    !> 
"""
function slaset!(uplo::Char, m::Integer, n::Integer, alpha::T, beta::T, A::AbstractMatrix{T}, lda::Integer) where T{<:AbstractFloat}
    # It should be noted that the size of the dimension of A is m x n.
    #   lda is used to figure out the how many positions in memory to jump between the first row in column i
    #   and the first row in column i+1, so essentially the actual in memory column length. It's possible for this
    #   to not line up with the number of rows m if A is a submatrix, or for memory alignment reasons, etc.

end

function slartg!(g::T, f::Ref{T},  c::Ref{T}, s::Ref{T}, r::Ref{T}) where {T<:AbstractFloat}

    #Templated for T to be a Float

    rtmin = sqrt(floatmin(T))
    rtmax = sqrt(floatmax(T)/2)

    f1 = abs(f[])
    g1 = abs(g)

    if g == 0
        c[] = one(T)
        s[] = zero(T)
        r[] = f[]
    
    elseif f[] == 0
        f[] = zero(T)
        s[] = sign(g)
        r[] = g1
    elseif f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax
        d = sqrt(f[]*f[] + g*g)
        c[] = f1/d
        r[] = copysign(d, f[])
        s[] = g/r[]
    else
        u = min(safmax, max(floatmin(T), f1, g1))
        fs = f[]/u
        gs = g/u
        d = sqrt(fs*fs + gs*gs)
        c[] = abs(fs) / d
        r[] = copysign(d, f[])
        s[] = gs/r
        r[] = r[]*u
    end
    return
end




function sbdsdc!(uplo, compq, n, d e, u, ldu, vt, ldvt, q, iq, work, iwork, info)
    

    #Initializing variables that will be used to give info about output
    info = 0
    iuplo = 0
    ONE = 1.0
    ZERO = 0.0
    TWO = 2.0

    #Checks if it is an upper or lower traingular
    if uplo == 'U'
        iuplo = 1
    elseif uplo == 'L'
        iuplo = 2
    end

    if compq == 'N'
        icompq = 10
    elseif compq == 'P'
        icompq = 1
    elseif compq == 'I'
        icompq = 2
    else
        icompq = -1
    end

    #Check for any errors and early return if any
    if iuplo == 0 
        info = -1
    end
    elseif icompq < 0
        info  -1
    elseif n < 0
        info = -3
    elseif (ldu < 1) || (icompq == 2 && ldu < n )
        info = -7
    elseif (ldvt < 1) || (icompq == 2 && ldvt < n)
        info = -9
    end

    if info != 0
        info = -info
        return
    end

    if n == 0
        return
    end

    # As defined at the linke bellow on line 713
    # https://www.netlib.org/lapack/explore-html/db/df3/group__ilaenv_gaa9a5648b5b1506869105554acf4f4b13.html
    smlsiz = 25

    if n == 1
        if icompq == 1
            q[0] = sign(d[1])
            q[smlsiz*n] = ONE
        elseif icompq == 2
            u[0,0] = sign(d[1])
            vt[0,0] = ONE
        end
        d[1] = abs(d[1])
        return
    end

    nm1 = n-1

    # If matrix lower bidiagonal, rotate to be upper bidiagonal
    # by applying Givens rotations on the left

    wstart = ONE
    qstart = ONE
    cs = Ref(Float32(0.0))
    sn = Ref(Float32(0.0))
    r = Ref(Float32(0.0))

    if icompq == 1
        copyto!(d,q[0])
        copyto!(e,q[n])
        # scopy!(n, d, 1, q[0], 1)
        # scopy!(n-1, e, 1, q[n], 1)
    end

    if iuplo == 2
        qstart = 5

        if icompq == 2
            wstart = 2*n -1
        end

        for i in 0 : n-2

            slartg!(@view e[i],@view d[i],  cs, sn, r)
            d[i] = r
            e[i] = sn * d[i+1]
            d[i+1] = cs * d[i+1]

            if icompq == 1
                q[i + 2*n] = cs
                q[i+3*n] = sn
            elseif icompq == 2
                work[i] = cs
                work[nm1 + i] = -sn
            end
        end
    end

    # If ICOMPQ = 0, use SLASDQ to compute the singular values.

    #Need to implement slasdq and slaset functions or call BLAS function
    if icompq == 0

        slasdq!('U', 0, n, 0, 0, 0, d, e, vt, ldvt, u, ldu, u, ldu, work[0], info)
    
    elseif n <= smlsiz

        if icompq == 2
            slaset!('A', n, n, ZERO, ONE, u, ldu)
            slaset!('A', n, n, ZERO, ONE, vt, ldvt)
            slasdq!('U', 0, n, n, n, 0, d, e, vt, ldvt, u, ldu, u, ldu, work[wstart-1], info)
            
        elseif icompq == 1
            iu = 1
            ivt = iu + N
            slaset!('A', n, n, ZERO, ONE, q[iu + (qstart-1)*n-1], n)
            slaset!('A', n, n, ZERO, ONE, q[ivt + (qstart-1)*n-1], n)
            slasdq!('U', 0, n, n, n, 0, d, e, q[ivt + (qstart-1)*n-1], n,
             q[iu + (qstart-1)*n-1], n, 
             q[iu + (qstart-1)*n-1], n, work[wstart-1], info)
        end

    else

        if icompq == 2
            slaset!('A', n, n, ZERO, ONE, u, ldu)
            slaset!('A', n, n, ZERO, ONE, vt, ldvt)
        end

        #Need to implement this function. It is a matrix norm function. May already exist
        orgnrm = slanst('M', n, d, e)

        #This gives the machine epsilon for float64. Need to verify that's what we want or if we should template this to
        #   accept multiple number types
        eps = eps()



    end

end
