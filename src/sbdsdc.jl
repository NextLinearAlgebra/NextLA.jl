using LinearAlgebra

function scopy!(n, sx, incx, sy, incy)

    if n <= 0:
        return
    end

    if inxc == 1 && incy == 1
        m = mod(n, 7)
        if m != 0
            for i in 0 : m-1
                sy[i] = sx[i]
            end
            if n < 7
                return
            end
        end

        mp1 = m + 1

        for i in mp1 - 1 : 7 : n - 1
            sy[i] = sx[i]
            sy[i + 1] = sx[i + 1]
            sy[i + 2] = sx[i + 2]
            sy[i + 3] = sx[i + 3]
            sy[i + 4] = sx[i + 4]
            sy[i + 5] = sx[i + 5]
            sy[i + 6] = sx[i + 6]
        end
    else
        ix = 1
        iy = 1

        if incx < 0 
            ix = (-n + 1)*incx + 1
        end
        if incy < 0 
            iy = (-n + 1)*incy + 1
        end

        for i in 0 : n-1
            sy[iy] = sx[ix]

            ix = ix + incx
            iy = iy + incy
        end
    end

    return
end

function slartg!(f::Ref{T}, g::Ref{T}, c::Ref{T}, s::Ref{T}, r::Ref{T}) where {T<:AbstractFloat}

    #Single precision. 
    safmin = floatmin(Float32)
    safmax = floatmax(Float32)

    rtmin = sqrt(safmin)
    rtmax = sqrt(safmax/2)

    f1 = abs(f[])
    g1 = abs(g[])

    if g[] == 0
        c[] = 1.0
        s[] = 0
        r[] = f[]
    
    elseif f[] == 0
        f[] = 0
        s[] = sign(g[])
        r[] = g1
    elseif f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax
        d = sqrt(f[]*f[] + g[]*g[])
        c[] = f1/d
        r[] = sign(d, f)
        s[] = g[]/r[]
    else
        u = min(safmax, max(safmin, f1, g1))
        fs = f/u
        gs = g/u
        d = sqrt(fs*fs + gs*gs)
        c[] = abs(fs) / d
        r[] = copysign(d, f)
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
        scopy!(n, d, 1, q[0], 1)
        scopy!(n-1, e, 1, q[n], 1)
    end

    if iuplo == 2
        qstart = 5

        if icompq == 2
            wstart = 2*n -1
        end

        for i in 0 : n-2

            slartg1(@view d[i], @view e[i], cs, sn, r)
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
