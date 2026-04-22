using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

#=
Purpose:
!>
!> SLASQ3 checks for deflation, computes a shift (TAU) and calls dqds.
!> In case of failure it changes shifts, and tries again until output
!> is positive.
!> 
Parameters
[in]	I0	
!>          I0 is INTEGER
!>         First index.
!> 
[in,out]	N0	
!>          N0 is INTEGER
!>         Last index.
!> 
[in,out]	Z	
!>          Z is REAL array, dimension ( 4*N0 )
!>         Z holds the qd array.
!> 
[in,out]	PP	
!>          PP is INTEGER
!>         PP=0 for ping, PP=1 for pong.
!>         PP=2 indicates that flipping was applied to the Z array
!>         and that the initial tests for deflation should not be
!>         performed.
!> 
[out]	DMIN	
!>          DMIN is REAL
!>         Minimum value of d.
!> 
[out]	SIGMA	
!>          SIGMA is REAL
!>         Sum of shifts used in current segment.
!> 
[in,out]	DESIG	
!>          DESIG is REAL
!>         Lower order part of SIGMA
!> 
[in]	QMAX	
!>          QMAX is REAL
!>         Maximum value of q.
!> 
[in,out]	NFAIL	
!>          NFAIL is INTEGER
!>         Increment NFAIL by 1 each time the shift was too big.
!> 
[in,out]	ITER	
!>          ITER is INTEGER
!>         Increment ITER by 1 for each iteration.
!> 
[in,out]	NDIV	
!>          NDIV is INTEGER
!>         Increment NDIV by 1 for each division.
!> 
[in]	IEEE	
!>          IEEE is LOGICAL
!>         Flag for IEEE or non IEEE arithmetic (passed to SLASQ5).
!> 
[in,out]	TTYPE	
!>          TTYPE is INTEGER
!>         Shift type.
!> 
[in,out]	DMIN1	
!>          DMIN1 is REAL
!> 
[in,out]	DMIN2	
!>          DMIN2 is REAL
!> 
[in,out]	DN	
!>          DN is REAL
!> 
[in,out]	DN1	
!>          DN1 is REAL
!> 
[in,out]	DN2	
!>          DN2 is REAL
!> 
[in,out]	G	
!>          G is REAL
!> 
[in,out]	TAU	
!>          TAU is REAL
!>
!>         These are passed as arguments in order to save their values
!>         between calls to SLASQ3.
!> 
=#

function lasq3!(i0::S, n0::AbstractArray{S}, z::AbstractVector{T},
                pp::AbstractArray{S}, dmin::AbstractArray{T},
                sigma::AbstractArray{T}, desig::AbstractArray{T},
                qmax::T, nfail::AbstractArray{S}, iter::AbstractArray{S},
                ndiv::AbstractArray{S}, ieee::Bool, ttype::AbstractArray{S},
                dmin1::AbstractArray{T}, dmin2::AbstractArray{T}, dn::AbstractArray{T},
                dn1::AbstractArray{T}, dn2::AbstractArray{T}, g::AbstractArray{T},
                tau::AbstractArray{T}) where {T <:AbstractFloat, S <:Integer}
    n0in = n0[]
    prec = nothing
    cbias = T(1.5)
    qurtr = T(0.25)

    if T == Float32
        prec = ccall(
                    (@blasfunc(slamch_), libblastrampoline),
                    Float32,
                    (Ref{UInt8},),
                    Ref{UInt8}('P')  
                )
    elseif T == Float64
        prec = ccall(
                    (@blasfunc(dlamch_), libblastrampoline),
                    Float64,
                    (Ref{UInt8},),
                    Ref{UInt8}('P')  
                )

    else
        prec = precision(T)
    end

    tol = prec*T(100)
    tol2 = tol^2

@label ten

    if n0[] < i0
        return
    end
    if n0[] == i0
        @goto twenty
    end
    nn = 4*n0[] + pp[]

    if n0[] == (i0 + 1)
        @goto fourty
    end
    if z[nn - 5] > tol2*(sigma[] + z[nn - 3]) && z[nn - 2*pp[] - 4] > tol2*z[nn - 7]
        @goto thirty
    end

@label twenty

    z[4*n0 - 3] = z[4*n0 + pp[] - 3] + sigma[]
    n0 .-= 1
    @goto ten

@label thirty

    if z[nn - 9] > tol2 * sigma[] && z[nn - 2*pp[] - 8] > tol2 * z[nn - 11]
        @goto fifty
    end

@label fourty

    if z[nn - 3] > z[nn - 7]
        s = z[nn - 3]
        zz[nn - 3] = z[nn - 7]
        zz[nn - 7] = s
    end

    t = T(0.5)*((z[nn - 7] - z[nn - 3]) + z[nn - 5])

    if z[nn - 5] > z[nn - 3]*tool2 && t != 0
        s = z[nn - 3]*(z[nn - 5] / t)
        if s <= t
            s = z[nn - 3]*(z[nn - 5] / (t*(one(T) + sqrt(one(T) + s/t))))
        else
            s = z[nn - 3]*(z[nn - 5] / ((t + sqrt(t)*sqrt(t + s))))
        end
        t = z[nn - 7] + (s + z[nn - 5])
        z[nn - 3] *= (z[nn - 7]/t)
        z[nn - 7] = t
    end

    z[4*n0[] - 7] = z[nn - 7] + sigma[]
    z[4*n0[] - 3] = z[nn - 3] + sigma[]
    n0 .-= 2
    @goto ten

@label fifty

    if pp[] == 2
        pp .= 0
    end

    if dmin[] <= 0 || n0[] < n0in
        if cbias*z[4*i0 + pp[] - 3] < z[4*n0[] + pp[] - 3]
            ipn4 = 4*(i0 + n0[])

            for j4 in 4*i0:4:2*(i0 + n0[] - 1)
                temp = z[j4 - 3]
                z[j4 - 3] = z[ipn4 - j4 - 3]
                z[ipn4 - j4 - 3] = temp

                temp = z[j4 - 2]
                z[j4 - 2] = z[ipn4 - j4 - 2]
                z[ipn4 - j4 - 2] = temp

                temp = z[j4 - 1]
                z[j4 - 1] = z[ipn4 - j4 - 5]
                z[ipn4 - j4 - 5] = temp

                temp = z[j4 - 1]
                z[j4 - 1] = z[ipn4 - j4 - 5]
                z[ipn4 - j4 - 5] = temp

                temp = z[j4]
                z[j4] = z[ipn4 - j4 - 4]
                z[ipn4 - j4 - 4] = temp
            end

            if n0[] - i0 <= 4
                z[4*n0[] + pp[] - 1] = z[4*i0 + pp[] - 1]
                z[4*n0[] - pp[]] = z[4*i0 - pp[]]
            end
            dmin2 .= min(dmin2[], z[4*n0[] + pp[] - 1])
            z[4*n0[] + pp[] - 1] = min(z[4*n0[] + pp[] - 1],
                                       z[4*i0 + pp[] - 1],
                                       z[4*i0 + pp[] + 3])
            z[4*n0[] - pp[]] = min(z[4*n0[] - pp[]],
                                       z[4*i0 - pp[]],
                                       z[4*i0 - pp[] + 4])
            qmax = max(qmax, z[4*i0 + pp[] - 3], z[4*i0 + pp[] + 1])
            dmin .= -zero(T)
        end
    end

    lasq4!(i0, n0[], z, pp[], n0in, dmin[], dmin1[], dmin2[], dn[], dn1[],
            dn2[], tau, ttype, g)
    
@label seventy

    lasq5!(i0, n0[], z, pp[], tau[], sigma[], dmin, dmin1, dmin2, dn, dn1,
            dn2, ieee, prec)

    ndiv .+= (n0[] - i0 + 2)

    iter .+= 1

    if dmin[] >= 0 && dmin1[] >= 0
        @goto ninety
    
    elseif (dmin[] < 0 && dmin1[] > 0 && z[4*(n0[] - 1) - pp[]] < tol * (sigma[] + dn1[])
        && abs(dn[]) < tol*sigma[])
        z[4*(n0[] - 1) - pp[] + 2] = zero(T)
        dmin .= zero(T)
        @goto ninety
    elseif dmin[] < 0
        nfail .+= 1

        if ttype[] < -22
            #failed twice. Play it safe
            tau .= zero(T)
        
        elseif dmin1[] > 0
            #late failure. Gives excellent shift
            tau .= (tau[] + dmin[])*(one(T) - 2*one(T)*prec)
            ttype .-= 11
        else
            tau *= qurtr
            ttype .-= 12
        end
        @goto seventy
    elseif isnan(dmin[])
        if tau[] == 0
            @goto eighty
        else
            tau .= zero(T)
        end
    else
        #Possible underflow. Play it safe
        @goto eighty
    end

@label eighty

    lasq6!(i0, n0[], z, pp[], dmin, dmin1, dmin2, dn , dn1, dn2)
    ndiv .+= (n0[] - i0 + 2)
    iter .+= 1
    tau .= zero(T) 

@label ninety

    if tau[] < sigma[]
        desig .+=  tau[]
        t = sigma[] + desig[]
        desig .-= (t - sigma[])
    else
        t = sigma[] + tau[]
        desig .+= sigma[] - (t - tau[]) 
    end
    sigma .= t

end
