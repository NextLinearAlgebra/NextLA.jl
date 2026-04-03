using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

#=
Parameters
[in]	I0	
!>          I0 is INTEGER
!>        First index.
!> 
[in]	N0	
!>          N0 is INTEGER
!>        Last index.
!> 
[in]	Z	
!>          Z is REAL array, dimension ( 4*N )
!>        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
!>        an extra argument.
!> 
[in]	PP	
!>          PP is INTEGER
!>        PP=0 for ping, PP=1 for pong.
!> 
[out]	DMIN	
!>          DMIN is REAL
!>        Minimum value of d.
!> 
[out]	DMIN1	
!>          DMIN1 is REAL
!>        Minimum value of d, excluding D( N0 ).
!> 
[out]	DMIN2	
!>          DMIN2 is REAL
!>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
!> 
[out]	DN	
!>          DN is REAL
!>        d(N0), the last value of d.
!> 
[out]	DNM1	
!>          DNM1 is REAL
!>        d(N0-1).
!> 
[out]	DNM2	
!>          DNM2 is REAL
!>        d(N0-2).
!> 
=#
function lasq6!(i0::S, n0::S, z::AbstractVector{T}, pp::S,
                dmin::AbstractVector{T}, dmin1::AbstractVector{T},
                dmin2::AbstractVector{T}, dn::AbstractVector{T},
                dnm1::AbstractVector{T}, dnm2::AbstractVector{T}) where {T <:AbstractFloat, S <:Integer}
    
    if n0 - i0 - 1 <= 0
        return
    end

    safmin = 0

    if T == Float32
        safmin = ccall(
                        (@blasfunc(slamch_), libblastrampoline),
                        Float32,
                        (Ref{UInt8},),
                        UInt8('S')
                        )
    elseif T ==  Float64
        safmin = ccall(
                        (@blasfunc(dlamch_), libblastrampoline),
                        Float64,
                        (Ref{UInt8},),
                        UInt8('S')
                        )

    else
        safmin = floatmin(T)
    end

    j4 = 4*i0 + pp - 3
    emin = z[j4+4]
    d = z[j4]
    dmin .= d

    if pp == 0
        for i in 4*i0:4:4*(n0 -3)
            z[j4 - 2] = d + z[j4 - 1]
            if z[j4 - 2] == 0
                z[j4] = zero(T)
                d == z[j4 + 1]
                dmin .= d
                emin = zero(T)
            elseif (safmin*z[j4 + 1] < z[j4 - 2] &&
                    safmin*z[j4 - 2] < z[j4 + 1])
                    temp = z[j4 + 1] / z[j4 - 2]
                    z[j4] = z[j4-1] * temp
                    d *= temp
            else
                z[j4] = z[j4 + 1] * (z[j4 - 1] / z[j4 - 2])
                d = z[j4 + 1]*(d/z[j4 - 2])
            end
            dmin .= min(dmin[], d)
            emin = min(emin, z[j4])
        end
    else
        for i in 4*i0:4:4*(n0 -3)
            z[j4 - 3] = d + z[j4 - 1]
            if z[j4 - 3] == 0
                z[j4 - 1] = zero(T)
                d == z[j4 + 2]
                dmin .= d
                emin = zero(T)
            elseif (safmin*z[j4 + 2] < z[j4 - 3] &&
                    safmin*z[j4 - 3] < z[j4 + 2])
                    temp = z[j4 + 2] / z[j4 - 3]
                    z[j4 - 1] = z[j4] * temp
                    d *= temp
            else
                z[j4 - 1] = z[j4 + 2] * (z[j4] / z[j4 - 3])
                d = z[j4 + 2]*(d/z[j4 - 3])
            end
            dmin .= min(dmin[], d)
            emin = min(emin, z[j4 - 1])
        end
    end

    #Unroll last two steps

    dnm2 .= d
    dmin2 .= dmin[]
    j4 = 4*(n0 - 2) - pp
    j4p2 = j4 + 2*pp - 1
    z[j4 - 2] = dnm2[] + z[j4p2]

    if z[j4 - 2] == 0
        z[j4] = zero(T)
        dnm1 .= z[j4p2 + 2]
        dmin .= dnm1[]
        emin = zero(T)
    elseif safmin*z[j4p2 + 2] < z[j4-2] && safmin * z[j4 - 2] < z[j4p2 + 2]
        temp = z[j4p2 + 2] / z[j4 - 2]
        z[j4] = z[j4p2] * temp
        dnm1 .= dnm2[]*temp
    else
        z[j4] = z[j4p2+2]*(z[j4p2]/z[j4-2])
        dnm1 .= z[j4p2 + 2]*(dnm2[]/z[j4 - 2])
    end
    dmin .= min(dmin[], dnm1[])

    dmin1 .= dmin[]

    j4 = j4 + 4
    j4p2 = j4 + d*pp - 1
    z[j4 - 2] = dnm1[] + z[j4p2]
    if z[j4 - 2] == 0
        z[j4] = zero(T)
        dn .= z[j4p2 + 2]
        dmin .= dn[]
        emin = zero(T)
    else if safmin * z[j4p2 + 2] < z[j4 - 2] && safmin * z[j4 - 2] < z[j4p2 + 2]
        temp = z[j4p2 + 2] / z[j4 - 2]
        z[j4] = z[j4p2] * temp
        dn .= temp * dnm1
    else
        z[j4] = z[j4p2 + 2] * (z[j4p2]/z[j4 - 2])
        dn .= z[j4p2 + 2]*(dnm1[]/z[j4 - 2])
    end

    dmin .= min(dmin[], dn[])
    z[j4 + 2] = dn[]
    z[4*n0 - pp] = emin
end
