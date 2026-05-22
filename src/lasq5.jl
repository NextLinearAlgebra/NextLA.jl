using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

#=
Purpose:
!>
!> SLASQ5 computes one dqds transform in ping-pong form, one
!> version for IEEE machines another for non IEEE machines.
!> 
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
[in]	TAU	
!>          TAU is REAL
!>        This is the shift.
!> 
[in]	SIGMA	
!>          SIGMA is REAL
!>        This is the accumulated shift up to this step.
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
[in]	IEEE	
!>          IEEE is LOGICAL
!>        Flag for IEEE or non IEEE arithmetic.
!> 
[in]	EPS	
!>         EPS is REAL
!>        This is the value of epsilon used.
!> 
=#
function lasq5!(i0::S, n0::S, z::AbstractVector{T}, pp::S,
                tau::T, sigma::T, dmin::AbstractVector{T},
                dmin1::AbstractVector{T},
                dmin2::AbstractVector{T}, dn::AbstractVector{T},
                dnm1::AbstractVector{T}, dnm2::AbstractVector{T},
                ieee::Bool, eps::T) where {T <:AbstractFloat, S <:Integer}

    half = T(0.5)
    if n0 - i0 -1 <= 0
        return
    end

    dthresh = eps*(sigma + tau)
    if tau < dthresh * half
        tau = zero(T)
    end
    emin = nothing
    if tau != 0
        j4 =  4*i0 + pp - 3
        emin = z[j4 + 4]
        d = z[j4] - tau
        dmin .= d
        dmin1 .= -z[j4]

        if ieee 
            if pp == 0
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 2] = d + z[j4-1]
                    temp = z[j4 + 1] / z[j4 - 2]
                    d = d*temp - tau
                    dmin .= min(dmin[], d)
                    z[j4] = z[j4-1]*temp
                    emin = min(z[j4], emin)
                end
            else 
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 3] = d + z[j4]
                    temp = z[j4 + 2] / z[j4 - 3]
                    d = d*temp - tau
                    dmin .= min(dmin[], d)
                    z[j4 - 1] = z[j4]*temp
                    emin = min(z[j4 - 1], emin)
                end
            end

            dnm2 .= d
            dmin2 .= dmin
            j4 = 4*(n0 - 2) - pp
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm2[] + z[j4p2]
            z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
            dnm1 .= z[j4p2 + 2]*(dnm2[] / z[j4 - 2]) - tau
            dmin .= min(dmin[], dnm1[])

            dmin1 .= dmin[]
            j4 = j4 + 4
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm1[] + z[j4p2]
            z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
            dn .= z[j4p2 + 2]*(dnm1[] / z[j4 - 2]) - tau
            dmin .= min(dmin[], dn[])
        else
            if pp == 0
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 2] = d + z[j4 - 1]
                    if d < 0
                        return
                    else
                        z[j4] = z[j4 + 1]*(z[j4 - 1]/z[j4 - 2])
                        d = z[j4 + 1]*(d/z[j4 - 2]) - tau
                    end
                    dmin .= min(dmin[], d)
                    emin = min(emin, z[j4])
                end
            else
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 3] = d + z[j4]
                    if d < 0
                        return
                    else
                        z[j4 - 1] = z[j4 + 2]*(z[j4]/z[j4 - 3])
                        d = z[j4 + 2]*(d/z[j4 - 3]) - tau
                    end
                    dmin .= min(dmin[], d)
                    emin = min(emin, z[j4 - 1])
                end
            end

            dnm2 .= d
            dmin2 .= dmin[]
            j4 = 4*(n0 - 2) - pp
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm2[] + z[j4p2]

            if dnm2[] < 0
                return
            else
                z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
                dnm1 .= z[j4p2 + 2]*(dnm2[] / z[j4 - 2]) - tau
            end
            dmin .= min(dmin[], dnm1[])

            dmin1 .= dmin
            j4 = j4 + 4
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm1[] + z[j4p2]
            if dnm1[] < 0
                return
            else
                z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
                dn .= z[j4p2 + 2]*(dnm1[] / z[j4 - 2]) - tau
            end
            dmin .= min(dmin[], dn[])
        end
    else
        j4 =  4*i0 + pp - 3
        emin = z[j4 + 4]
        d = z[j4] - tau
        dmin .= d
        dmin1 .= -z[j4]

        if ieee 
            if pp == 0
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 2] = d + z[j4-1]
                    temp = z[j4 + 1] / z[j4 - 2]
                    d = d*temp - tau
                    if d < dthresh
                        d = zero(T)
                    end
                    dmin .= min(dmin[], d)
                    z[j4] = z[j4-1]*temp
                    emin = min(z[j4], emin)
                end
            else 
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 3] = d + z[j4]
                    temp = z[j4 + 2] / z[j4 - 3]
                    d = d*temp - tau
                    if d < dthresh
                        d = zero(T)
                    end
                    dmin .= min(dmin[], d)
                    z[j4 - 1] = z[j4]*temp
                    emin = min(z[j4 - 1], emin)
                end
            end

            dnm2 .= d
            dmin2 .= dmin[]
            j4 = 4*(n0 - 2) - pp
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm2[] + z[j4p2]
            z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
            dnm1 .= z[j4p2 + 2]*(dnm2[] / z[j4 - 2]) - tau
            dmin .= min(dmin[], dnm1[])

            dmin1 .= dmin[]
            j4 = j4 + 4
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm1[] + z[j4p2]
            z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
            dn .= z[j4p2 + 2]*(dnm1[] / z[j4 - 2]) - tau
            dmin .= min(dmin[], dn[])
        else
            if pp == 0
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 2] = d + z[j4 - 1]
                    if d < 0
                        return
                    else
                        z[j4] = z[j4 + 1]*(z[j4 - 1]/z[j4 - 2])
                        d = z[j4 + 1]*(d/z[j4 - 2]) - tau
                    end
                    if d < dthresh
                        d = zero(T)
                    end
                    dmin .= min(dmin[], d)
                    emin = min(emin, z[j4])
                end
            else
                for j4 in 4*i0:4:4*(n0-3)
                    z[j4 - 3] = d + z[j4]
                    if d < 0
                        return
                    else
                        z[j4 - 1] = z[j4 + 2]*(z[j4]/z[j4 - 3])
                        d = z[j4 + 2]*(d/z[j4 - 3]) - tau
                    end
                    if d < dthresh
                        d = zero(T)
                    end
                    dmin .= min(dmin[], d)
                    emin = min(emin, z[j4 - 1])
                end
            end

            dnm2 .= d
            dmin2 .= dmin[]
            j4 = 4*(n0 - 2) - pp
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm2[] + z[j4p2]

            if dnm2[] < 0
                return
            else
                z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
                dnm1 .= z[j4p2 + 2]*(dnm2[] / z[j4 - 2]) - tau
            end
            dmin .= min(dmin[], dnm1[])

            dmin1 .= dmin[]
            j4 = j4 + 4
            j4p2 = j4 + 2*pp - 1
            z[j4 - 2] = dnm1[] + z[j4p2]
            if dnm1[] < 0
                return
            else
                z[j4] = z[j4p2 + 2]*(z[j4p2]/z[j4 - 2])
                dn .= z[j4p2 + 2]*(dnm1[] / z[j4 - 2]) - tau
            end
            dmin .= min(dmin[], dn[])
        end
    end
    z[j4 + 2] = dn[]
    z[4*n0 - pp] = emin
end
