using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

#=
Purpose:
!>
!> SLASQ4 computes an approximation TAU to the smallest eigenvalue
!> using values of d from the previous transform.
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
!>          Z is REAL array, dimension ( 4*N0 )
!>        Z holds the qd array.
!> 
[in]	PP	
!>          PP is INTEGER
!>        PP=0 for ping, PP=1 for pong.
!> 
[in]	N0IN	
!>          N0IN is INTEGER
!>        The value of N0 at start of EIGTEST.
!> 
[in]	DMIN	
!>          DMIN is REAL
!>        Minimum value of d.
!> 
[in]	DMIN1	
!>          DMIN1 is REAL
!>        Minimum value of d, excluding D( N0 ).
!> 
[in]	DMIN2	
!>          DMIN2 is REAL
!>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
!> 
[in]	DN	
!>          DN is REAL
!>        d(N)
!> 
[in]	DN1	
!>          DN1 is REAL
!>        d(N-1)
!> 
[in]	DN2	
!>          DN2 is REAL
!>        d(N-2)
!> 
[out]	TAU	
!>          TAU is REAL
!>        This is the shift.
!> 
[out]	TTYPE	
!>          TTYPE is INTEGER
!>        Shift type.
!> 
[in,out]	G	
!>          G is REAL
!>        G is passed as an argument in order to save its value between
!>        calls to SLASQ4.
!> 
=#

function lasq4!(i0::S, n0::S, z::AbstractVector{T}, pp::S,
                n0in::S, dmin::T,
                dmin1::T, dmin2::T,
                dn::T, dn1::T,
                dn2::T, tau::AbstractVector{T},
                ttype::AbstractVector{S}, g::AbstractVector{T}) where {T <: AbstractFloat, S <: Integer}
    
    qurtr = T(0.25)
    half = T(0.5)
    third = T(1)/T(3)
    hundrd = T(100)
    cnst1 = T(0.563)    # 9/16
    cnst2 = T(1.01)    # 1/16
    cnst3 = T(1.05)   # 5/32
    two = T(2)
    
    if dmin <= 0
        tau .= -dmin
        ttype .= -1
        return
    end
    
    nn = 4*n0 + pp
    
    if n0in == n0
        # No eigenvalues deflated
        
        if dmin == dn || dmin == dn1
            
            b1 = sqrt(z[nn-3]) * sqrt(z[nn-5])
            b2 = sqrt(z[nn-7]) * sqrt(z[nn-9])
            a2 = z[nn-7] + z[nn-5]
            
            # Cases 2 and 3
            if dmin == dn && dmin1 == dn1
                gap2 = dmin2 - a2 - dmin2*qurtr
                if gap2 > 0 && gap2 > b2
                    gap1 = a2 - dn - (b2/gap2)*b2
                else
                    gap1 = a2 - dn - (b1 + b2)
                end
                
                if gap1 > 0 && gap1 > b1
                    s = max(dn - (b1/gap1)*b1, half*dmin)
                    ttype .= -2
                else
                    s = zero(T)
                    if dn > b1
                        s = dn - b1
                    end
                    if a2 > (b1 + b2)
                        s = min(s, a2 - (b1 + b2))
                    end
                    s = max(s, third*dmin)
                    ttype .= -3
                end
            else
                # Case 4
                ttype .= -4
                s = qurtr*dmin
                
                if dmin == dn
                    gam = dn
                    a2 = zero(T)
                    if z[nn-5] > z[nn-7]
                        return
                    end
                    b2 = z[nn-5] / z[nn-7]
                    np = nn - 9
                else
                    np = nn - 2*pp
                    gam = dn1
                    if z[np-4] > z[np-2]
                        return
                    end
                    a2 = z[np-4] / z[np-2]
                    if z[nn-9] > z[nn-11]
                        return
                    end
                    b2 = z[nn-9] / z[nn-11]
                    np = nn - 13
                end
                
                # Approximate contribution to norm squared from I < NN-1
                a2 = a2 + b2
                for i4 in np:-4:4*i0 - 1 + pp
                    if b2 == 0
                        break
                    end
                    b1 = b2
                    if z[i4] > z[i4-2]
                        return
                    end
                    b2 = b2 * (z[i4] / z[i4-2])
                    a2 = a2 + b2
                    if hundrd*max(b2, b1) < a2 || cnst1 < a2
                        break
                    end
                end
                
                a2 = cnst3*a2
                
                # Rayleigh quotient residual bound
                if a2 < cnst1
                    s = gam * (one(T) - sqrt(a2)) / (one(T) + a2)
                end
            end
            
        elseif dmin == dn2
            # Case 5
            ttype .= -5
            s = qurtr*dmin
            
            # Compute contribution to norm squared from I > NN-2
            np = nn - 2*pp
            b1 = z[np-2]
            b2 = z[np-6]
            gam = dn2
            
            if z[np-8] > b2 || z[np-4] > b1
                return
            end
            
            a2 = (z[np-8] / b2) * (one(T) + z[np-4] / b1)
            
            # Approximate contribution to norm squared from I < NN-2
            if n0 - i0 > 2
                b2 = z[nn-13] / z[nn-15]
                a2 = a2 + b2
                
                for i4 in nn - 17:-4:4*i0 - 1 + pp
                    if b2 == 0
                        break
                    end
                    b1 = b2
                    if z[i4] > z[i4-2]
                        return
                    end
                    b2 = b2 * (z[i4] / z[i4-2])
                    a2 = a2 + b2
                    if hundrd*max(b2, b1) < a2 || cnst1 < a2
                        break
                    end
                end
                
                a2 = cnst3*a2
            end
            
            if a2 < cnst1
                s = gam * (one(T) - sqrt(a2)) / (one(T) + a2)
            end
        else
            # Case 6, no information to guide us
            if ttype[] == -6
                g .= g[] + third * (one(T) - g[])
            elseif ttype[] == -18
                g .= qurtr * third
            else
                g .= qurtr
            end
            s = g[] * dmin
            ttype .= -6
        end
        
    elseif n0in == n0 + 1
        # One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
        
        if dmin1 == dn1 && dmin2 == dn2
            # Cases 7 and 8
            ttype .= -7
            s = third * dmin1
            
            if z[nn-5] > z[nn-7]
                return
            end
            
            b1 = z[nn-5] / z[nn-7]
            b2 = b1
            
            if b2 != 0
                for i4 in 4*n0 - 9 + pp:-4:4*i0 - 1 + pp
                    a2 = b1
                    if z[i4] > z[i4-2]
                        return
                    end
                    b1 *= (z[i4] / z[i4-2])
                    b2 += b1
                    if hundrd * max(b1, a2) < b2
                        break
                    end
                end
            end
            
            b2 = sqrt(cnst3 * b2)
            a2 = dmin1 / (one(T) + b2^2)
            gap2 = half * dmin2 - a2
            
            if gap2 > 0 && gap2 > b2*a2
                s = max(s, a2 * (one(T) - cnst2*a2*(b2/gap2)*b2))
            else
                s = max(s, a2 * (one(T) - cnst2*b2))
                ttype .= -8
            end
        else
            # Case 9
            s = qurtr * dmin1
            if dmin1 == dn1
                s = half * dmin1
            end
            ttype .= -9
        end
        
    elseif n0in == n0 + 2
        # Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
        # Cases 10 and 11
        
        if dmin2 == dn2 && two*z[nn-5] < z[nn-7]
            ttype .= -10
            s = third * dmin2
            
            if z[nn-5] > z[nn-7]
                return
            end
            
            b1 = z[nn-5] / z[nn-7]
            b2 = b1
            
            if b2 != 0
                for i4 in 4*n0 - 9 + pp:-4:4*i0 - 1 + pp
                    if z[i4] > z[i4-2]
                        return
                    end
                    b1 = b1 * (z[i4] / z[i4-2])
                    b2 = b2 + b1
                    if hundrd * b1 < b2
                        break
                    end
                end
            end
            
            b2 = sqrt(cnst3 * b2)
            a2 = dmin2 / (one(T) + b2^2)
            gap2 = z[nn-7] + z[nn-9] - sqrt(z[nn-11])*sqrt(z[nn-9]) - a2
            
            if gap2 > 0 && gap2 > b2*a2
                s = max(s, a2 * (one(T) - cnst2*a2*(b2/gap2)*b2))
            else
                s = max(s, a2 * (one(T) - cnst2*b2))
            end
        else
            s = qurtr * dmin2
            ttype .= -11
        end
        
    elseif n0in > n0 + 2
        # Case 12, more than two eigenvalues deflated. No information.
        s = zero(T)
        ttype .= -12
    end
    
    tau .= s
    
end
