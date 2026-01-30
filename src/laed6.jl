using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

function laed6!(
    kniter::S,
    orgati::Bool,
    rho::T, 
    d::AbstractVector{T},
    z::AbstractVector{T},
    finit::T,
    tau::AbstractVector{T},
    info::AbstractVector{S}
) where {S<: Integer, T<:AbstractFloat}

    zscale = zeros(T,3)
    dscale = zeros(T,3)
    lbd = zero(T)
    ubd = zero(T)
    maxit = 40
    sclfac = 0
    sclinv = 0
    info .= 0
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    a = 0
    b = 0
    c = 0

    if orgati
        lbd = d[2]
        ubd = d[3]
    else
        lbd = d[1]
        ubd = d[2]
    end

    if finit < 0
        lbd = zero(T)
    else
        ubd = zero(T)
    end

    niter = 1
    tau .= zero(T)

    if kniter == 2

        if orgati
            temp = (d[3] - d[2])/T(2)
            c = rho + z[1]/ ((d[1]-d[2]) - temp)
            a = c*(d[2]+d[3]) + z[2] + z[3]
            b = c*d[2]*d[3] + z[2]*d[3] + z[3]*d[2]
        else
            temp = (d[1] - d[2]) / T(2)
            c = rho + z[3] / ((d[3]- d[2]) - temp)
            a = c*(d[1] + d[2]) + z[1] + z[2]
            b = c*d[1]*d[2]+ z[1]*d[2] + z[2]*d[1]
        end

        temp = max(abs(a), abs(b), abs(c))
        a /= temp
        b /= temp
        c /= temp

        if c == 0
            tau .= b / a
        elseif a <= 0
            tau .= (a - sqrt(abs(a^2 - T(4)*b*c)))/ (T(2)*c)
        else
            tau .= T(2)*b / (a + sqrt(abs(a^2-T(4)*b*c)))
        end

        if tau[1] < lbd || tau[1] > ubd
            tau .= (lbd + ubd)/T(2)
        end
        if d[1] == tau[1]|| d[2] == tau[1] || d[3] == tau[1]
            tau .= zero(T)
        else
            temp = finit + (tau[1]*z[1]/(d[1]*(d[1]-tau[1])) + 
            tau[1]*z[2]/(d[2]*(d[2]-tau[1])) +
            tau[1]*z[3]/(d[3]*(d[3]-tau[1])))

            if temp <= 0
                lbd = tau[1]
            else
                ubd = tau[1]
            end

            if abs(finit) <= abs(temp)
                tau .= zero(T)
            end
        end
    end

    #Get machine paremeters for possible scaling to avoid overflow

    #=
        modified by Sven: parameters small1, sminv1, small2, sminv2,
        eps
    =#

    mach_eps = 0

    if T == Float32
        mach_eps = ccall(
                            (@blasfunc(slamch_), libblastrampoline),
                            Float32,
                            (Ref{UInt8},),
                            UInt8('E')
                        )
    elseif T ==  Float64
        mach_eps = ccall(
                            (@blasfunc(dlamch_), libblastrampoline),
                            Float64,
                            (Ref{UInt8},),
                            UInt8('E')
                        )
    else
        mach_eps = eps(T)
    end
    base = T(2) #The base for floating point numbers. Assumed to be two
    min_float = 0
    if T == Float32
        min_float = ccall(
                            (@blasfunc(slamch_), libblastrampoline),
                            Float32,
                            (Ref{UInt8},),
                            UInt8('S')
                        )
    elseif T ==  Float64
        min_float = ccall(
                            (@blasfunc(dlamch_), libblastrampoline),
                            Float64,
                            (Ref{UInt8},),
                            UInt8('S')
                        )
    else
        min_float = floatmin(T)
    end
    expo = Int(trunc(log(min_float)/log(base)/T(3)))
    small1 = base^(expo)
    sminv1 = one(T)/small1
    small2 = small1^2
    sminv2 = sminv1^2

    #=
        Determined if scaling of inputs necessary to avoid overflow
            when computing 1/temp^3
    =#

    if orgati
        temp = min(abs(d[2] - tau[1]), abs(d[3]-tau[1]))
    else
        temp = min(abs(d[1]-tau[1]), abs(d[2]-tau[1]))
    end

    scale = false
 
    if temp <= small1
        scale = true
        if temp <= small2
            sclfac = sminv2
            sclinv = small2
        else
            sclfac = sminv1
            sclinv = small1
        end
        
        for i in 1:3
            dscale[i] = d[i]*sclfac
            zscale[i] = z[i]*sclfac
        end
        
        tau .*= sclfac
        lbd *= sclfac
        ubd *= sclfac
        
    else
        for i in 1:3
            dscale[i] = d[i]
            zscale[i] = z[i]
        end
    end
    
    fc = zero(T)
    df = zero(T)
    ddf = zero(T)
    
    for i in 1:3
        temp = one(T)/ (dscale[i]-tau[1])
        temp1 = zscale[i]*temp
        temp2 = temp1*temp
        temp3 = temp2*temp
        fc += temp1/dscale[i]
        df += temp2
        ddf += temp3
    end
    
    f = finit + tau[1]*fc
    
    if abs(f) <= 0
        if scale
            tau .*= sclinv
        end
        return
    end
    
    if f <= 0
        lbd = tau[1]
    else
        ubd = tau[1]
    end
    
    #=
    Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
    scheme
    
    It is not hard to see that
    
    1) Iterations will go up monotonically
    if finit < 0
        
        2) Iterations will go down monotonically
        if finit > 0
            =#
            
    iter = niter + 1
            
    for niter in iter:maxit
        if orgati
            temp1 = dscale[2] -  tau[1]
            temp2 = dscale[3] -  tau[1]
        else
            temp1 = dscale[1] -  tau[1]
            temp2 = dscale[2] -  tau[1]
        end

        a = (temp1+temp2)*f - temp1*temp2*df
        b = temp1*temp2*f
        c = f - (temp1+temp2)*df + temp1*temp2*ddf

        temp = max(abs(a), abs(b), abs(c))
        a /= temp
        b /= temp
        c /= temp

        if c == 0
            eta = b/a
        elseif a <= 0
            eta = (a - sqrt(abs(a^2-T(4)*b*c)))/(T(2)*c)
        else
            eta = T(2)*b / (a + sqrt(abs(a^2-T(4)*b*c)))
        end
        
        if f*eta >= 0
            eta = -f/df
        end
        
        tau .+= eta
        
        if tau[1] < lbd || tau[1] > ubd
            tau .= (lbd + ubd)/T(2)
        end
        
        fc = zero(T)
        erretm = zero(T)
        df = zero(T)
        ddf = zero(T)

        for i in 1:3
            if (dscale[i] - tau[1]) != 0
                temp = one(T)/(dscale[i] - tau[1])
                temp1 = zscale[i]*temp
                temp2 = temp1*temp
                temp3 = temp2*temp
                temp4 = temp1/dscale[i]
                
                fc += temp4
                erretm += abs(temp4)
                df += temp2
                ddf += temp3
            else
                if scale
                    tau .*= sclinv
                end
                return
            end
        end

        f = finit + tau[1]*fc
        erretm = T(8)*(abs(finit)+abs(tau[1])*erretm) + abs(tau[1])*df
        if abs(f) <= T(4)*mach_eps*erretm || (ubd-lbd)<= T(4)*mach_eps*abs(tau[1])
            if scale
                tau .*= sclinv
            end
            return
        end
        
        if f <= 0
            lbd = tau[1]
        else
            ubd = tau[1]
        end

    end

    info .= 1
    if scale
        tau .*= sclinv
    end
    return

end
