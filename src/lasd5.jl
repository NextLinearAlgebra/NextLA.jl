using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

function lasd5!(
    i::S,
    d::AbstractVector{T},
    z::AbstractVector{T},
    delta::AbstractVector{T},
    rho::T,
    dsigma::AbstractVector{T},
    work::AbstractVector{T}
) where {S<:Integer, T<:AbstractFloat}

    # println("slasd5 length d: $(length(d))")
    del = d[2] - d[1]
    delsq = del*(d[2]+d[1])
    b = 0
    c = 0
    if i == 1
        w = one(T) + T(4)*rho*(z[2]^2/(d[1]+T(3)*d[2])
        - z[1]^2 /(T(3)*d[1]+d[2]))/del 

        if w > 0
            b = delsq + rho*(z[1]^2 + z[2]^2)
            c = rho*z[1]^2*delsq


            #The following tay is dsigma^2 - d[1]^2

            tau = T(2)*c/(b + sqrt(abs(b^2-T(4)*c)))

            # the following tau is dsigma - d[1]
             tau /= (d[1]+sqrt(d[1]^2+tau))
             dsigma .= d[1] + tau
             delta[1] = -tau
             delta[2] = del - tau
             work[1] = T(2)*d[1] + tau
             work[2] = (d[1]+tau) + d[2]
        else
            b = -delsq + rho*(z[1]^2 + z[2]^2)
            c = rho*z[2]^2*delsq

            if b > 0
                tau = -T(2)*c / (b + sqrt(b^2+T(4)*c))
            else
                tau = (b - sqrt(b^2+T(4)*c)) / T(2)
            end

            tau /= (d[2] + sqrt(abs(d[2]^2 + tau)))
            dsigma .= d[2] + tau
            delta[1] = -(del + tau)
            delta[2] = -tau
            work[1] = d[1] + tau + d[2]
            work[2] = T(2)*d[2] + tau

        end
    else
        b = -delsq + rho*(z[1]^2 + z[2]^2)
        c = rho * z[2]^2*delsq

        if b > 0
            tau = (b + sqrt(b^2+T(4)*c))/T(2)
        else
            tau = T(2)*c / (-b + sqrt(b^2 + T(4)*c))
        end

        tau /= (d[2] + sqrt(d[2]^2 + tau))
        dsigma .= d[2] + tau
        delta[1] = -(del+tau)
        delta[2] = -tau
        work[1] = d[1] + tau + d[2]
        work[2] = T(2) * d[2] + tau
    end
    # println("")
end
