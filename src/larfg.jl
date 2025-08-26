function larfg(n, alpha, x, incx, tau)
    one = oneunit(eltype(alpha))
    zero0 = zero(eltype(alpha)) 
    type = eltype(alpha)
    if n <= 0
        tau = zero0
        return alpha, tau
    end
    
    if n == 1
        xnorm = 0
    else
        xnorm = norm(x,2)
    end

    alphr = real(alpha)
    alphi = imag(alpha)

    if xnorm == 0 && alphi == 0
        tau = zero0
    else
        beta = -copysign(sqrt(alphr^2 + alphi^2 + xnorm^2), alphr)
        safmin = lamch(eltype(alphr), 'S') / lamch(eltype(alphr), 'E')
        rsafmn = one / safmin
        knt = 0

        if abs(beta) < safmin
            #  xnorm, beta may be inaccurate, scale x and recompute
            
            while true
                knt += 1
                x .*= rsafmn
                beta *= rsafmn
                alphr *= rsafmn
                alphi *= rsafmn
                alpha *= rsafmn

                if abs(beta) < safmin
                    break
                end
            end                

            #recompute 
            xnorm = norm(x)
            if type <: Complex
                alpha = alphr + im * alphi
            end
            beta = -copysign(sqrt(alphr^2 + alphi^2 + xnorm^2), alphr)
        end
        if type <: Complex
            tau = ( beta-alphr ) / beta - im * alphi / beta 
        else
            tau = ( beta - alphr ) / beta
        end
        x .*= (one / (alpha-beta))
        
        for j in 1:knt
            beta *= safmin
        end

        alpha = beta
    end

    return alpha, tau
end
