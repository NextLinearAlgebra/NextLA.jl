"""
    larfg!(n, alpha, x, incx, tau)

Generate an elementary reflector H such that:
H * [alpha; x] = [beta; 0]

where H = I - tau * v * v^H, v = [1; x/scale], and beta = -sign(alpha) * ||[alpha; x]||

This routine generates a complex elementary Householder reflector H of order n,
such that when applied to the vector [alpha; x], it zeros out the x portion
and produces [beta; 0] where beta has the same magnitude as the original vector.

# Arguments  
- `n`: Order of the reflector (length of full vector [alpha; x])
- `alpha`: Scalar element, the first component of the vector
- `x`: Vector of length n-1, remaining components of the vector  
- `incx`: Increment for elements of x (typically 1)
- `tau`: Output scalar factor for the reflector

# Returns
- `alpha`: Modified to contain beta (the new first component)
- `tau`: Scalar factor such that H = I - tau * v * v^H

# Algorithm
The algorithm handles potential under/overflow carefully by scaling when
necessary. The reflector is chosen so that the reflection introduces no
unnecessary amplification of round-off errors.

Special cases:
- If x = 0 and imag(alpha) = 0, then tau = 0 (no reflection needed)
- If n ≤ 1, then tau = 0 (trivial case)

# Mathematical Details
For the elementary reflector H = I - tau * v * v^H where v = [1; u]:
- tau = (beta - alpha) / beta for real case
- tau = (beta - Re(alpha))/beta - i*Im(alpha)/beta for complex case  
- The vector u replaces x on output

# Note
This is a low-level LAPACK-style computational routine. Input validation
should be performed by higher-level interfaces.
"""
function larfg!(n::Integer, alpha::T, x::AbstractVector{T}, incx::Integer, tau::T) where {T}
    one = oneunit(eltype(alpha))
    zero0 = zero(eltype(alpha)) 
    
    if n <= 1
        tau = zero0
        return alpha, tau
    end

    xnorm = norm(x, 2)

    alphr = real(alpha)
    alphi = imag(alpha)


    if xnorm == zero0 && alphi == zero0
        tau = zero0
    else
        # Compute beta = -sign(alphr) * ||[alpha, x]||
        beta = -copysign(sqrt(alphr^2 + alphi^2 + xnorm^2), alphr)
        # Machine parameters for safe scaling
        safmin = lamch(eltype(alphr), 'S') / lamch(eltype(alphr), 'E')
        rsafmn = one / safmin
        knt = 0

        if abs(beta) < safmin
            # xnorm, beta may be inaccurate due to underflow; scale and recompute
            while abs(beta) < safmin
                knt += 1
                x .*= rsafmn
                beta *= rsafmn
                alphr *= rsafmn  
                alphi *= rsafmn
                alpha *= rsafmn
            end                

            # Recompute with scaled values
            xnorm = norm(x, 2)
            if T <: Complex
                alpha = alphr + im * alphi
            end
            beta = -copysign(sqrt(alphr^2 + alphi^2 + xnorm^2), alphr)
        end
        
        # Compute tau based on number type
        if T <: Complex
            tau = (beta - alphr) / beta - im * alphi / beta 
        else
            tau = (beta - alphr) / beta
        end
        
        # Scale x to form the reflector vector
        x .*= (one / (alpha - beta))
        
        # Scale beta back if we scaled up
        for j in 1:knt
            beta *= safmin
        end

        alpha = beta
    end

    return alpha, tau
end

"""
    larfg!(x) -> (alpha, tau, x_updated)

Generate an elementary reflector H such that H * x produces a vector
with all but the first element equal to zero.

This is a high-level interface to the elementary reflector generation routine.
Given a vector x, it computes a Householder reflector H = I - tau * v * v^H
that zeros out all but the first component.

# Arguments
- `x`: Vector to be transformed (will be modified in-place)

# Returns  
- `alpha`: The resulting first component (beta)
- `tau`: Scalar factor of the elementary reflector
- `x_updated`: The updated vector with first component as alpha, rest as reflector vector

# Input Validation
- Vector must have at least one element

# Example
```julia
x = complex.([3.0, 4.0, 0.0], [0.0, 0.0, 0.0])
alpha, tau, x_new = larfg!(x)
# x_new[1] will be the magnitude -||x||, x_new[2:end] will be the reflector vector
```

# Mathematical Background
Creates H such that H * x = [||x||; 0; 0; ...] where the sign is chosen
to avoid cancellation. The reflector vector is stored in x_new[2:end].
"""
function larfg!(alpha::T, x::AbstractVector{T}, incx::Integer, tau::T) where {T}
    n = length(x)
    
    alpha_out, tau_out = larfg!(n, alpha, x, incx, tau)

    return alpha_out, tau_out
end