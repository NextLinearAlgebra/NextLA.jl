"""
    lartg(f::R, g::S) where {R,S}

Generate a plane rotation (Givens rotation) such that:

    [ c   s ]' * [ f ] = [ r ]
    [-s   c ]    [ g ]   [ 0 ]

where `c` is real and `s` may be real or complex. The scalar `r` has the
phase of `f` when `f` is nonzero.

# Arguments
- `f`: Scalar element (real or complex)
- `g`: Scalar element (real or complex)

# Returns
- `c`: Real cosine of the rotation
- `s`: Sine of the rotation (real or complex)
- `r`: Resulting scalar after applying the rotation

# Algorithm
The implementation follows LAPACK's scaling strategy to avoid
over/underflow when computing norms. For complex inputs, `c` is always
real and `s` carries the phase so that `r` aligns with `f`.

Special cases:
- If `g == 0`, then `c = 1`, `s = 0`, `r = f`
- If `f == 0`, then `c = 0`, `s = conj(g)/abs(g)`, `r = abs(g)`

# Note
This is a low-level LAPACK-style computational routine. Input validation
should be performed by higher-level interfaces.
"""
function lartg(f::R, g::S) where {R,S}
    T = promote_type(R, S)
    RT = real(T)

    f = convert(T, f)
    g = convert(T, g)

    sfmin = lamch(RT, 'S')
    sfmax = one(RT) / sfmin
    rtmin = sqrt(sfmin)
    rtmax = one(RT) / rtmin

    if iszero(g)
        return one(RT), zero(T), f
    end

    if iszero(f)
        gmax = max(abs(real(g)), abs(imag(g)))
        if rtmin < gmax < rtmax
            c = zero(RT)
            s = g / abs(g)
            r = convert(T, abs(g))
            return c, s, r
        else
            u = min(sfmax, max(sfmin, gmax))
            gs = g / u
            c = zero(RT)
            s = gs / abs(gs)
            r = convert(T, abs(gs) * u)
            return c, s, r
        end
    end

    fmax = max(abs(real(f)), abs(imag(f)))
    gmax = max(abs(real(g)), abs(imag(g)))

    if (rtmin < fmax < rtmax) && (rtmin < gmax < rtmax)
        # unscaled algorithm
        f2 = abs2(f)
        g2 = abs2(g)
        h2 = f2 + g2

        d = (f2 > rtmin && h2 < rtmax) ? sqrt(f2 * h2) : sqrt(f2) * sqrt(h2)
        p = inv(d)

        c = convert(RT, f2 * p)
        s = conj(g) * (f * p)
        r = f * (h2 * p)
        return c, s, r
    else
        # scaled algorithm
        u = min(sfmax, max(sfmin, fmax, gmax))
        gs = g / u
        g2 = abs2(gs)

        if fmax / u < rtmin
            # different scalings for f and g
            v = min(sfmax, max(sfmin, fmax))
            w = v / u
            fs = f / v
            f2 = abs2(fs)
            h2 = f2 * (w * w) + g2
        else
            # same scaling for f and g
            w = one(RT)
            fs = f / u
            f2 = abs2(fs)
            h2 = f2 + g2
        end

        d = (f2 > rtmin && h2 < rtmax) ? sqrt(f2 * h2) : sqrt(f2) * sqrt(h2)
        p = inv(d)

        c = convert(RT, (f2 * p) * w)
        s = conj(gs) * (fs * p)
        r = (fs * (h2 * p)) * u
        return c, s, r
    end
end
