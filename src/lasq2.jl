#=
SLASQ2 — eigenvalues from the qd array Z for the symmetric positive definite
tridiagonal factorization (LAPACK computational routine).

Reference: SLASQ2 in LAPACK Single precision
https://netlib.org/lapack/explore-html/d4/d4b/group__lasq2_gaab8c98d5e07394ba14d8e0404d184504.html
=#

"""
    lasq2!(n, z, info)

Translation of LAPACK **`SLASQ2`**. Uses `lasq3!` for dqds steps and `sort!(…; rev=true)`
in place of `SLASRT` with descending order (`'D'`).

`z` length must be at least `4*n` (`n ≥ 1`). Successful exit (`n ≥ 3`, general path)
stores statistics at indices `2*n+1` … `2*n+5`.

`info[1]` follows LAPACK conventions (0 OK; negative illegal element; positive failure).
"""
function lasq2!(n::S, z::AbstractVector{T}, info::AbstractArray{S}) where {T <: AbstractFloat, S <: Integer}
    info[1] = 0

    cbias = T(1.5)
    hundrd = T(100)
    half = T(0.5)
    onev = one(T)
    twov = onev + onev
    four = twov + twov

    epsv = lamch(T, 'E')
    safmin = lamch(T, 'S')
    tol = epsv * hundrd
    tol2 = tol * tol

    # Match LAPACK behavior:
    # - SLASQ2 hard-disables IEEE path for single precision
    # - DLASQ2 uses IEEE path detection (effectively true on modern platforms)
    ieee = (T == Float64)

    # --- Tiny orders (LAPACK scalar cases)
    if n < 0
        info[1] = -1
        return
    elseif n == 0
        return
    elseif n == 1
        if z[1] < zero(T)
            info[1] = -201
        end
        return
    elseif n == 2
        _lasq2_n2!(z, tol2, half, onev, info); return
    end

    # --- Checks and trace (LAPACK lines 217–267)
    z[2 * n] = zero(T)
    emin0 = z[2]
    qmx = zero(T)
    dsum = zero(T)
    esum = zero(T)

    for kk in 1:2:(2 * (n - 1))
        if z[kk] < zero(T)
            info[1] = -(200 + kk)
            return
        elseif z[kk + 1] < zero(T)
            info[1] = -(200 + kk + 1)
            return
        end
        dsum += z[kk]
        esum += z[kk + 1]
        qmx = max(qmx, z[kk])
        emin0 = min(emin0, z[kk + 1])
    end
    if z[2 * n - 1] < zero(T)
        info[1] = -(200 + 2 * n - 1)
        return
    end
    dsum += z[2 * n - 1]
    qmx = max(qmx, z[2 * n - 1])

    if iszero(esum)
        for kk in 2:n
            z[kk] = z[2 * kk - 1]
        end
        sort!(view(z, 1:n), alg=QuickSort, rev=true)
        z[2 * n - 1] = dsum
        return
    end

    trace_val = dsum + esum
    if iszero(trace_val)
        z[2 * n - 1] = zero(T)
        return
    end

    # --- Expand to locality layout Z = (q1,qq1,e1,ee1, …)  (lines 279–286)
    for kk in reverse(2:2:(2 * n))
        z[2 * kk] = zero(T)
        z[2 * kk - 1] = z[kk]
        z[2 * kk - 2] = zero(T)
        z[2 * kk - 3] = z[kk - 1]
    end

    i0 = 1
    n0 = n

    if cbias * z[4 * i0 - 3] < z[4 * n0 - 3]
        ipn4 = 4 * (i0 + n0)
        for i4 in 4 * i0:4:(2 * (i0 + n0 - 1))
            temp = z[i4 - 3]
            z[i4 - 3] = z[ipn4 - i4 - 3]
            z[ipn4 - i4 - 3] = temp
            temp = z[i4 - 1]
            z[i4 - 1] = z[ipn4 - i4 - 5]
            z[ipn4 - i4 - 5] = temp
        end
    end

    # Initial dqd / Li (lines 307–355)
    pp = 0
    for _ in 1:2
        d = z[4 * n0 + pp - 3]
        for i4 in (4 * (n0 - 1) + pp):-4:(4 * i0 + pp)
            if z[i4 - 1] <= tol2 * d
                z[i4 - 1] = -zero(T)
                d = z[i4 - 3]
            else
                d = z[i4 - 3] * (d / (d + z[i4 - 1]))
            end
        end

        emin0 = z[4 * i0 + pp + 1]
        d = z[4 * i0 + pp - 3]
        for i4 in (4 * i0 + pp):4:(4 * (n0 - 1) + pp)
            z[i4 - 2 * pp - 2] = d + z[i4 - 1]
            if z[i4 - 1] <= tol2 * d
                z[i4 - 1] = -zero(T)
                z[i4 - 2 * pp - 2] = d
                z[i4 - 2 * pp] = zero(T)
                d = z[i4 + 1]
            elseif safmin * z[i4 + 1] < z[i4 - 2 * pp - 2] &&
                   safmin * z[i4 - 2 * pp - 2] < z[i4 + 1]
                temp = z[i4 + 1] / z[i4 - 2 * pp - 2]
                z[i4 - 2 * pp] = z[i4 - 1] * temp
                d *= temp
            else
                z[i4 - 2 * pp] = z[i4 + 1] * (z[i4 - 1] / z[i4 - 2 * pp - 2])
                d = z[i4 + 1] * (d / z[i4 - 2 * pp - 2])
            end
            emin0 = min(emin0, z[i4 - 2 * pp])
        end
        z[4 * n0 - pp - 2] = d

        qmx = z[4 * i0 - pp - 2]
        for i4 in (4 * i0 - pp + 2):4:(4 * n0 - pp - 2)
            qmx = max(qmx, z[i4])
        end
        pp = 1 - pp
    end

    # Persisted dqds bookkeeping (Fortran iter=2 …)
    tt = zeros(S, 1)
    dmin1_a = zeros(T, 1)
    dmin2_a = zeros(T, 1)
    dn_a = zeros(T, 1)
    dn1_a = zeros(T, 1)
    dn2_a = zeros(T, 1)
    g_a = zeros(T, 1)
    tau_a = zeros(T, 1)
    iter_a = zeros(S, 1); iter_a[1] = 2
    nfail_a = zeros(S, 1)
    ndiv_a = zeros(S, 1); ndiv_a[1] = 2 * (n0 - i0)

    n0_a = zeros(S, 1); n0_a[1] = n0

    _lasq2_mainloops!(tol2, ieee, cbias, half, four, trace_val,
                      z, n, info, tt, dmin1_a, dmin2_a, dn_a, dn1_a, dn2_a,
                      g_a, tau_a, iter_a, nfail_a, ndiv_a, n0_a)

    return nothing
end

function _lasq2_n2!(z::AbstractVector{T}, tol2::T, half::T, onev::T, info) where {T <: AbstractFloat}
    if z[1] < zero(T)
        info[1] = -201
        return
    elseif z[2] < zero(T)
        info[1] = -202
        return
    elseif z[3] < zero(T)
        info[1] = -203
        return
    end
    if z[3] > z[1]
        d = z[3]
        z[3] = z[1]
        z[1] = d
    end
    z[5] = z[1] + z[2] + z[3]
    if z[2] > z[3] * tol2
        t = half * ((z[1] - z[3]) + z[2])
        s = z[3] * (z[2] / t)
        if s <= t
            s = z[3] * (z[2] / (t * (onev + sqrt(onev + s / t))))
        else
            s = z[3] * (z[2] / (t + sqrt(t) * sqrt(t + s)))
        end
        t = z[1] + (s + z[2])
        z[3] *= (z[1] / t)
        z[1] = t
    end
    z[2] = z[3]
    z[6] = z[2] + z[1]
    return nothing
end

function _lasq2_mainloops!(tol2::Real, ieee, cbias::T, half::T, four::T,
                           trace_val, z::AbstractVector{T}, n::S, info,
                           tt::AbstractVector{S}, dmin1_a, dmin2_a, dn_a, dn1_a, dn2_a,
                           g_a, tau_a, iter_a, nfail_a, ndiv_a,
                           n0_a::AbstractVector{S}) where {T <: AbstractFloat, S <: Integer}

    zerot = zero(T)

    local_i0::S = 1

    for _out in 1:(n + 1)
        n0_cur = Int(n0_a[1])
        if n0_cur < 1
            # --- Finished: eigenvalues packed (LAPACK lines 565–587)
            for kk in 2:n
                z[kk] = z[4 * kk - 3]
            end
            sort!(view(z, 1:n), alg=QuickSort, rev=true)
            eigsum = zerot
            for kk in n:-1:1
                eigsum += z[kk]
            end
            z[2 * n + 1] = trace_val
            z[2 * n + 2] = eigsum
            z[2 * n + 3] = T(iter_a[1])
            z[2 * n + 4] = T(ndiv_a[1]) / T(n)^2
            it = iter_a[1]
            # Match OpenBLAS/LAPACK as linked via LBT: HUNDRD*NFAIL/REAL(ITER), not INTEGER(NFAIL/ITER).
            z[2 * n + 5] = T(100) * T(nfail_a[1]) / T(max(one(S), it))

            info[1] = 0
            return nothing
        end

        desig_a = zeros(T, 1)
        sigma_a = zeros(T, 1)
        sigma_a[1] = ifelse(n0_cur == n, zerot, -z[4 * n0_cur - 1])
        if sigma_a[1] < zerot
            info[1] = 1
            return nothing
        end

        emax::T = zerot
        emin_seg::T = (n0_cur > local_i0) ? abs(z[4 * n0_cur - 5]) : zerot
        qmin::T = z[4 * n0_cur - 3]
        qmax_hold::T = qmin

        rstart = 4 * n0_cur
        i4_here = S(4)
        jumped_split = false
        iv = rstart
        while iv >= 8
            if z[iv - 5] <= zerot
                jumped_split = true
                i4_here = S(iv)
                break
            end
            if qmin >= four * emax
                qmin = min(qmin, z[iv - 3])
                emax = max(emax, z[iv - 5])
            end
            qmax_hold = max(qmax_hold, z[iv - 7] + z[iv - 5])
            emin_seg = min(emin_seg, z[iv - 5])
            iv -= 4
        end
        if !jumped_split
            i4_here = S(4)
        end

        local_i0 = i4_here ÷ S(4)
        pp_a = zeros(S, 1)
        pp_a[1] = 0

        n0_eff = Int(n0_a[1])
        if n0_eff - local_i0 > 1
            dee = z[4 * local_i0 - 3]
            deemin = dee
            kmin = local_i0
            for jj in (4 * local_i0 + 1):4:(4 * n0_eff - 3)
                dee = z[jj] * (dee / (dee + z[jj - 2]))
                if dee <= deemin
                    deemin = dee
                    kmin = (jj + 3) ÷ 4
                end
            end
            if (kmin - local_i0) * 2 < n0_eff - kmin &&
               deemin <= half * z[4 * n0_eff - 3]
                ipn4 = 4 * (local_i0 + n0_eff)
                pp_a[1] = 2
                for j4 in (4 * local_i0):4:(2 * (local_i0 + n0_eff - 1))
                    temp = z[j4 - 3]
                    z[j4 - 3] = z[ipn4 - j4 - 3]
                    z[ipn4 - j4 - 3] = temp
                    temp = z[j4 - 2]
                    z[j4 - 2] = z[ipn4 - j4 - 2]
                    z[ipn4 - j4 - 2] = temp
                    temp = z[j4 - 1]
                    z[j4 - 1] = z[ipn4 - j4 - 5]
                    z[ipn4 - j4 - 5] = temp
                    temp = z[j4]
                    z[j4] = z[ipn4 - j4 - 4]
                    z[ipn4 - j4 - 4] = temp
                end
            end
        end

        dmin_a = zeros(T, 1)
        dmin_a[1] = -max(zerot, qmin - T(2) * sqrt(max(qmin, zerot)) * sqrt(max(emax, zerot)))

        nbig = 100 * (Int(n0_a[1]) - local_i0 + 1)

        qmax_holder = T[qmax_hold]

        inner_ok = false
        for _in in 1:nbig
            if local_i0 > Int(n0_a[1])
                inner_ok = true
                break
            end

            lasq3!(local_i0, n0_a, z, pp_a, dmin_a, sigma_a, desig_a,
                   qmax_holder[1], nfail_a, iter_a, ndiv_a, ieee,
                   tt, dmin1_a, dmin2_a, dn_a, dn1_a, dn2_a,
                   g_a, tau_a)

            pp_a[1] = 1 - pp_a[1]

            n0_here = Int(n0_a[1])
            if pp_a[1] == 0 && (n0_here - local_i0 >= 3)
                sigmas = sigma_a[1]
                if z[4 * n0_here] <= tol2 * qmax_holder[1] || z[4 * n0_here - 1] <= tol2 * sigmas
                    splt = local_i0 - 1
                    qsx = z[4 * local_i0 - 3]
                    emin_here = z[4 * local_i0 - 1]
                    oldemn_here = z[4 * local_i0]
                    for i4ss in (4 * local_i0):4:(4 * (n0_here - 3))
                        if z[i4ss] <= tol2 * z[i4ss - 3] ||
                           z[i4ss - 1] <= tol2 * sigmas
                            z[i4ss - 1] = -sigmas
                            splt = i4ss ÷ 4
                            qsx = zerot
                            emin_here = z[i4ss + 3]
                            oldemn_here = z[i4ss + 4]
                        else
                            qsx = max(qsx, z[i4ss + 1])
                            emin_here = min(emin_here, z[i4ss - 1])
                            oldemn_here = min(oldemn_here, z[i4ss])
                        end
                    end
                    z[4 * n0_here - 1] = emin_here
                    z[4 * n0_here] = oldemn_here
                    local_i0 = splt + 1
                end
            end
        end

        if inner_ok
            continue
        end

        info[1] = 2
        # Failure recovery (LAPACK lines 508–548)
        i0_bad = local_i0
        n0_bad = Int(n0_a[1])
        i1::S = i0_bad
        n1_b::S = n0_bad

        sigma_rec = sigma_a[1]
        while true
            tempq = z[4 * i0_bad - 3]
            z[4 * i0_bad - 3] = z[4 * i0_bad - 3] + sigma_rec
            for kk in (i0_bad + 1):n0_bad
                tempe = z[4 * kk - 5]
                z[4 * kk - 5] = z[4 * kk - 5] * (tempq / z[4 * kk - 7])
                tempq = z[4 * kk - 3]
                z[4 * kk - 3] = z[4 * kk - 3] + sigma_rec + tempe - z[4 * kk - 5]
            end

            if i1 > 1
                n1_b = i1 - 1
                while i1 >= 2 && z[4 * i1 - 5] >= zerot
                    i1 -= 1
                end
                if i1 >= 1
                    sigma_rec = -z[4 * n1_b - 1]
                    continue
                end
            end
            break
        end

        for kk in 1:n
            z[2 * kk - 1] = z[4 * kk - 3]
            if kk < n0_bad
                z[2 * kk] = z[4 * kk - 1]
            else
                z[2 * kk] = zerot
            end
        end
        return nothing
    end

    info[1] = 3
    return nothing
end
