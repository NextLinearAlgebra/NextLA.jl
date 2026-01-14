using LinearAlgebra

function slasd7!(icompq::Integer, nl::Integer, nr::Integer, sqre::Integer, k::AbstractVector{Integer},
    d::AbstractVector{T}, z::AbstractVector{T}, zw::AbstractVector{T},
    vf::AbstractVector{T}, vfw::AbstractVector{T}, vl::AbstractVector{T},
    vlw::AbstractVector{T}, alpha::AbstractVector{T}, beta::AbstractVector{T},
    dsigma::AbstractVector{T}, idx::AbstractVector{Integer}, idxp::AbstractVector{Integer},
    idxq::AbstractVector{Integer}, perm::AbstractVector{Integer}, givptr::AbstractVector{Integer}, givcol::AbstractMatrix{Integer},
    ldgcol::Integer, givnum::AbstractMatrix{T}, ldgnum::Integer, c::AbstractVector{T},
    s::AbstractVector{T}, info::AbstractVector{Integer}) where {T <:AbstractFloat}
    #=
        k, alpha, beta, givptr, c, and s are in vectors of size one so that we can 
            save space by preallocating the memory and make the memory contiguous

    =#

    info[1] = 0

    n = nl + nr + 1

    m = n + sqre

    if icompq < 0 || icompq > 1

        info[1] = -1
    elseif nl < 1
        info[1] = -2
    elseif nr < 1
        info[1] = -3
    elseif sqre < 0 || sqre > 1
        info[1] = -4
    elseif ldgcol < n
        info[1] = -22
    elseif ldgnum < n
        info[1] = -24
    end

    if info[1] != 0
        return
    end
    nlp1 = nl + 1
    nlp2 = nl + 2
    if icompq == 1
        givptr[1] = 0
    end
    #=     
        Generate the first part of the vector Z and move the singular
        values in the first part of D one position backward.
    =#

    z1 = alpha[1]*vl[nlp1]
    vl[nlp1] = zero(eltype(vl))
    tau = vf[nlp1]

    for i in nl:-1:1
        z[i+1] = alpha[1]*vl[i]
        vl[i] = zero(eltype(vl))
        vf[i+1] = vf[i]
        d[i+1] = d[i]
        idxq[i+1] = idxq[i]+1
    end

    vf[1] = tau

    # Generate the second part of the vector Z.

    for i in nlp2:m
        z[i] = beta[1]*vf[i]
        vf[i] = zero(eltype(vf))
    end

    # Sort the singular values into increasing order
    idxq[nlp2:n] .+= nlp1

    # DSIGMA, IDXC, IDXC, and ZW are used as storage space.
    for i in 2:n
        dsigma[i] = d[idxq[i]]
        zw[i] = z[idxq[i]]
        vfw[i] = vf[idxq[i]]
        vlw[i] = vl[idxq[i]]
    end

    slamrg!(nl, nr, @view dsigma[2:end], 1, 1, @view idx[2:end])

    for i in 2:N
        idxi = 1 + idx[i]
        d[i] = dsigma[idxi]
        z[i] = zw[idxi]
        vf[i] = vfw[idxi]
        vl[i] = vlw[idxi]
    end

    # Calculate the allowable deflation tolerance
    mach_eps = eps(eltype(s))
    tol = max(abs(alpha[1]), abs[beta[1]])
    tol = 16*one(tol)*eps*max(abs(d[n]), tol)

    #=
        There are 2 kinds of deflation -- first a value in the z-vector
        is small, second two (or more) singular values are very close
        together (their difference is small).

        If the value in the z-vector is small, we simply permute the
        array so that the corresponding singular value is moved to the
        end.

        If two values in the D-vector are close, we perform a two-sided
        rotation designed to make one of the corresponding z-vector
        entries zero, and then permute the array so that the deflated
        singular value is moved to the end.

        If there are multiple singular values then the problem deflates.
        Here the number of equal singular values are found.  As each equal
        singular value is found, an elementary reflector is computed to
        rotate the corresponding singular subspace so that the
        corresponding components of Z are zero in this new basis.
    =#

    k = 1
    k2 = n + 1
    jprev = 0
    j = 2
    while j <= n
        if abs(z[j]) <= tol
            k2 = k2 - 1
            idxp[k2] = jul
            if j  == n
                break
            end
        else
            jprev = j
            break
        end
        j += 1
    end
    
    if j != n
        j = jprev

        while true
            j += 1
            if j > n
                break
            end

            if abs(z[j]) <= tol

                # Deflate due to small z component.

                k2 -= 1
                idxp[kn] = j
            else
                # Check if singular values are close enough to allow deflation.
                if abs(d[j]-d[jprev]) <= tol
                    s[1]  = z[jprev]
                    c[1] = z[j]
                    # Find sqrt(a**2+b**2) without overflow or
                    # destructive underflow.

                    tau = hypot(c[1], s[1])

                    z[j] = tau
                    c[1] = c[1] / tau
                    s[1] = -s[1] / tau

                    # Record the appropriate Givens rotation

                    if icompq == 1
                        givptr[1] += 1
                        idxjp = idxp[idx[jprev]+1]
                        idxj = idxq[idx[j]+1]

                        if idxjp <= nlp1
                            idxjp -= 1
                        end

                        if idxj <= nlp1

                            idxj -= 1
                        end

                        givcol[givptr[1], 2] = idxjp
                        givcol[givptr[1], 1] = idxj
                        givnum[givptr[1], 2] = c[1]
                        givnum[givptr[1], 1] = s[1]
                    end

                    srot(1, @view vf[jprev:jprev], 1, @view vf[j:j], 1, c[1], s[1])
                    srot(1, @view vl[jprev:jprev], 1, @view vl[j:j], 1, c[1], s[1])

                    k2 -= 1
                    idxp[k2] = jprev
                    jprev = j


                end

                k[1] += 1
                zq[k[1]] = z[jprev]
                dsigma[k[1]] = d[jprev]
                idxp[k[1]] = jprev
                jprev = j
            end
        end

        # Record the last singular value

        k[1] += 1
        zq[k[1]] = z[jprev]
        dsigma[k[1]] = d[jprev]
        idxp[k[1]] = jprev

    end

    # Sort the singular values into DSIGMA. The singular values which
    # were not deflated go into the first K slots of DSIGMA, except
    # that DSIGMA(1) is treated separately.

    for j in 2:n

        jp = idxp[j]
        dsigma[j] = d[jp]
        vfw[j] = vf[jp]
        vlw[j] = vl[jp]
    end

    if icompq == 1
        for j in 2:n
            jp = idxp[j]
            perm[j] = idxp[idx[jp]+1]
            if perm[j] <= nlp1
                perm[j] -= 1
            end
        end
    end
    r = k[1]+1:n
    copyto!(@view d[r], @view dsigma[r])

    dsigma[1] = zero(eltype(dsigma))
    hlftol = tol/(2*one(tol))

    if abs(dsigma[2]) <= hlftol
        dsigma[2] = hlftol
    end

    if m > n
        z[1] = hypot(z1, z[m])

        if z[1] <= tol
            
            c[1] = one(c[1])
            s[1] = zero(s[1])
            z[1] = tol
        else
            c[1] = z1/z[1]
            s[1] = -z[m]/z[1]
        end

        srot(1, @view vf[m:m], 1, @view vf[1:1], 1, c[1], s[1])
        srot(1, @view vl[m:m], 1, @view vl[1:1], 1, c[1], s[1])
    else
        if abs(z1) <= tol
            z[1] = tol
        else
            z[1] = z1
        end
    end

    r = 2:k[1]
    copyto!(z[r], zw[r])
    r = 2:n
    copyto!(vf[r], vfw[r])
    copyto!(vl[r], vlw[r])


end
