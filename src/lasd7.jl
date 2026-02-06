using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

function lasd7!(icompq::S, nl::S, nr::S, sqre::S, k::AbstractArray{<:Integer},
    d::AbstractVector{T}, z::AbstractVector{T}, zw::AbstractVector{T},
    vf::AbstractVector{T}, vfw::AbstractVector{T}, vl::AbstractVector{T},
    vlw::AbstractVector{T}, alpha::T, beta::T,
    dsigma::AbstractVector{T}, idx::AbstractVector{<:Integer}, idxp::AbstractVector{<:Integer},
    idxq::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}, givptr::AbstractArray{<:Integer}, givcol::AbstractMatrix{<:Integer},
    ldgcol::S, givnum::AbstractMatrix{T}, ldgnum::S, c::AbstractArray{T},
    s::AbstractArray{T}, info::AbstractArray{<:Integer}) where {T <:AbstractFloat, S <:Integer}

    #=
        k, givptr, c, and s are in vectors of size one so that we can 
            save space by preallocating the memory and make the memory contiguous

    =#

    info .= 0

    n = nl + nr + 1

    m = n + sqre

    if icompq < 0 || icompq > 1

        info .= -1
    elseif nl < 1
        info .= -2
    elseif nr < 1
        info .= -3
    elseif sqre < 0 || sqre > 1
        info .= -4
    elseif ldgcol < n
        info .= -22
    elseif ldgnum < n
        info .= -24
    end

    if info[] != 0
        return
    end

    nlp1 = nl + 1
    nlp2 = nl + 2
    if icompq == 1
        givptr .= 0
    end
    
    #=     
        Generate the first part of the vector Z and move the singular
        values in the first part of D one position backward.
    =#

    z1 = alpha*vl[nlp1]
    vl[nlp1] = zero(T)
    tau = vf[nlp1]
    for i in nl:-1:1
        z[i+1] = alpha*vl[i]
        vl[i] = zero(T)
        vf[i+1] = vf[i]
        d[i+1] = d[i]
        idxq[i+1] = idxq[i]+1
    end
    
    vf[1] = tau
    
    # Generate the second part of the vector Z.
    
    for i in nlp2:m
        z[i] = beta*vf[i]
        vf[i] = zero(T)
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

    slamrg!(nl, nr, view(dsigma, 2:n), 1, 1, view(idx, 2:n))

    for i in 2:n
        idxi = 1 + idx[i]
        d[i] = dsigma[idxi]
        z[i] = zw[idxi]
        vf[i] = vfw[idxi]
        vl[i] = vlw[idxi]
    end


    # Calculate the allowable deflation tolerance
    tol = max(abs(alpha), abs(beta))
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
    tol = T(64)*mach_eps*max(abs(d[n]), tol)

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

    k .= 1
    k2 = n + 1
    jprev = 0
    for j in 2:n
        if abs(z[j]) <= tol
            k2 -=  1
            idxp[k2] = j
            if j  == n
                jprev = 0
                break
            end
        else
            jprev = j
            break
        end
    end

    if jprev > 0
        j = jprev

        while j < n
            j += 1
            if abs(z[j]) <= tol
                # Deflate due to small z component.
                k2 -= 1
                idxp[k2] = j
            else
                # Check if singular values are close enough to allow deflation.
                # if abs(d[j]-d[jprev]) <= tol
                if abs(d[j]-d[jprev]) <= tol
                    
                    s .= z[jprev]
                    c .= z[j]
                    # Find sqrt(a**2+b**2) without overflow or
                    # destructive underflow.

                    tau = hypot(c[], s[])
 
                    z[j] = tau
                    z[jprev] = zero(T)
                    c ./=  tau
                    s ./= -tau

                    # Record the appropriate Givens rotation

                    if icompq == 1
                        givptr .+= 1
                        idxjp = idxq[idx[jprev]+1]
                        idxj = idxq[idx[j]+1]

                        if idxjp <= nlp1
                            idxjp -= 1
                        end

                        if idxj <= nlp1

                            idxj -= 1
                        end

                        givcol[givptr[], 2] = idxjp
                        givcol[givptr[], 1] = idxj
                        givnum[givptr[], 2] = c[]
                        givnum[givptr[], 1] = s[]
                    end

                    srot!(1, view(vf, jprev:jprev), 1, view(vf, j:j), 1, c[], s[])
                    srot!(1, view(vl,jprev:jprev), 1, view(vl, j:j), 1, c[], s[])

                    k2 -= 1
                    idxp[k2] = jprev
                    jprev = j
                else
                    k .+= 1
                    zw[k[]] = z[jprev]
                    dsigma[k[]] = d[jprev]
                    idxp[k[]] = jprev
                    jprev = j
                end

            end
        end

        # Record the last singular value

        k .+= 1
        zw[k[]] = z[jprev]
        dsigma[k[]] = d[jprev]
        idxp[k[]] = jprev

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
            perm[j] = idxq[idx[jp]+1]
            if perm[j] <= nlp1
                perm[j] -= 1
            end
        end
    end
    r = k[]+1:n
    copyto!(view(d, r), view(dsigma, r))

    dsigma[1] = zero(T)
    hlftol = tol/(T(2))

    if abs(dsigma[2]) <= hlftol
        dsigma[2] = hlftol
    end

    if m > n
        z[1] = hypot(z1, z[m])

        if z[1] <= tol
            
            c  .= one(T)
            s .= zero(T)
            z[1] = tol
        else
            c .= z1/z[1]
            s .= -z[m]/z[1]
        end

        srot!(1, view(vf, m:m), 1, view(vf, 1:1), 1, c[], s[])
        srot!(1, view(vl, m:m), 1, view(vl, 1:1), 1, c[], s[])
    else
        if abs(z1) <= tol
            z[1] = tol
        else
            z[1] = z1
        end
    end

    r = 2:k[]
    copyto!(view(z, r), view(zw, r))
    r = 2:n
    copyto!(view(vf, r), view(vfw,r))
    copyto!(view(vl, r), view(vlw, r))


end
