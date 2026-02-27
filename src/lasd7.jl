using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

#=
NOtes:

LLook at while look in jprev > 0 block to understand algorithm and what can be optimzied:

move said block inside for loop to avoid conditional Check: Done

look for other points of optimziation.

Benchmark different parts of the Julia function and lapack function by using the compiled LAPACK code
    from the OpenBLAS repo I cloned

=#


#=
Purpose:
!>
!> SLASD7 merges the two sets of singular values together into a single
!> sorted set. Then it tries to deflate the size of the problem. There
!> are two ways in which deflation can occur:  when two or more singular
!> values are close together or if there is a tiny entry in the Z
!> vector. For each such occurrence the order of the related
!> secular equation problem is reduced by one.
!>
!> SLASD7 is called from SLASD6.
!> 
Parameters
[in]	ICOMPQ	
!>          ICOMPQ is INTEGER
!>          Specifies whether singular vectors are to be computed
!>          in compact form, as follows:
!>          = 0: Compute singular values only.
!>          = 1: Compute singular vectors of upper
!>               bidiagonal matrix in compact form.
!> 
[in]	NL	
!>          NL is INTEGER
!>         The row dimension of the upper block. NL >= 1.
!> 
[in]	NR	
!>          NR is INTEGER
!>         The row dimension of the lower block. NR >= 1.
!> 
[in]	SQRE	
!>          SQRE is INTEGER
!>         = 0: the lower block is an NR-by-NR square matrix.
!>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
!>
!>         The bidiagonal matrix has
!>         N = NL + NR + 1 rows and
!>         M = N + SQRE >= N columns.
!> 
[out]	K	
!>          K is INTEGER
!>         Contains the dimension of the non-deflated matrix, this is
!>         the order of the related secular equation. 1 <= K <=N.
!> 
[in,out]	D	
!>          D is REAL array, dimension ( N )
!>         On entry D contains the singular values of the two submatrices
!>         to be combined. On exit D contains the trailing (N-K) updated
!>         singular values (those which were deflated) sorted into
!>         increasing order.
!> 
[out]	Z	
!>          Z is REAL array, dimension ( M )
!>         On exit Z contains the updating row vector in the secular
!>         equation.
!> 
[out]	ZW	
!>          ZW is REAL array, dimension ( M )
!>         Workspace for Z.
!> 
[in,out]	VF	
!>          VF is REAL array, dimension ( M )
!>         On entry, VF(1:NL+1) contains the first components of all
!>         right singular vectors of the upper block; and VF(NL+2:M)
!>         contains the first components of all right singular vectors
!>         of the lower block. On exit, VF contains the first components
!>         of all right singular vectors of the bidiagonal matrix.
!> 
[out]	VFW	
!>          VFW is REAL array, dimension ( M )
!>         Workspace for VF.
!> 
[in,out]	VL	
!>          VL is REAL array, dimension ( M )
!>         On entry, VL(1:NL+1) contains the  last components of all
!>         right singular vectors of the upper block; and VL(NL+2:M)
!>         contains the last components of all right singular vectors
!>         of the lower block. On exit, VL contains the last components
!>         of all right singular vectors of the bidiagonal matrix.
!> 
[out]	VLW	
!>          VLW is REAL array, dimension ( M )
!>         Workspace for VL.
!> 
[in]	ALPHA	
!>          ALPHA is REAL
!>         Contains the diagonal element associated with the added row.
!> 
[in]	BETA	
!>          BETA is REAL
!>         Contains the off-diagonal element associated with the added
!>         row.
!> 
[out]	DSIGMA	
!>          DSIGMA is REAL array, dimension ( N )
!>         Contains a copy of the diagonal elements (K-1 singular values
!>         and one zero) in the secular equation.
!> 
[out]	IDX	
!>          IDX is INTEGER array, dimension ( N )
!>         This will contain the permutation used to sort the contents of
!>         D into ascending order.
!> 
[out]	IDXP	
!>          IDXP is INTEGER array, dimension ( N )
!>         This will contain the permutation used to place deflated
!>         values of D at the end of the array. On output IDXP(2:K)
!>         points to the nondeflated D-values and IDXP(K+1:N)
!>         points to the deflated singular values.
!> 
[in]	IDXQ	
!>          IDXQ is INTEGER array, dimension ( N )
!>         This contains the permutation which separately sorts the two
!>         sub-problems in D into ascending order.  Note that entries in
!>         the first half of this permutation must first be moved one
!>         position backward; and entries in the second half
!>         must first have NL+1 added to their values.
!> 
[out]	PERM	
!>          PERM is INTEGER array, dimension ( N )
!>         The permutations (from deflation and sorting) to be applied
!>         to each singular block. Not referenced if ICOMPQ = 0.
!> 
[out]	GIVPTR	
!>          GIVPTR is INTEGER
!>         The number of Givens rotations which took place in this
!>         subproblem. Not referenced if ICOMPQ = 0.
!> 
[out]	GIVCOL	
!>          GIVCOL is INTEGER array, dimension ( LDGCOL, 2 )
!>         Each pair of numbers indicates a pair of columns to take place
!>         in a Givens rotation. Not referenced if ICOMPQ = 0.
!> 
[in]	LDGCOL	
!>          LDGCOL is INTEGER
!>         The leading dimension of GIVCOL, must be at least N.
!> 
[out]	GIVNUM	
!>          GIVNUM is REAL array, dimension ( LDGNUM, 2 )
!>         Each number indicates the C or S value to be used in the
!>         corresponding Givens rotation. Not referenced if ICOMPQ = 0.
!> 
[in]	LDGNUM	
!>          LDGNUM is INTEGER
!>         The leading dimension of GIVNUM, must be at least N.
!> 
[out]	C	
!>          C is REAL
!>         C contains garbage if SQRE =0 and the C-value of a Givens
!>         rotation related to the right null space if SQRE = 1.
!> 
[out]	S	
!>          S is REAL
!>         S contains garbage if SQRE =0 and the S-value of a Givens
!>         rotation related to the right null space if SQRE = 1.
!> 
[out]	INFO	
!>          INFO is INTEGER
!>         = 0:  successful exit.
!>         < 0:  if INFO = -i, the i-th argument had an illegal value.
!> 
=#
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
    #Add vector size check at start
    info .= 0

    n = nl + nr + 1

    m = n + sqre

    @assert length(d) == n
    @assert length(z) == m
    @assert length(zw) == m
    @assert length(vf) == m
    @assert length(vfw) == m
    @assert length(vl) == m
    @assert length(vlw) == m
    @assert length(dsigma) == n
    @assert length(idxp) == n
    @assert length(idxq) == n
    @assert length(perm) == n
    @assert ldgnum >= n
    @assert ldgcol >= n
    @assert size(givcol) == (ldgcol, 2)
    @assert size(givnum) == (ldgcol, 2)

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
    @inbounds for i in nl:-1:1
        z[i+1] = alpha*vl[i]
        vl[i] = zero(T)
        vf[i+1] = vf[i]
        d[i+1] = d[i]
        idxq[i+1] = idxq[i]+1
    end
    
    vf[1] = tau
    
    # Generate the second part of the vector Z.
    
    @inbounds @simd for i in nlp2:m
        z[i] = beta*vf[i]
        vf[i] = zero(T)
    end

    # Sort the singular values into increasing order
    idxq[nlp2:n] .+= nlp1

    # DSIGMA, IDXC, IDXC, and ZW are used as storage space.
    @inbounds @simd for i in 2:n
        dsigma[i] = d[idxq[i]]
        zw[i] = z[idxq[i]]
        vfw[i] = vf[idxq[i]]
        vlw[i] = vl[idxq[i]]
    end

    slamrg!(nl, nr, view(dsigma, 2:n), 1, 1, view(idx, 2:n))

    @inbounds @simd for i in 2:n
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
    @inbounds for j in 2:n
        if abs(z[j]) <= tol
            k2 -=  1
            idxp[k2] = j
            if j  == n
                break
            end
        else
            jprev = j
            
            #=
            What this loop does:
                Deflates values that have a small z[j] or deflates jprev if abs(d[j]-d[jprev]) <= tol.
                    When it deflates, it puts the index that it's deflating at idxp[k2] after decrementing
                    k2.
            =#

            @inbounds for jj in j+1:n
                if abs(z[jj]) <= tol
                    # Deflate due to small z component.
                    k2 -= 1
                    idxp[k2] = jj
                else
                    # Check if singular values are close enough to allow deflation.
                    # if abs(d[j]-d[jprev]) <= tol
                    if abs(d[jj]-d[jprev]) <= tol
                        
                        s .= z[jprev]
                        c .= z[jj]
                        # Find sqrt(a**2+b**2) without overflow or
                        # destructive underflow.

                        tau = hypot(c[], s[])
    
                        z[jj] = tau
                        z[jprev] = zero(T)
                        c ./=  tau
                        s ./= -tau

                        # Record the appropriate Givens rotation

                        if icompq == 1
                            givptr .+= 1
                            idxjp = idxq[idx[jprev]+1]
                            idxj = idxq[idx[jj]+1]

                            if idxjp <= nlp1
                                idxjp -= 1
                            end

                            if idxj <= nlp1
                                idxj -= 1
                            end

                            givcol[givptr[], 2] = idxjp
                            givcol[givptr[], 1] = idxj

                            #Records rotation values that can potentially be used to reconstruct
                            # Is later used by slasda since it repeatedly calls slasd6 which calls this
                            # It is also used by sbdsdc 
                            givnum[givptr[], 2] = c[]
                            givnum[givptr[], 1] = s[]
                        end
                        
                        #inline srot instead of calling function
                        # stemp = c[] * vf[jprev] + s[]*vf[j]
                        # vf[j] = c[] * vf[j] - s[]*vf[jprev]
                        # vf[jprev] = stemp
                        srot!(1, view(vf, jprev:jprev), 1, view(vf, jj:jj), 1, c[], s[])
                        srot!(1, view(vl,jprev:jprev), 1, view(vl, jj:jj), 1, c[], s[])
                        #inline srot instead of calling function
                        # stemp = c[] * vl[jprev] + s[]*vl[j]
                        # vl[j] = c[] * vl[j] - s[]*vl[jprev]
                        # vl[jprev] = stemp

                        k2 -= 1
                        idxp[k2] = jprev
                        jprev = jj
                    else
                        k .+= 1
                        zw[k[]] = z[jprev]
                        dsigma[k[]] = d[jprev]
                        idxp[k[]] = jprev
                        jprev = jj
                    end

                end
            end

            # Record the last singular value

            k .+= 1
            zw[k[]] = z[jprev]
            dsigma[k[]] = d[jprev]
            idxp[k[]] = jprev
            break
        end
    end

    # Sort the singular values into DSIGMA. The singular values which
    # were not deflated go into the first K slots of DSIGMA, except
    # that DSIGMA(1) is treated separately.
    @inbounds @simd for j in 2:n

        jp = idxp[j]
        dsigma[j] = d[jp]
        vfw[j] = vf[jp]
        vlw[j] = vl[jp]
    end

    if icompq == 1
        @inbounds @simd for j in 2:n
            jp = idxp[j]
            perm[j] = idxq[idx[jp]+1]
            if perm[j] <= nlp1
                perm[j] -= 1
            end
        end
    end
    r = k[]+1:n
    @inbounds @simd for i in r
        d[i] = dsigma[i]
    end
    # copyto!(view(d, r), view(dsigma, r))

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
    @inbounds @simd for i in r
        z[i] = zw[i]
    end
    # copyto!(view(z, r), view(zw, r))
    # r = 2:n
    @inbounds @simd for i in 2:n
        vf[i] = vfw[i]
        vl[i] = vlw[i]
    end
    # copyto!(view(vf, r), view(vfw, r))
    # copyto!(view(vl, r), view(vlw, r))
end
