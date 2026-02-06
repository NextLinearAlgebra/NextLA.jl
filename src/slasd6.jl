
using LinearAlgebra

#=
Parameters
[in]	ICOMPQ	
!>          ICOMPQ is INTEGER
!>         Specifies whether singular vectors are to be computed in
!>         factored form:
!>         = 0: Compute singular values only.
!>         = 1: Compute singular vectors in factored form as well.
!> 
[in]	NL	
!>          NL is INTEGER
!>         The row dimension of the upper block.  NL >= 1.
!> 
[in]	NR	
!>          NR is INTEGER
!>         The row dimension of the lower block.  NR >= 1.
!> 
[in]	SQRE	
!>          SQRE is INTEGER
!>         = 0: the lower block is an NR-by-NR square matrix.
!>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
!>
!>         The bidiagonal matrix has row dimension N = NL + NR + 1,
!>         and column dimension M = N + SQRE.
!> 
[in,out]	D	
!>          D is REAL array, dimension (NL+NR+1).
!>         On entry D(1:NL,1:NL) contains the singular values of the
!>         upper block, and D(NL+2:N) contains the singular values
!>         of the lower block. On exit D(1:N) contains the singular
!>         values of the modified matrix.
!> 
[in,out]	VF	
!>          VF is REAL array, dimension (M)
!>         On entry, VF(1:NL+1) contains the first components of all
!>         right singular vectors of the upper block; and VF(NL+2:M)
!>         contains the first components of all right singular vectors
!>         of the lower block. On exit, VF contains the first components
!>         of all right singular vectors of the bidiagonal matrix.
!> 
[in,out]	VL	
!>          VL is REAL array, dimension (M)
!>         On entry, VL(1:NL+1) contains the  last components of all
!>         right singular vectors of the upper block; and VL(NL+2:M)
!>         contains the last components of all right singular vectors of
!>         the lower block. On exit, VL contains the last components of
!>         all right singular vectors of the bidiagonal matrix.
!> 
[in,out]	ALPHA	
!>          ALPHA is REAL
!>         Contains the diagonal element associated with the added row.
!> 
[in,out]	BETA	
!>          BETA is REAL
!>         Contains the off-diagonal element associated with the added
!>         row.
!> 
[in,out]	IDXQ	
!>          IDXQ is INTEGER array, dimension (N)
!>         This contains the permutation which will reintegrate the
!>         subproblem just solved back into sorted order, i.e.
!>         D( IDXQ( I = 1, N ) ) will be in ascending order.
!> 
[out]	PERM	
!>          PERM is INTEGER array, dimension ( N )
!>         The permutations (from deflation and sorting) to be applied
!>         to each block. Not referenced if ICOMPQ = 0.
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
!>         leading dimension of GIVCOL, must be at least N.
!> 
[out]	GIVNUM	
!>          GIVNUM is REAL array, dimension ( LDGNUM, 2 )
!>         Each number indicates the C or S value to be used in the
!>         corresponding Givens rotation. Not referenced if ICOMPQ = 0.
!> 
[in]	LDGNUM	
!>          LDGNUM is INTEGER
!>         The leading dimension of GIVNUM and POLES, must be at least N.
!> 
[out]	POLES	
!>          POLES is REAL array, dimension ( LDGNUM, 2 )
!>         On exit, POLES(1,*) is an array containing the new singular
!>         values obtained from solving the secular equation, and
!>         POLES(2,*) is an array containing the poles in the secular
!>         equation. Not referenced if ICOMPQ = 0.
!> 
[out]	DIFL	
!>          DIFL is REAL array, dimension ( N )
!>         On exit, DIFL(I) is the distance between I-th updated
!>         (undeflated) singular value and the I-th (undeflated) old
!>         singular value.
!> 
[out]	DIFR	
!>          DIFR is REAL array,
!>                   dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and
!>                   dimension ( K ) if ICOMPQ = 0.
!>          On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not
!>          defined and will not be referenced.
!>
!>          If ICOMPQ = 1, DIFR(1:K,2) is an array containing the
!>          normalizing factors for the right singular vector matrix.
!>
!>         See SLASD8 for details on DIFL and DIFR.
!> 
[out]	Z	
!>          Z is REAL array, dimension ( M )
!>         The first elements of this array contain the components
!>         of the deflation-adjusted updating row vector.
!> 
[out]	K	
!>          K is INTEGER
!>         Contains the dimension of the non-deflated matrix,
!>         This is the order of the related secular equation. 1 <= K <=N.
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
[out]	WORK	
!>          WORK is REAL array, dimension ( 4 * M + 4). The plus 4 is for the things we need to write out to
!> 
[out]	IWORK	
!>          IWORK is INTEGER array, dimension ( 3 * N + 3) The plus 3 is for the things we need to write out to that are integers
!> 
[out]	INFO	
!>          INFO is INTEGER
!>          = 0:  successful exit.
!>          < 0:  if INFO = -i, the i-th argument had an illegal value.
!>          > 0:  if INFO = 1, a singular value did not converge
=#
function slasd6!(icompq::Integer, nl::Integer, nr::Integer, sqre::Integer, d::AbstractVector{<:AbstractFloat},
                 vf::AbstractVector{<:AbstractFloat}, vl::AbstractVector{<:AbstractFloat}, idxq::Vector{Integer},
                 perm::Vector{Integer}, givcol::AbstractMatrix{Integer},
                 ldgcol::Integer, givnum::AbstractMatrix{<:AbstractFloat}, ldgnum::Integer,
                 poles::AbstractMatrix{<:AbstractFloat}, difl::AbstractVector{<:AbstractFloat},
                 difr::AbstractMatrix{<:AbstractFloat}, z::AbstractVector{<:AbstractFloat},
                 work::AbstractVector{<:AbstractFloat}, iwork::AbstractVector{Integer}, )
    
    
    n = nl + nr + 1
    m = n + sqre

    #=
    for iwork:
        3n + 1 = givptr
        3n + 2 = k
        3n + 3 = info
    =#
    k = 3n + 2
    k = 3n + 2

    #=
    for work:
        4m + 1 = alpha
        4m + 2 = beta
        4m + 3 = c
        4m + 4 = s
    =#
    
    alpha = view(work, 4m + 1:4m + 1)
    beta =  view(work, 4m + 2:4m + 2)
    c = view(work, 4m + 3:4m + 3)
    s = view(work, 4m + 4:4m + 4)
    info = view(iwork, 3n + 3:3n + 3)
    info .= 0

    if icompq < 0 || icompq > 1
        info .= -1

    elseif nl < 1
        info .= -1
    elseif nr < 1
        info .= -3
    elseif sqre < - || sqre > 1
        info .= -4
    elseif ldgcol < n
        info .= -14
    elseif ldfnum < N
        info .= -16
    end

    if info[1] != 0
        return
    end
    

#=
    The following values are for bookkeeping purposes only.  They are
    integer pointers which indicate the portion of the workspace
    used by a particular array in SLASD7 and SLASD8.
=#

    isigma = 1
    iw = isigma + n
    ivfw = iw + m
    ivlw = ivfw + m

    idx = 1
    idxc = idx + n
    idxp = idxc + n

    orgnrm = max(abs(alpha[1]), abs(beta[1]))

    d[nl+1] = zero(eltype(d))

    for 1:n 
        if abs(d[i]) > orgnrm
            orgnrm = abs(d[i])
        end
    end

    d ./= orgnrm
    alpha ./= /orgnrm
    beta ./= orgnrm

    slasd7!(icompq, nl, nr, sqre, view(iwork, 3n + 2:3n + 2), d, z, view(work, iw:ivfw-1), vf, view(work, ivfw:ivlw-1), vl,
            view(work, ivlw:4*m), work[alpha], beta, view(work, isigma:iw-1), view(iwork, idx:idxc-1), view(iwork, idxp:3*n),
            idxq, perm, view(iwork, 3n + 1:3n + 1), givcol, ldgcol, givnum, ldgnum, c, c, info)

    slasd8!(icompq, iwork[k], d, z, vf, vl, difl, difr, ldgnum, view(work, isigma:isigma+iwork[k]), view(work, iw:iw+3*iwork[k]), info)

    if info[1] != 0
        return
    end

    if icompq == 1
        copyto!(view(poles, 1:iwork[k], 1), view(d, 1:iwork[k]))
        copyto!(view(poles, 1:iwork[k], 2), view(word, isigma:isigma+iwork[k]))
    end

    d .*= orgnrm
    n1 = iwork[k]
    n2 = n - iwork[k]

    slamrg!(n1, n2, d, 1, -1, idxq)

end
