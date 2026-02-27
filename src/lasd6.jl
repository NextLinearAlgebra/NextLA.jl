using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
#=
Purpose:
!>
!> SLASD6 computes the SVD of an updated upper bidiagonal matrix B
!> obtained by merging two smaller ones by appending a row. This
!> routine is used only for the problem which requires all singular
!> values and optionally singular vector matrices in factored form.
!> B is an N-by-M matrix with N = NL + NR + 1 and M = N + SQRE.
!> A related subroutine, SLASD1, handles the case in which all singular
!> values and singular vectors of the bidiagonal matrix are desired.
!>
!> SLASD6 computes the SVD as follows:
!>
!>               ( D1(in)    0    0       0 )
!>   B = U(in) * (   Z1**T   a   Z2**T    b ) * VT(in)
!>               (   0       0   D2(in)   0 )
!>
!>     = U(out) * ( D(out) 0) * VT(out)
!>
!> where Z**T = (Z1**T a Z2**T b) = u**T VT**T, and u is a vector of dimension M
!> with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
!> elsewhere; and the entry b is empty if SQRE = 0.
!>
!> The singular values of B can be computed using D1, D2, the first
!> components of all the right singular vectors of the lower block, and
!> the last components of all the right singular vectors of the upper
!> block. These components are stored and updated in VF and VL,
!> respectively, in SLASD6. Hence U and VT are not explicitly
!> referenced.
!>
!> The singular values are stored in D. The algorithm consists of two
!> stages:
!>
!>       The first stage consists of deflating the size of the problem
!>       when there are multiple singular values or if there is a zero
!>       in the Z vector. For each such occurrence the dimension of the
!>       secular equation problem is reduced by one. This stage is
!>       performed by the routine SLASD7.
!>
!>       The second stage consists of calculating the updated
!>       singular values. This is done by finding the roots of the
!>       secular equation via the routine SLASD4 (as called by SLASD8).
!>       This routine also updates VF and VL and computes the distances
!>       between the updated singular values and the old singular
!>       values.
!>
!> SLASD6 is called from SLASDA.
!> 
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
!>          WORK is REAL array, dimension ( 4 * M )
!> 
[out]	IWORK	
!>          IWORK is INTEGER array, dimension ( 3 * N )
!> 
[out]	INFO	
!>          INFO is INTEGER
!>          = 0:  successful exit.
!>          < 0:  if INFO = -i, the i-th argument had an illegal value.
!>          > 0:  if INFO = 1, a singular value did not converge
!> 
=#
function lasd6!(icompq::S, nl::S, nr::S, sqre::S, d::AbstractVector{T},
                vf::AbstractVector{T}, vl::AbstractVector{T}, alpha::AbstractVector{T},
                beta::AbstractVector{T}, idxq::AbstractVector{S}, perm::AbstractVector{S},
                givptr::AbstractVector{S}, givcol::AbstractMatrix{S}, ldgcol::S, givnum::AbstractMatrix{T},
                ldgnum::S, poles::AbstractMatrix{T}, difl::AbstractMatrix{T},
                z::AbstractVector{T}, k::AbstractVector{S}, c::AbstractVector{T},
                s::AbstractVector{T}, work::AbstractVector{T}, iwork::AbstractVector{S},
                info::AbstractVector{S}) where {T <:AbstractFloat, S <:Integer}


end
