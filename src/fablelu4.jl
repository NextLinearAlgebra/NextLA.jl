using LinearAlgebra
using CUDA
using CUDA: CUSOLVER
using StochasticRounding
include("fullmixedprec.jl")   # FullMixedPrec, reconstruct_matrix
include("recmixedprectri.jl") # TriMixedPrec
include("matmul.jl")          # recgemm!
include("rectrxm.jl")         # unified_rectrxm!
include("wrappers.jl")

# =============================================================================
# Base case: non-pivoting CUSOLVER getrf
#
# cuSOLVER's dense getrf performs *no pivoting* when devIpiv is passed as a
# null device pointer -- this is the documented mechanism for the nonpivoting
# variant, so we wrap cusolverDn<t>getrf directly with devIpiv = CU_NULL.
# =============================================================================
for (bname, fname, elty) in
    ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
     (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64))
    @eval begin
        function getrf_nopivot!(A::StridedCuMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            devinfo = CuArray{Cint}(undef, 1)

            function bufferSize()
                out = Ref{Cint}(0)
                CUSOLVER.$bname(CUSOLVER.dense_handle(), m, n, A, lda, out)
                return out[] * sizeof($elty)
            end

            CUDA.with_workspace(bufferSize) do buffer
                CUSOLVER.$fname(CUSOLVER.dense_handle(), m, n, A, lda,
                                buffer, CU_NULL, devinfo)
            end
            return A
        end
    end
end

# Float16 base case: cuSOLVER has no fp16 getrf, so the base block is promoted
# to Float32, factorized, and copied back (mirroring the promote/copy-back
# pattern used by dispatch_trsm!/dispatch_trmm! in rectrxm.jl).
function getrf_nopivot!(A::StridedCuMatrix{Float16})
    A32 = Float32.(A)
    getrf_nopivot!(A32)
    copy!(A, A32)
    return A
end

# =============================================================================
# Recursive formulation of nonpivoting block LU
#
#   A = [A11 A12]  =  [L11  0 ] [U11 U12]
#       [A21 A22]     [L21 L22] [ 0  U22]
#
#   1. A11 <- L11 * U11                       (recursive LU)
#   2. A12 <- U12 = L11^-1 * A12              (left, lower, unit-diag TRSM)
#   3. A21 <- L21 = A21 * U11^-1              (right, upper, non-unit TRSM)
#   4. A22 <- A22 - L21 * U12                 (Schur complement, recgemm!)
#   5. A22 <- L22 * U22                       (recursive LU)
#
# L (unit lower) and U (upper) overwrite A in place, LAPACK-style.
# =============================================================================

"""
    getrf_recursive!(A, block_size)

Performs an in-place, nested recursive, nonpivoting LU factorization on the
dense matrix `A`. The recursion dynamically splits the matrix until the
sub-block size is less than or equal to `block_size`, at which point it falls
back to the CUSOLVER `getrf` routine (with pivoting disabled).
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        getrf_nopivot!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1,     1:n1]
    A12 = @view A[1:n1,     n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    getrf_recursive!(A11, block_size)

    if eltype(A11) == Float16
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)  # A12 <- L11^-1 A12
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)  # A21 <- A21 U11^-1
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    recgemm!(-1.0, A21, A12, 1.0, A22)                             # A22 <- A22 - L21 U12

    getrf_recursive!(A22, block_size)
end

# =============================================================================
# FullMixedPrec support
# =============================================================================

"""
    TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char)

Dynamically converts one triangle of an LU-factorized `FullMixedPrec` matrix
into a `TriMixedPrec` format (zero-copy: the off-diagonal blocks and their
quantization scales are shared, not duplicated).

- `uplo = 'L'` extracts the unit-lower factor `L` (reusing the `A21` blocks).
- `uplo = 'U'` extracts the upper factor `U` (reusing the `A12` blocks).
"""
function TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing,
            nothing, A.base_scale, A.BaseCase,
            uplo, A.sz
        )
    end

    OffDiag  = (uplo == 'L') ? A.A21 : A.A12
    OffScale = (uplo == 'L') ? A.A21_scale : A.A12_scale

    return TriMixedPrec{T_Base}(
        TriMixedPrec(A.A11, uplo),
        TriMixedPrec(A.A22, uplo),
        OffDiag,
        OffScale,
        nothing,
        nothing,
        uplo,
        A.sz
    )
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs an in-place, nested recursive, nonpivoting LU factorization on a full
mixed-precision matrix structure `A`. The recursion performs the two panel
TRSM updates and the Schur-complement GEMM update through the mixed-precision
subroutines, and falls back to the CUSOLVER routine at the base case.

The unit-lower factor `L` overwrites the strictly-lower storage (`A21` blocks
and the lower triangles of the base cases); the upper factor `U` overwrites
the upper storage (`A12` blocks and the upper triangles of the base cases).

Note: fp16 dynamic-quantization scales on the *off-diagonal* blocks are
handled exactly (they factor linearly through the TRSMs, and are folded into
the Schur-complement `alpha` below). Diagonal base cases are assumed to have
`base_scale === nothing` (i.e. the diagonal blocks fit the fp16 range), the
same assumption made by the reference `potrf_recursive!` implementation.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    getrf_recursive!(A.A11)

    # A12 <- U12 = L11^-1 * A12   (L11 = unit-lower triangle of factored A11)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', TriMixedPrec(A.A11, 'L'), A.A12)

    # A21 <- L21 = A21 * U11^-1   (U11 = upper triangle of factored A11)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', TriMixedPrec(A.A11, 'U'), A.A21)

    # A22 <- A22 - L21 * U12.
    # If A21/A12 are dynamically quantized fp16 blocks, their stored payloads
    # represent (block / scale); the TRSM above is linear in B, so the scales
    # survive the solve and are folded into alpha here: -1 * s21 * s12.
    s21 = A.A21_scale !== nothing ? A.A21_scale : 1.0f0
    s12 = A.A12_scale !== nothing ? A.A12_scale : 1.0f0
    recgemm!(-Float64(s21 * s12), A.A21, A.A12, 1.0, A.A22)

    getrf_recursive!(A.A22)
end