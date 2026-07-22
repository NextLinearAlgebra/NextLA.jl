using LinearAlgebra
using CUDA
using CUDA.CUSOLVER
include("fullmixedprec.jl")
include("recmixedprectri.jl")
include("rectrxm.jl")
include("matmul.jl")
include("wrappers.jl")

# ---------------------------------------------------------------------------
# CUSOLVER non-pivoting LU base case
# ---------------------------------------------------------------------------

"""
    getrf_npvt!(A::StridedCuMatrix)

Performs an in-place, non-pivoting LU factorization of `A` using CUSOLVER
(`cusolverDn<t>getrf`). Passing a null device pivot array (`CU_NULL`) instructs
CUSOLVER to skip partial pivoting, which is required here since the recursive
blocked algorithm operates on sub-blocks that cannot exchange rows globally.

On exit, the strictly lower triangle of `A` holds the unit-lower factor `L`
(the unit diagonal is implicit) and the upper triangle (including the diagonal)
holds `U`.
"""
function getrf_npvt! end

for (bname, fname, elty) in
    ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
     (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64))
    @eval begin
        function getrf_npvt!(A::StridedCuMatrix{$elty})
            m, n = size(A)
            lda  = max(1, stride(A, 2))
            dh   = CUSOLVER.dense_handle()

            lwork = Ref{Cint}(0)
            CUSOLVER.$bname(dh, m, n, A, lda, lwork)
            work = CuArray{$elty}(undef, lwork[])
            info = CuArray{Cint}(undef, 1)

            # devIpiv == CU_NULL  =>  LU without partial pivoting
            CUSOLVER.$fname(dh, m, n, A, lda, work, CU_NULL, info)
            return A
        end
    end
end

# Float16 blocks have no CUSOLVER getrf kernel: round-trip through Float32.
function getrf_npvt!(A::StridedCuMatrix{Float16})
    A32 = Float32.(A)
    getrf_npvt!(A32)
    copy!(A, A32)
    return A
end

# ---------------------------------------------------------------------------
# Triangular reinterpretation of packed LU factors
# ---------------------------------------------------------------------------

"""
    TriMixedPrec(A::FullMixedPrec, uplo::Char)

Zero-copy reinterpretation of a `FullMixedPrec` matrix holding packed LU factors
as a `TriMixedPrec` triangular structure.

For `uplo == 'L'` the resulting structure aliases the `A21` off-diagonal blocks
(the unit-lower factor `L`); for `uplo == 'U'` it aliases the `A12` blocks (the
upper factor `U`). Base-case blocks are shared directly, so the triangular
routines simply reference the appropriate triangle of the packed storage — no
data is copied and quantization scales are carried through unchanged.
"""
function TriMixedPrec(A::FullMixedPrec{T_Base}, uplo::Char) where {T_Base}
    if A.BaseCase !== nothing
        return TriMixedPrec{T_Base}(
            nothing, nothing, nothing,
            nothing, A.base_scale, A.BaseCase,
            uplo, A.sz
        )
    end

    OffDiag   = (uplo == 'L') ? A.A21       : A.A12
    off_scale = (uplo == 'L') ? A.A21_scale : A.A12_scale

    return TriMixedPrec{T_Base}(
        TriMixedPrec(A.A11, uplo),
        TriMixedPrec(A.A22, uplo),
        OffDiag,
        off_scale,
        nothing,
        nothing,
        uplo,
        A.sz
    )
end

# ---------------------------------------------------------------------------
# Dense (AbstractMatrix) recursive LU — base-case framework
# ---------------------------------------------------------------------------

"""
    getrf_recursive!(A, block_size)

Performs an in-place, non-pivoting, nested recursive block LU factorization on
the dense matrix `A`. The recursion dynamically splits the matrix until the
sub-block size is less than or equal to `block_size`, at which point it falls
back to the CUSOLVER non-pivoting `getrf` base case.

Block updates use the provided mixed-precision subroutines: panel solves go
through `unified_rectrxm!` and the Schur complement goes through `recgemm!`,
so the precision hierarchy is preserved end-to-end.
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        getrf_npvt!(A)
        return
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    A11 = @view A[1:mid,     1:mid]
    A12 = @view A[1:mid,     mid+1:end]
    A21 = @view A[mid+1:end, 1:mid]
    A22 = @view A[mid+1:end, mid+1:end]

    # Step 1: A11 -> L11 * U11 (packed in place)
    getrf_recursive!(A11, block_size)

    # Step 2: A12 <- L11^-1 * A12   (L11 is unit lower triangular)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)

    # Step 3: A21 <- A21 * U11^-1   (U11 is non-unit upper triangular)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)

    # Step 4: A22 <- A22 - A21 * A12  (Schur complement)
    recgemm!(-1.0, A21, A12, 1.0, A22)

    # Step 5: A22 -> L22 * U22
    getrf_recursive!(A22, block_size)
end

# ---------------------------------------------------------------------------
# FullMixedPrec recursive LU
# ---------------------------------------------------------------------------

"""
    getrf_recursive!(A::FullMixedPrec)

Performs an in-place, non-pivoting, nested recursive block LU factorization on
a full mixed-precision matrix structure `A`. The recursion follows the block
hierarchy of the `FullMixedPrec` structure itself: diagonal blocks `A11` / `A22`
recurse, off-diagonal blocks `A12` / `A21` are updated by triangular solves,
and the trailing block receives the Schur complement update. The base case
falls back to the dense recursive routine backed by CUSOLVER.

After completion, `A.A21` holds `L21`, `A.A12` holds `U12`, and the diagonal
blocks hold the packed unit-lower / upper LU factors of their respective
sub-problems.

Notes:
- Panel solves reference the freshly factored `A11` through the zero-copy
  `TriMixedPrec(A11, uplo)` views, so `L11` (unit-lower, `A21` side) and `U11`
  (upper, `A12` side) are consumed directly from the packed storage.
- Because the triangular solves are linear maps, solving in place on quantized
  `Float16` storage preserves the stored-value/scale relationship: if the true
  block is `s * stored`, the solved block is again `s * stored_new`. The Schur
  complement therefore folds both off-diagonal scales into `alpha`.
- As with `recgemm!`'s hierarchical path, conformal square sub-blocks are
  assumed, so `A` should have power-of-two dimensions.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Step 1: A11 -> L11 * U11
    getrf_recursive!(A.A11)

    # Step 2: A12 <- L11^-1 * A12   (unit lower triangle of the packed A11)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', TriMixedPrec(A.A11, 'L'), A.A12)

    # Step 3: A21 <- A21 * U11^-1   (non-unit upper triangle of the packed A11)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', TriMixedPrec(A.A11, 'U'), A.A21)

    # Step 4: A22 <- A22 - A21 * A12 (Schur complement), folding the
    # dynamic Float16 quantization scales of both off-diagonal panels
    # into the scalar multiplier.
    s21 = A.A21_scale === nothing ? 1.0f0 : A.A21_scale
    s12 = A.A12_scale === nothing ? 1.0f0 : A.A12_scale
    recgemm!(-Float64(s21 * s12), A.A21, A.A12, 1.0, A.A22)

    # Step 5: A22 -> L22 * U22
    getrf_recursive!(A.A22)
end