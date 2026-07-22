using LinearAlgebra
using StochasticRounding
include("fullmixedprec.jl")
include("recmixedprectri.jl")
include("trsm.jl")
include("trmm.jl")
include("matmul.jl")
include("rectrxm.jl")
include("recgemm.jl")
include("getrf.jl")
include("wrappers.jl")

"""
    getrf_recursive!(A, block_size)

Performs a non-pivoting, in-place, nested recursive block LU factorization on
the dense GPU matrix `A`. The recursion splits the matrix until the sub-block
size is less than or equal to `block_size`, at which point `getrf!`
(cusolverDnSgetrf / cusolverDnDgetrf) is called as the CUSOLVER base case.

The five-step block LU for A = [A11 A12; A21 A22]:

  1. A11 → L11·U11           `getrf_recursive!(A11, block_size)`
  2. A12 ← L11⁻¹·A12        TRSM left-lower-unit
  3. A21 ← A21·U11⁻¹        TRSM right-upper-non-unit
  4. A22 ← A22 − A21·A12    Schur complement via `recgemm!`
  5. A22 → L22·U22           `getrf_recursive!(A22, block_size)`

TRSM dispatches through `unified_rectrxm!` for Float16 blocks and through the
direct `trsm!` wrapper otherwise, matching the style of `potrf_recursive!`.
The Schur complement dispatches through `recgemm!` for Float16 operands and
through `gemm!` otherwise.
"""
function getrf_recursive!(A, block_size)
    n = size(A, 1)
    if n <= block_size
        getrf!(A)
        return
    end

    n1  = 2^floor(Int, log2(n)) ÷ 2

    A11 = @view A[1:n1,     1:n1]
    A12 = @view A[1:n1,     n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    # Step 1 – factor A11
    getrf_recursive!(A11, block_size)

    # Step 2 – A12 ← L11⁻¹ · A12  (left, lower-triangular, unit diagonal)
    if eltype(A11) == Float16
        unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
    else
        trsm!('L', 'L', 'N', 'U', 1.0, A11, A12)
    end

    # Step 3 – A21 ← A21 · U11⁻¹  (right, upper-triangular, non-unit diagonal)
    if eltype(A11) == Float16
        unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('R', 'U', 'N', 'N', 1.0, A11, A21)
    end

    # Step 4 – Schur complement  A22 ← A22 − A21 · A12
    if eltype(A21) == Float16 || eltype(A12) == Float16
        recgemm!(-1.0, A21, A12, 1.0, A22)
    else
        gemm!('N', 'N', eltype(A22)(-1.0), A21, A12, eltype(A22)(1.0), A22)
    end

    # Step 5 – factor Schur complement
    getrf_recursive!(A22, block_size)
end

"""
    getrf_recursive!(A::FullMixedPrec)

Performs a non-pivoting, in-place, nested recursive block LU factorization on
a `FullMixedPrec` mixed-precision matrix structure.

The recursion descends the `FullMixedPrec` block hierarchy until reaching a
`BaseCase` dense block, at which point it delegates to `getrf_recursive!(A.BaseCase, 4096)`
for a final level of dense recursion before the CUSOLVER leaf call.

The five-step block LU mirrors the dense overload exactly, with all panel
operations expressed through the provided mixed-precision infrastructure:

  1. `getrf_recursive!(A.A11)`
       — recurse on the top-left block (or BaseCase → CUSOLVER at the leaf).

  2. `unified_rectrxm!('L','L','N','U', 1.0,'S', A.A11, A.A12)`
       — solve L11·U12 = A12 in-place, writing U12 into A.A12.
       `unified_rec_mixed` uses `hasproperty(A.A11, :A21)` → true for
       `FullMixedPrec`, so it picks A.A11.A21 as the lower off-diagonal panel,
       correctly routing the L11 sub-structure into the triangular solve.

  3. `unified_rectrxm!('R','U','N','N', 1.0,'S', A.A11, A.A21)`
       — solve L21·U11 = A21 (i.e. X·U11 = A21) in-place, writing L21 into
       A.A21. With uplo='U', `unified_rec_mixed` picks A.A11.A12 as the upper
       off-diagonal panel, routing U11 correctly.

  4. `recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)`
       — Schur complement update: A22 ← A22 − L21·U12, descending into the
       `FullMixedPrec` structure of A.A22 at each recursive level.

  5. `getrf_recursive!(A.A22)`
       — recurse on the Schur complement.

No auxiliary triangular extraction functions are required: `unified_rec_mixed`
already dispatches on `hasproperty(A, :A21)` and `hasproperty(A, :A21_scale)`,
so passing `A.A11` (a `FullMixedPrec`) directly as the triangular operand is
both correct and zero-overhead.
"""
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, 4096)
        return
    end

    # Step 1 – factor A11
    getrf_recursive!(A.A11)

    # Step 2 – A12 ← L11⁻¹ · A12  (left, lower-triangular, unit diagonal)
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A.A11, A.A12)

    # Step 3 – A21 ← A21 · U11⁻¹  (right, upper-triangular, non-unit diagonal)
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A.A11, A.A21)

    # Step 4 – Schur complement  A22 ← A22 − A21 · A12
    recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)

    # Step 5 – factor Schur complement
    getrf_recursive!(A.A22)
end