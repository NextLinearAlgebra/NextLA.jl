# Recursive unpivoted LU, splitting into a 2x2 block scheme and driving the
# updates through NextLA's own triangular solve and recursive GEMM.
#
# Source: vicki-development @ 694d019, by Vicki <vickicar@mit.edu>. Verbatim
# apart from the leaf.
#
# The branch called CUSOLVER.getrf! at the leaf behind a top-level `using
# CUDA`, which is neither portable nor needed, and that getrf! pivots: its
# pivots were discarded inside a recursion that is otherwise unpivoted, which
# gives a wrong factorisation whenever a leaf swaps rows (0.41 relative error on
# a 4x4 matrix that has an unpivoted LU). The leaf here is unpivoted too:
# LinearAlgebra.lu!(A, NoPivot()) on the CPU, and per backend where that falls
# back to scalar indexing -- ext/cuda/lu.jl calls cuSOLVER's getrf with a null
# pivot array, ext/amdgpu/lu.jl rocSOLVER's getrf_npvt. A zero pivot throws
# ZeroPivotException. The matrix must still need no pivoting -- diagonally
# dominant input, for instance; a small pivot is not checked.

# The leaf factorisation, unpivoted. ext/cuda/lu.jl and ext/amdgpu/lu.jl add the
# device methods. Returns A.
_lu_leaf!(A::AbstractMatrix) = (LinearAlgebra.lu!(A, LinearAlgebra.NoPivot()); A)

# The default block_size of lu_recursive! and lu_recursive_nopiv!, and of the
# mixed-precision lu_recursive!: blocks of at most this many rows go to the
# unpivoted leaf factorisation. vicki-development's value; not tuned.
const LU_BLOCK_SIZE = 2048

export lu_recursive!

"""
    lu_recursive!(A::AbstractMatrix, block_size::Integer=LU_BLOCK_SIZE) -> A

Recursive unpivoted LU of a square matrix, in place: `A = L * U` with `L` unit lower
triangular and `U` upper triangular, packed into `A`. The recursion splits `n` at the largest
power of two below it (at `n ÷ 2` when `n` is one) and factors blocks of at most `block_size`
rows without pivoting, keeping the bulk of operations in `A`'s precision.
[`lu_recursive_nopiv!`](@ref) takes a matrix of any shape.

!!! warning
    Unpivoted, leaves included, so the matrix must need no pivoting -- a diagonally
    dominant one, for instance. A zero pivot throws `ZeroPivotException`; a small one is
    not checked.
"""
function lu_recursive!(A::AbstractMatrix, block_size::Integer=LU_BLOCK_SIZE)
    block_size >= 1 || throw(ArgumentError("block_size must be at least 1"))
    size(A, 1) == size(A, 2) ||
        throw(DimensionMismatch("lu_recursive! takes a square matrix, got $(size(A)); use lu_recursive_nopiv!"))
    n = size(A, 1)
    T = eltype(A)

    if n <= block_size
        if T == Float16
            A_f32 = Float32.(A)
            _lu_leaf!(A_f32)
            A .= Float16.(A_f32)
        else
            _lu_leaf!(A)
        end
        return A
    end

    n1 = _rec_split(n)

    A11 = @view A[1:n1, 1:n1]
    A12 = @view A[1:n1, n1+1:end]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    lu_recursive!(A11, block_size)

    unified_rectrxm!('L', 'L', 'N', 'U', one(T), 'S', A11, A12)

    unified_rectrxm!('R', 'U', 'N', 'N', one(T), 'S', A11, A21)

    recgemm!(-one(T), A21, A12, one(T), A22)

    lu_recursive!(A22, block_size)
    return A
end

# Rectangular recursive LU, splitting by columns rather than into square blocks.
#
# Source: vicki-development, src/recursive_nopivlu.jl. Its 34-line function is
# the whole of the merge: the rest of that file is a top-level script that
# allocates a 20480x20480 matrix -- about 3.3 GB -- and factors it at module
# load time, which is why the file could not be included as it stood.
#
# The algorithm is the same right-looking recursion lu_recursive! performs, and
# the difference is the shape it accepts. lu_recursive! takes its size from
# size(A, 1) and cuts square blocks out of the corner, so it needs a square
# matrix; this one halves the columns and carries the full height down the left
# panel, which is the form a panel factorisation wants. For a square matrix the
# two compute the same factors by the same route.
#
# Not verbatim. The branch called BLAS.trsm! and BLAS.gemm! directly, which is
# CPU-only and was the reason a portable replacement was wanted at all. Those
# two calls become unified_rectrxm! and recgemm!, exactly as reclu.jl's own
# driver uses them, so the solves and updates run on every backend. The leaf is
# lu_recursive!'s _lu_leaf!, so it reaches rocSOLVER on AMDGPU as well.
#
# It shares lu_recursive!'s unpivoted leaf and therefore its standing
# assumption: the matrix must need no pivoting, which nothing here checks beyond
# a zero pivot -- use diagonally dominant input.
export lu_recursive_nopiv!

"""
    lu_recursive_nopiv!(A::AbstractMatrix, block_size::Integer=LU_BLOCK_SIZE) -> A

Recursive unpivoted LU of a matrix of any shape, in place: `A = L * U` with `L`
unit lower trapezoidal and `U` upper trapezoidal, packed into `A`.

Halves the columns, factors the left panel over the full height, solves for the
`U` block to its right and updates the trailing block before recurring into it.
[`lu_recursive!`](@ref) is the square-only counterpart; prefer it when `A` is
square, and this when it is not.

!!! warning
    Unpivoted, leaves included, so the matrix must need no pivoting -- a
    diagonally dominant one, for instance. A zero pivot throws
    `ZeroPivotException`; a small one is not checked.
"""
function lu_recursive_nopiv!(A::AbstractMatrix, block_size::Integer=LU_BLOCK_SIZE)
    block_size >= 1 || throw(ArgumentError("block_size must be at least 1"))
    m, n = size(A)
    T = eltype(A)

    # The leaf handles a rectangular block too. Float16 goes through Float32
    # for accuracy, not for want of a method: LinearAlgebra.lu! does factor a
    # Float16 matrix, through its generic path, but accumulates in Float16 as
    # it goes. Measured on diagonally dominant input, relative L*U error
    # against the original is 1.7e-3 at n=32 and 1.6e-3 at n=64 that way,
    # against 2.5e-4 and 2.8e-4 through Float32 -- the same treatment the TRSM
    # base case and GEMM_ADD!/GEMM_SUB! give half precision. lu_recursive!'s
    # leaf does likewise.
    if min(m, n) <= block_size
        if T == Float16
            A_f32 = Float32.(A)
            _lu_leaf!(A_f32)
            A .= Float16.(A_f32)
        else
            _lu_leaf!(A)
        end
        return A
    end

    # Halve the shorter side, not the columns. The branch splits at n ÷ 2, which
    # is the same thing whenever n <= m but walks off the top of a wide matrix:
    # at 48x100 it asks for A[1:50, 1:50] and raises a BoundsError. LAPACK's
    # recursive getrf splits at min(m, n) ÷ 2 for this reason, and that is what
    # is used here. For a square or tall matrix it is the branch's own split.
    n1 = min(m, n) ÷ 2

    # The left panel, full height: [A11; A21] becomes L11 (unit) over L21, with
    # U11 in A11's upper triangle.
    lu_recursive_nopiv!(view(A, :, 1:n1), block_size)

    A11 = view(A, 1:n1, 1:n1)
    A12 = view(A, 1:n1, (n1 + 1):n)
    A21 = view(A, (n1 + 1):m, 1:n1)
    A22 = view(A, (n1 + 1):m, (n1 + 1):n)

    # U12 = L11 \ A12, with L11 unit lower triangular.
    unified_rectrxm!('L', 'L', 'N', 'U', one(T), 'S', A11, A12)

    # A22 -= L21 * U12, then the same recursion on what is left.
    recgemm!(-one(T), A21, A12, one(T), A22)
    lu_recursive_nopiv!(A22, block_size)

    return A
end
