# The default block_size of potrf_recursive!, dense and mixed-precision: blocks
# of at most this many rows go to potrf!. vicki-development's leaf size; not tuned.
const POTRF_BLOCK_SIZE = 4096

export potrf_recursive!

"""
    potrf_recursive!(A, block_size=POTRF_BLOCK_SIZE; uplo='L')

Perform an in-place, nested recursive Cholesky factorization of the symmetric
positive definite matrix `A`: `A = L * Lᵀ` with `L` written into the lower triangle
for `uplo = 'L'`, or `A = Uᵀ * U` with `U` in the upper triangle for `'U'`.

The recursion splits `A` into a 2x2 block scheme and stops once a sub-block is
`block_size` or smaller, at which point it falls back to [`potrf!`](@ref).

The panel solve always goes through [`unified_rectrxm!`](@ref), main's portable
triangular solve. The branch called a vendor `trsm!` from `src/wrappers.jl` for
everything but `Float16`; that file is not merged, and `BLAS.trsm!` would pin
this routine to the CPU, so the recursive path is used throughout. The rank-k
update still splits: `syrk!` reaches BLAS or the vendor library, but has no
`Float16` method anywhere, so half precision takes [`recsyrk!`](@ref).

The other triangle is not referenced and is left as it was on entry. A matrix that is not
positive definite throws `PosDefException(k)`, `k` the order of the first leading minor that
is not, as `LinearAlgebra.cholesky!` does; `A` is then partly factored. The recursion splits
`n` at the largest power of two below it (at `n ÷ 2` when `n` is one).
"""
function potrf_recursive!(A::AbstractMatrix, block_size::Integer=POTRF_BLOCK_SIZE; uplo::Char='L')
    uplo in ('L', 'U') || throw(ArgumentError("uplo must be 'L' or 'U', got '$uplo'"))
    block_size >= 1 || throw(ArgumentError("block_size must be at least 1"))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("potrf_recursive!: A must be square, got $(size(A))"))
    return _potrf_recursive!(A, block_size, uplo, 0)
end

# offset is where A starts in the matrix the caller passed, so a leaf's info
# becomes the order of the leading minor in the whole matrix.
function _potrf_recursive!(A, block_size, uplo::Char, offset::Int)
    n = size(A, 1)

    if n <= block_size
        _, info = potrf!(uplo, A)
        info > 0 && throw(LinearAlgebra.PosDefException(offset + Int(info)))
        return A
    end

    n1 = _rec_split(n)

    A11 = @view A[1:n1, 1:n1]
    A22 = @view A[(n1 + 1):end, (n1 + 1):end]

    _potrf_recursive!(A11, block_size, uplo, offset)

    if uplo == 'L'
        # L21 = A21 * L11⁻ᵀ, then A22 -= L21 * L21ᵀ
        A21 = @view A[(n1 + 1):end, 1:n1]
        unified_rectrxm!('R', 'L', 'T', 'N', one(eltype(A11)), 'S', A11, A21)
        if eltype(A21) == Float16
            recsyrk!(-1.0, A21, 1.0, A22)
        else
            syrk!('L', 'N', -one(eltype(A22)), A21, one(eltype(A22)), A22)
        end
    else
        # U12 = U11⁻ᵀ * A12, then A22 -= U12ᵀ * U12
        A12 = @view A[1:n1, (n1 + 1):end]
        unified_rectrxm!('L', 'U', 'T', 'N', one(eltype(A11)), 'S', A11, A12)
        if eltype(A12) == Float16
            recsyrk!(-1.0, copy(transpose(A12)), 1.0, A22; uplo='U')
        else
            syrk!('U', 'T', -one(eltype(A22)), A12, one(eltype(A22)), A22)
        end
    end

    _potrf_recursive!(A22, block_size, uplo, offset + n1)
    return A
end
