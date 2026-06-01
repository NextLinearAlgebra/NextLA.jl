export syrk!

@inline function _syrk_dims(uplo::Char, trans::Char, A::AbstractMatrix, C::AbstractMatrix)
    n = size(A, trans == 'N' ? 1 : 2)
    k = size(A, trans == 'N' ? 2 : 1)
    size(C, 1) == size(C, 2) || throw(DimensionMismatch("syrk!: C must be square"))
    size(C, 1) == n || throw(DimensionMismatch("syrk!: A and C have incompatible dimensions"))
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("Unsupported uplo flag `$uplo`"))
    return n, k
end

"""
    syrk!(uplo, trans, alpha, A, beta, C)

Perform a symmetric rank-k update and store the result in `C`:

`C := alpha * op(A) * op(A)' + beta * C`

where `op(A)` is determined by `trans`.

`uplo` selects which triangle of `C` is updated:
- `'U'`: upper triangle
- `'L'`: lower triangle

`trans` may be:
- `'N'`: no transpose
- `'T'`: transpose
- `'C'`: adjoint

## Notes
- On the CPU, `syrk!` is a thin wrapper around `LinearAlgebra.BLAS.syrk!`.
- GPU backends may dispatch to native SYRK kernels when available.
"""
function syrk!(uplo::Char,
               trans::Char,
               alpha,
               A::AbstractMatrix,
               beta,
               C::AbstractMatrix)
    _syrk_dims(uplo, trans, A, C)
    return BLAS.syrk!(uplo, trans, eltype(C)(alpha), A, eltype(C)(beta), C)
end
