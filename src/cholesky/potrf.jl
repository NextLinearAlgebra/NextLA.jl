export potrf!

@inline function _potrf_dims(uplo::Char, A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("potrf!: A must be square"))
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("Unsupported uplo flag `$uplo`"))
    return size(A, 1)
end

"""
    potrf!(uplo, A) -> (A, info)

Compute the Cholesky factorization of the symmetric positive definite matrix
`A`, in place, and return `(A, info)`.

`uplo` selects which triangle is read and overwritten with the factor:
- `'U'`: `A = Uᵀ * U`, the upper triangle
- `'L'`: `A = L * Lᵀ`, the lower triangle

The opposite triangle is neither read nor written, so it keeps whatever it held
on entry. `info` follows LAPACK: `0` on success, and `k > 0` when the leading
minor of order `k` is not positive definite, at which point the factorization
is incomplete.

    potrf!(A) -> (A, info)

Convenience form, equivalent to `potrf!('L', A)`.

## Notes
- On the CPU, `potrf!` is a thin wrapper around `LinearAlgebra.LAPACK.potrf!`.
- GPU backends dispatch to native Cholesky routines when available; see
  `ext/{cuda,amdgpu}/potrf.jl`. There is no generic device fallback, so a
  backend without a wrapper raises rather than silently copying to the host.
- This is the single-matrix entry point. For a batch of independent matrices
  use [`potrf_batched!`](@ref), which has its own backend wrappers.
- Unlike `LinearAlgebra.cholesky`, a non-positive-definite input is reported
  through `info` rather than raising `PosDefException`.
"""
function potrf!(uplo::Char, A::AbstractMatrix)
    backend = KernelAbstractions.get_backend(A)
    backend isa KernelAbstractions.CPU || throw(ArgumentError("NextLA.potrf! has no generic non-CPU implementation; use a backend wrapper"))
    _potrf_dims(uplo, A)
    return LAPACK.potrf!(uplo, A)
end

# Half precision. No LAPACK, cuSOLVER or rocSOLVER routine takes Float16, so a
# Float16 matrix is factored as a Float32 copy on the same device and rounded
# back once, on store. The opposite triangle round-trips Float16 -> Float32 ->
# Float16 exactly, so it is still untouched. It is the treatment
# vicki-development's wrappers.jl gave potrf!(A), and the one the TRSM base case
# gives Float16; without it the recursive Cholesky's Float16 leaves had no
# method at all.
function potrf!(uplo::Char, A::AbstractMatrix{Float16})
    _potrf_dims(uplo, A)
    A32, info = potrf!(uplo, Float32.(A))
    A .= Float16.(A32)          # broadcast, so the copy back stays on the device
    return A, info
end

potrf!(A::AbstractMatrix) = potrf!('L', A)
