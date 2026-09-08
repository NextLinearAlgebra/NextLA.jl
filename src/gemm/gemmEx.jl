export gemm, gemm!

"""
    gemm(A, B; kwargs...)

Allocation-returning matrix product. Structured matrix formats specialize this
generic when the result's storage layout depends on ranks discovered at runtime.
"""
function gemm end

"""
    gemm!(C, A, B; alpha=true, beta=false, kwargs...)

Compute `C := alpha * A * B + beta * C` in place.

`gemm!` is the common NextLA dispatch point for dense and structured matrix
products. Dense matrix products forward to [`LinearAlgebra.mul!`](@ref), while
structured matrix types provide specialized methods.
"""
function gemm! end

function gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
               alpha::Number=true, beta::Number=false)
    return LinearAlgebra.mul!(C, A, B, alpha, beta)
end

function gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix,
               alpha::Number, beta::Number)
    return LinearAlgebra.mul!(C, A, B, alpha, beta)
end

@inline function _gemm_dims(transA::Char,
                            transB::Char,
                            A::AbstractMatrix,
                            B::AbstractMatrix,
                            C::AbstractMatrix)
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch("A has dimension $(size(A)), B has dimension $(size(B)) and C has dimension $(size(C))"))
    end
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldc = max(1, stride(C, 2))
    return m, n, k, lda, ldb, ldc
end

"""
    gemmEx!(transA, transB, alpha, A, B, beta, C; compute_type=...)

Compute the matrix product

`C := alpha * op(A) * op(B) + beta * C`

and store the result in `C`.

`gemmEx!` is an advanced `NextLA` API for backends that support mixed-type GEMM
where the result storage may differ from the operand storage. It is available
as `NextLA.gemmEx!` and is not part of the primary exported surface.

`op(A)` and `op(B)` are determined by `transA` and `transB`, which may be:
- `'N'`: no transpose
- `'T'`: transpose
- `'C'`: adjoint

## Notes
- The accumulation type can be selected explicitly with `compute_type`; by
  default it is inferred from `alpha`, `beta`, and the operand/result element
  types using [`default_compute_type`](@ref).
- `gemmEx!` is currently implemented only for `CUDA` and `AMDGPU` backends.
- If a backend does not support a requested storage and compute type
  combination, backend-specific errors may be raised.
- Algorithm selection and backend-specific math modes are out of scope for this
  API.
"""
function gemmEx!(transA::Char,
                 transB::Char,
                 alpha,
                 A::AbstractMatrix,
                 B::AbstractMatrix,
                 beta,
                 C::AbstractMatrix;
                 compute_type::Type = default_compute_type(alpha, A, B, beta, C))
    _check_compute_type(compute_type)
    throw(ArgumentError("NextLA.gemmEx! is supported only on CUDA and AMDGPU"))
end
