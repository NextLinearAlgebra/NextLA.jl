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
    gemmEx!(transA, transB, alpha, A, B, beta, C; compute_type=default_compute_type(alpha, A, B, beta, C))

Compute the matrix product

`C := alpha * op(A) * op(B) + beta * C`

and store the result in `C`.

`gemmEx!` is an advanced `NextLA` API for backends that support explicit
compute-type GEMM. It is available as `NextLA.gemmEx!` and is not part of the
primary exported surface.

Use `gemmEx!` when you want to choose the GEMM compute type explicitly.

`op(A)` and `op(B)` are determined by `transA` and `transB`, which may be:
- `'N'`: no transpose
- `'T'`: transpose
- `'C'`: adjoint

## Notes
- On GPU backends, if `compute_type` matches the backend default for the given
  inputs, `gemmEx!` falls back to the standard backend GEMM path.
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
    throw(ArgumentError("NextLA.gemmEx! is supported only on CUDA and AMDGPU"))
end
