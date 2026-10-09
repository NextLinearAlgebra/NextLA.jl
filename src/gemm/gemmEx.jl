export gemm, gemm!
export gemmEx!

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
products. Dense matrix products forward to `LinearAlgebra.mul!`, while
structured matrix types provide specialized methods.
"""
function gemm! end

gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; alpha::Number=true, beta::Number=false) =
    gemm!(C, A, B, alpha, beta)

function gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix,
               alpha::Number, beta::Number)
    return LinearAlgebra.mul!(C, A, B, alpha, beta)
end

"""
    gemm!(transA, transB, alpha, A, B, beta, C) -> C

The BLAS argument order: `C := alpha * op(A) * op(B) + beta * C`, where `op` is
the identity, `transpose` or `adjoint` for `'N'`, `'T'` or `'C'`.

vicki-development called GEMM this way through `src/wrappers.jl`, a CUDA/AMD
shim this repo does not take. Here it forwards to the output-first method, so
it runs on every backend and reaches cuBLAS/rocBLAS through `mul!` on GPU
arrays.
"""
function gemm!(transA::Char, transB::Char, alpha::Number, A::AbstractMatrix,
               B::AbstractMatrix, beta::Number, C::AbstractMatrix)
    return gemm!(C, _gemm_op(transA, A), _gemm_op(transB, B), alpha, beta)
end

@inline _gemm_op(t::Char, X) =
    t == 'N' ? X : t == 'T' ? transpose(X) : t == 'C' ? adjoint(X) :
    throw(ArgumentError("trans must be 'N', 'T' or 'C', got '$t'"))

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

# op(A) as a lazy wrapper. 'C' is the adjoint and differs from 'T' only for
# complex operands, but the two must not be collapsed: TLR passes
# adjoint_blas_char(T), which is 'C' precisely when that difference matters.
@inline _op(A::AbstractMatrix, t::Char) =
    t == 'N' ? A : (t == 'C' ? adjoint(A) : transpose(A))

# Whether BLAS can take this triple directly. Float16 is in NATIVE_GEMM_TYPES
# because a GPU can do it natively, but it is not a BlasFloat, so it has to be
# excluded here specifically.
@inline _cpu_blas_gemm(::Type{T}, ::Type{T}, ::Type{T}) where
    {T<:Union{Float32,Float64,ComplexF32,ComplexF64}} = true
@inline _cpu_blas_gemm(::Type, ::Type, ::Type) = false

"""
    gemmEx!(transA, transB, alpha, A, B, beta, C; compute_type=...)

Compute the matrix product

`C := alpha * op(A) * op(B) + beta * C`

and store the result in `C`.

`gemmEx!` is an advanced `NextLA` API for backends that support mixed-type GEMM
where the result storage may differ from the operand storage. It is available
unqualified (it is exported) and as `NextLA.gemmEx!`.

`op(A)` and `op(B)` are determined by `transA` and `transB`, which may be:
- `'N'`: no transpose
- `'T'`: transpose
- `'C'`: adjoint

## Notes
- The accumulation type can be selected explicitly with `compute_type`; by
  default [`default_compute_type`](@ref) chooses it from the storage types of
  `A`, `B` and `C`. `alpha` and `beta` take no part and are converted to it.
- Mixed storage and compute types are a `CUDA` and `AMDGPU` feature. The `CPU`
  accepts any combination, accumulating in `compute_type`. `oneAPI` and `Metal`
  have no `gemmEx!` and throw for every signature, same-type ones included.
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
    get_backend(C) isa KernelAbstractions.CPU ||
        throw(ArgumentError("NextLA.gemmEx! has no method for $(typeof(get_backend(C)))"))
    _gemm_dims(transA, transB, A, B, C)

    TA, TB, TC = eltype(A), eltype(B), eltype(C)

    # Same shape as the AMDGPU method: hand a triple BLAS can take straight to
    # BLAS, and only fall back to a wide accumulation for the rest.
    if _cpu_blas_gemm(TA, TB, TC) && compute_type == TC && !iszero(alpha)
        return BLAS.gemm!(transA, transB, TC(alpha), A, B, TC(beta), C)
    end

    # alpha and beta live in compute_type, not in eltype(C) -- that is what
    # cuBLAS and rocBLAS do (CuRef{scalar_type}, Ref{compute_type}), and a
    # Float16 alpha would otherwise round the scale factor before it is used.
    #
    # recgemm! and recsyrk! ask for Float32 over Float16 storage. Accumulating
    # wider than the storage is exactly right; the result is rounded once, on
    # store.
    W = compute_type
    a, b = W(alpha), W(beta)

    # alpha == 0 means A and B are not read at all. BLAS should guarantee this,
    # but some OpenBLAS kernels read A anyway, so it is always handled here --
    # otherwise an Inf in an operand, which a Float16 overflow produces readily,
    # turns 0 * Inf into a NaN that poisons a result the caller never asked to
    # depend on A or B. It also skips the product entirely, which is the whole
    # cost of the call.
    if iszero(a)
        if iszero(b)
            fill!(C, zero(TC))
        else
            @inbounds @simd for i in eachindex(C)
                C[i] = TC(b * W(C[i]))
            end
        end
        return C
    end

    Aw = TA === W ? A : W.(A)
    Bw = TB === W ? B : W.(B)
    acc = _op(Aw, transA) * _op(Bw, transB)
    if iszero(b)
        @inbounds @simd for i in eachindex(C)
            C[i] = TC(a * acc[i])
        end
    else
        @inbounds @simd for i in eachindex(C)
            C[i] = TC(a * acc[i] + b * W(C[i]))
        end
    end
    return C
end
