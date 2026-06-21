export gemm_batched!

"""
    gemm_batched!(transA, transB, alpha, A, B, beta, C)

Compute a batched matrix product and store the result in `C`.

Two storage forms are supported:
- `AbstractArray{T,3}` for strided batches
- `AbstractVector` of matrices for pointer batches

Each batch entry applies

`C[i] := alpha * op(A[i]) * op(B[i]) + beta * C[i]`

where `op(A)` and `op(B)` are determined by `transA` and `transB`.

## Notes
- On the CPU, `gemm_batched!` falls back to a loop of standard GEMMs.
- GPU backends may dispatch to native pointer-batched or strided-batched GEMM
  kernels when available.
"""
function gemm_batched!(transA::Char,
                       transB::Char,
                       alpha,
                       A::AbstractArray{<:Any, 3},
                       B::AbstractArray{<:Any, 3},
                       beta,
                       C::AbstractArray{<:Any, 3})
    batchA = size(A, 3)
    batchB = size(B, 3)
    batchC = size(C, 3)
    (batchA == batchC || batchA == 1) || throw(DimensionMismatch("gemm_batched!: A and C batch sizes are incompatible"))
    (batchB == batchC || batchB == 1) || throw(DimensionMismatch("gemm_batched!: B and C batch sizes are incompatible"))

    for i in 1:batchC
        Ai = @view A[:, :, batchA == 1 ? 1 : i]
        Bi = @view B[:, :, batchB == 1 ? 1 : i]
        Ci = @view C[:, :, i]
        BLAS.gemm!(transA, transB, eltype(Ci)(alpha), Ai, Bi, eltype(Ci)(beta), Ci)
    end
    return C
end

"""
    gemm_batched_ptrs!(transA, transB, alpha, Aptrs, Aref, Bptrs, Bref, beta, Cptrs, Cref, batch_count)

Internal batched GEMM helper that accepts device arrays of matrix base pointers.
`Aref`, `Bref`, and `Cref` are representative matrices used only to infer the
shared GEMM dimensions and leading dimensions for the pointed-to batches.
"""
function gemm_batched_ptrs!(transA::Char,
                            transB::Char,
                            alpha,
                            Aptrs,
                            Aref::AbstractMatrix,
                            Bptrs,
                            Bref::AbstractMatrix,
                            beta,
                            Cptrs,
                            Cref::AbstractMatrix,
                            batch_count::Integer)
    throw(ArgumentError("NextLA.gemm_batched_ptrs! is supported only on CUDA and AMDGPU"))
end

function gemm_batched!(transA::Char,
                       transB::Char,
                       alpha,
                       A::AbstractVector{<:AbstractArray{<:Any, 2}},
                       B::AbstractVector{<:AbstractArray{<:Any, 2}},
                       beta,
                       C::AbstractVector{<:AbstractArray{<:Any, 2}})
    length(A) == length(B) == length(C) || throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    for i in eachindex(A, B, C)
        BLAS.gemm!(transA, transB, eltype(C[i])(alpha), A[i], B[i], eltype(C[i])(beta), C[i])
    end
    return C
end

"""
    gemmEx_batched!(transA, transB, alpha, A, B, beta, C; compute_type=default_compute_type(alpha, A, B, beta, C))

Compute a batched matrix product with explicit control over the compute type.

Each batch entry applies

`C[i] := alpha * op(A[i]) * op(B[i]) + beta * C[i]`

where `op(A)` and `op(B)` are determined by `transA` and `transB`.

`gemmEx_batched!` is an advanced `NextLA` API for backends that support
explicit compute-type batched GEMM. It is available as
`NextLA.gemmEx_batched!` and is not part of the primary exported surface.

Use `gemmEx_batched!` when you want to choose the GEMM compute type explicitly
for a batched operation.

## Notes
- If `compute_type` matches the backend default, `gemmEx_batched!` falls back
  to `gemm_batched!`.
- `gemmEx_batched!` is currently implemented for CUDA and AMDGPU. Other
  backends, including CPU, report that it is unsupported.
- If a backend does not provide a dedicated Ex batched GEMM path, it may fall
  back to the corresponding standard batched GEMM implementation.
- If a backend does not support a requested storage and compute type
  combination, backend-specific errors may be raised.
- Algorithm selection and backend-specific math modes are out of scope for this
  API.
"""
function gemmEx_batched!(transA::Char,
                         transB::Char,
                         alpha,
                         A::AbstractArray{<:Any, 3},
                         B::AbstractArray{<:Any, 3},
                         beta,
                         C::AbstractArray{<:Any, 3};
                         compute_type::Type = default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is supported only on CUDA and AMDGPU"))
end

function gemmEx_batched!(transA::Char,
                         transB::Char,
                         alpha,
                         A::AbstractVector{<:AbstractArray{<:Any, 2}},
                         B::AbstractVector{<:AbstractArray{<:Any, 2}},
                         beta,
                         C::AbstractVector{<:AbstractArray{<:Any, 2}};
                         compute_type::Type = default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is supported only on CUDA and AMDGPU"))
end
