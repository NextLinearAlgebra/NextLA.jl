export gemm_batched!

# --- shared batched-GEMM validation helpers ---------------------------------
#
# These are used by the CPU fallbacks below and by every GPU backend extension
# to avoid re-implementing the same batch-shape checks in each wrapper.

"""
    _check_batch_lengths(A, B, C) -> length(C)

Assert that vector-of-matrices batches `A`, `B`, `C` have matching lengths.
"""
@inline function _check_batch_lengths(A, B, C)
    length(A) == length(B) == length(C) ||
        throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    return length(C)
end

"""
    _check_batch_dims(A, B, C) -> (batchA, batchB, batchC)

Assert that strided 3-D batches broadcast consistently against `C`: each of `A`
and `B` must either match `C`'s batch count or have a batch count of 1.
"""
@inline function _check_batch_dims(A, B, C)
    batchA, batchB, batchC = size(A, 3), size(B, 3), size(C, 3)
    (batchA == batchC || batchA == 1) ||
        throw(DimensionMismatch("gemm_batched!: A and C batch sizes are incompatible"))
    (batchB == batchC || batchB == 1) ||
        throw(DimensionMismatch("gemm_batched!: B and C batch sizes are incompatible"))
    return batchA, batchB, batchC
end

"""
    _strided_batch_layout(transA, transB, A, B, C)
        -> (m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchC)

Validate batch shapes and compute the shared GEMM dimensions, leading
dimensions, and batch strides for a strided-batched call. A broadcast operand
(batch count 1) is given a batch stride of 0.
"""
@inline function _strided_batch_layout(transA::Char, transB::Char, A, B, C)
    batchA, batchB, batchC = _check_batch_dims(A, B, C)
    m, n, k, lda, ldb, ldc = _gemm_dims(
        transA, transB, @view(A[:, :, 1]), @view(B[:, :, 1]), @view(C[:, :, 1]),
    )
    strideA = batchA == 1 ? 0 : stride(A, 3)
    strideB = batchB == 1 ? 0 : stride(B, 3)
    strideC = stride(C, 3)
    return m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchC
end

"""
    _try_same_type_batched!(transA, transB, alpha, A, B, beta, C, compute_type) -> C or nothing

If the operands share an element type and `compute_type` is the inferred default,
dispatch to the plain `gemm_batched!` and return `C`. Otherwise return `nothing`,
letting the caller raise a backend-specific "mixed-type unsupported" error. Used
by backends without a native mixed-type batched GEMM primitive.
"""
@inline function _try_same_type_batched!(transA::Char, transB::Char, alpha, A, B, beta, C,
                                         compute_type::Type)
    if _batch_eltype(A) == _batch_eltype(B) == _batch_eltype(C) &&
       compute_type == default_compute_type(alpha, A, B, beta, C)
        return gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    return nothing
end

"""
    gemm_batched!(transA, transB, alpha, A, B, beta, C)

Compute a batched matrix product and store the result in `C`.

Three storage forms are supported:
- `AbstractArray{T,3}` for strided batches
- `AbstractVector` of matrices for pointer batches
- backend pointer arrays plus representative matrices for explicit pointer-batch calls

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
    batchA, batchB, batchC = _check_batch_dims(A, B, C)

    for i in 1:batchC
        Ai = @view A[:, :, batchA == 1 ? 1 : i]
        Bi = @view B[:, :, batchB == 1 ? 1 : i]
        Ci = @view C[:, :, i]
        BLAS.gemm!(transA, transB, eltype(Ci)(alpha), Ai, Bi, eltype(Ci)(beta), Ci)
    end
    return C
end

function gemm_batched!(transA::Char,
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
    return gemm_batched_ptrs!(
        transA, transB, alpha, Aptrs, Aref, Bptrs, Bref, beta, Cptrs, Cref, batch_count,
    )
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

"""
    gemmEx_batched_ptrs!(transA, transB, alpha, Aptrs, Aref, Bptrs, Bref, beta, Cptrs, Cref, batch_count; compute_type=...)

Internal batched GEMMEx helper that accepts device arrays of matrix base
pointers. `Aref`, `Bref`, and `Cref` are representative matrices used only to
infer the shared GEMM dimensions, element types, and leading dimensions for the
pointed-to batches.
"""
function gemmEx_batched_ptrs!(transA::Char,
                              transB::Char,
                              alpha,
                              Aptrs,
                              Aref::AbstractMatrix,
                              Bptrs,
                              Bref::AbstractMatrix,
                              beta,
                              Cptrs,
                              Cref::AbstractMatrix,
                              batch_count::Integer;
                              compute_type::Type = default_compute_type(alpha, Aref, Bref, beta, Cref))
    _check_compute_type(compute_type)
    throw(ArgumentError("NextLA.gemmEx_batched_ptrs! is supported only on CUDA and AMDGPU"))
end

function gemm_batched!(transA::Char,
                       transB::Char,
                       alpha,
                       A::AbstractVector{<:AbstractArray{<:Any, 2}},
                       B::AbstractVector{<:AbstractArray{<:Any, 2}},
                       beta,
                       C::AbstractVector{<:AbstractArray{<:Any, 2}})
    _check_batch_lengths(A, B, C)
    for i in eachindex(A, B, C)
        BLAS.gemm!(transA, transB, eltype(C[i])(alpha), A[i], B[i], eltype(C[i])(beta), C[i])
    end
    return C
end

"""
    gemmEx_batched!(transA, transB, alpha, A, B, beta, C; compute_type=...)

Compute a mixed-type batched matrix product.

Each batch entry applies

`C[i] := alpha * op(A[i]) * op(B[i]) + beta * C[i]`

where `op(A)` and `op(B)` are determined by `transA` and `transB`.

`gemmEx_batched!` is an advanced `NextLA` API for batched mixed-type GEMM. It
is available as `NextLA.gemmEx_batched!` and is not part of the primary
exported surface.

## Notes
- The accumulation type can be selected explicitly with `compute_type`; by
  default it is inferred from `alpha`, `beta`, and the operand/result element
  types using [`default_compute_type`](@ref).
- Same-type batched calls may fall back to `gemm_batched!` on backends that do
  not expose a mixed-type batched GEMM primitive.
- Mixed-type batched GEMM is currently implemented for CUDA and AMDGPU.
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
    _check_compute_type(compute_type)
    r = _try_same_type_batched!(transA, transB, alpha, A, B, beta, C, compute_type)
    r === nothing || return r
    throw(ArgumentError("NextLA.gemmEx_batched! is supported only on CUDA and AMDGPU for mixed-type batched GEMM"))
end

function gemmEx_batched!(transA::Char,
                         transB::Char,
                         alpha,
                         A::AbstractVector{<:AbstractArray{<:Any, 2}},
                         B::AbstractVector{<:AbstractArray{<:Any, 2}},
                         beta,
                         C::AbstractVector{<:AbstractArray{<:Any, 2}};
                         compute_type::Type = default_compute_type(alpha, A, B, beta, C))
    _check_compute_type(compute_type)
    r = _try_same_type_batched!(transA, transB, alpha, A, B, beta, C, compute_type)
    r === nothing || return r
    throw(ArgumentError("NextLA.gemmEx_batched! is supported only on CUDA and AMDGPU for mixed-type batched GEMM"))
end

function gemmEx_batched!(transA::Char,
                         transB::Char,
                         alpha,
                         Aptrs,
                         Aref::AbstractMatrix,
                         Bptrs,
                         Bref::AbstractMatrix,
                         beta,
                         Cptrs,
                         Cref::AbstractMatrix,
                         batch_count::Integer;
                         compute_type::Type = default_compute_type(alpha, Aref, Bref, beta, Cref))
    return gemmEx_batched_ptrs!(
        transA, transB, alpha, Aptrs, Aref, Bptrs, Bref, beta, Cptrs, Cref, batch_count;
        compute_type,
    )
end
