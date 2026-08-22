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

# --- persistent pointer-batch descriptors -----------------------------------
#
# `gemm_batched!`/`gemmEx_batched!`'s `Vector`-of-matrix methods build a fresh
# device pointer array and free it again on every call (see each backend
# extension's `_unsafe_batch_strided`/`_device_batch_strided`). That is the
# right default for one-shot calls, but wrong for a loop that issues the same
# batched shape on every iteration (e.g. one call per ARA sampling pass): the
# repeated device allocation, H2D upload, and free dominates the actual GEMM
# for small batches. `BatchPtrDescriptor` lets such a caller build the pointer
# table once and reuse it, updating only the tiny address table (via
# `swap_batch_ptrs!`) when which logical member occupies which physical slot
# changes, rather than rebuilding the table or moving the numeric data itself.

export BatchPtrDescriptor, swap_batch_ptrs!, set_batch_ptrs!

"""
    BatchPtrDescriptor(ptrs)
    BatchPtrDescriptor(batch::AbstractVector{<:AbstractMatrix})

A device array of matrix base pointers built once and reused across many
batched GEMM calls (via [`gemm_batched_ptrs!`](@ref)/[`gemmEx_batched_ptrs!`](@ref)),
in contrast to the transient pointer arrays the `Vector`-of-matrix
`gemm_batched!`/`gemmEx_batched!` methods build and free on every call.

A descriptor carries only addresses, no shape/element-type/leading-dimension
information — callers supply a fresh, cheap representative `Aref`/`Bref`/`Cref`
view at each call to size that call's GEMM. Slot `k` is stable across calls:
if which logical batch member occupies slot `k` changes (e.g. active-prefix
packing retiring a converged member), that must be expressed with
[`swap_batch_ptrs!`](@ref), which moves addresses between slots, never by
physically reordering the pointed-to data or by rebuilding the descriptor.

The caller owns the descriptor's lifetime and must keep the pointed-to
storage alive for as long as the descriptor is used; unlike the transient
pointer arrays built by the `Vector`-of-matrix methods, nothing frees a
`BatchPtrDescriptor`'s pointer array automatically.
"""
struct BatchPtrDescriptor{PT<:AbstractVector,H<:AbstractVector}
    ptrs::PT
    host::H
end

function BatchPtrDescriptor(batch::AbstractVector{<:AbstractMatrix})
    ptrs = _build_batch_ptrs(batch)
    host = Vector{eltype(ptrs)}(undef, length(ptrs))
    @inbounds for k in eachindex(batch)
        host[k] = eltype(ptrs)(pointer(batch[k]))
    end
    return BatchPtrDescriptor(ptrs, host)
end

Base.length(d::BatchPtrDescriptor) = length(d.ptrs)
KernelAbstractions.get_backend(d::BatchPtrDescriptor) = get_backend(d.ptrs)

"""
    _build_batch_ptrs(batch::AbstractVector{<:AbstractMatrix}) -> device pointer array

Backend hook constructing the actual device array of base pointers for
[`BatchPtrDescriptor`](@ref). CUDA and AMDGPU provide it, with the same body
as their existing transient pointer-array constructors. There is no CPU
method: CPU batched GEMM loops the `Vector` of matrices directly and never
needs a pointer array, so a `BatchPtrDescriptor` is never constructed there.
"""
function _build_batch_ptrs(batch::AbstractVector{<:AbstractMatrix})
    throw(ArgumentError("NextLA._build_batch_ptrs is supported only on CUDA and AMDGPU"))
end

@kernel function _swap_batch_ptr_kernel!(ptrs, p::Int, q::Int)
    _ = @index(Global, Linear)
    @inbounds begin
        x = ptrs[p]
        ptrs[p] = ptrs[q]
        ptrs[q] = x
    end
end

@kernel function _swap_batch_ptr_block_kernel!(ptrs, p::Int, q::Int, blocklen::Int)
    i = @index(Global, Linear)
    @inbounds begin
        pi = (p - 1) * blocklen + i
        qi = (q - 1) * blocklen + i
        x = ptrs[pi]
        ptrs[pi] = ptrs[qi]
        ptrs[qi] = x
    end
end

"""
    swap_batch_ptrs!(d::BatchPtrDescriptor, p::Int, q::Int)

Swap which base address occupies descriptor slots `p` and `q`. Only the small
address table moves; the numeric data the addresses point to is untouched and
does not move. Use this instead of physically swapping or copying the
underlying batch members when active-prefix packing retires one.
"""
function swap_batch_ptrs!(d::BatchPtrDescriptor, p::Int, q::Int)
    p == q && return d
    _swap_batch_ptr_kernel!(get_backend(d))(d.ptrs, p, q; ndrange=(1,))
    return d
end

"""
    swap_batch_ptrs!(d::BatchPtrDescriptor, p::Int, q::Int, blocklen::Int)

Block form of [`swap_batch_ptrs!`](@ref) for descriptors where each logical
batch member owns `blocklen` consecutive descriptor slots (e.g. one pointer
per contraction tile of a member's factor panel, laid out member-major):
swaps the two length-`blocklen` contiguous slot ranges
`[(p-1)*blocklen+1:p*blocklen]` and `[(q-1)*blocklen+1:q*blocklen]`.
"""
function swap_batch_ptrs!(d::BatchPtrDescriptor, p::Int, q::Int, blocklen::Int)
    (p == q || blocklen == 0) && return d
    _swap_batch_ptr_block_kernel!(get_backend(d))(
        d.ptrs, p, q, blocklen; ndrange=(blocklen,),
    )
    return d
end

"""
    set_batch_ptrs!(d, first, matrices)

Replace a contiguous descriptor range with the base addresses of `matrices`.
The descriptor storage is retained; only its small address table is updated.
This is used when a rolling scheduler admits a new logical member into an
existing physical slot.
"""
function set_batch_ptrs!(d::BatchPtrDescriptor, first::Int,
                         matrices::AbstractVector{<:AbstractMatrix})
    last = first + length(matrices) - 1
    1 <= first <= last + 1 && last <= length(d) ||
        throw(BoundsError(d.ptrs, first:last))
    isempty(matrices) && return d
    @inbounds for k in eachindex(matrices)
        d.host[first + k - 1] = eltype(d.ptrs)(pointer(matrices[k]))
    end
    copyto!(d.ptrs, d.host)
    return d
end
