"""
    syrk_batched!(uplo, trans, alpha, A, beta, C)

Perform a batched symmetric rank-k update.

Two storage forms are supported:
- `AbstractArray{T,3}` for strided batches
- `AbstractVector` of matrices for pointer batches

Each batch entry applies

`C[i] := alpha * op(A[i]) * op(A[i])' + beta * C[i]`

where `op(A)` is determined by `trans`.

This is currently an internal `NextLA` API and is not part of the primary
exported surface.

## Notes
- If a backend does not provide a native batched SYRK kernel, `syrk_batched!`
  falls back to batched GEMM.
- When this fallback is used, `syrk_batched!` emits `@warn`.
"""
function syrk_batched!(uplo::Char,
                       trans::Char,
                       alpha,
                       A::AbstractArray{<:Any, 3},
                       beta,
                       C::AbstractArray{<:Any, 3})
    size(A, 3) == size(C, 3) || size(A, 3) == 1 ||
        throw(DimensionMismatch("syrk_batched!: A and C batch sizes are incompatible"))
    @warn "syrk_batched! falling back to batched gemm!" backend=string(typeof(A)) layout=:strided
    return _syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
end

function _syrk_batched_fallback!(uplo::Char,
                                 trans::Char,
                                 alpha,
                                 A::AbstractArray{<:Any, 3},
                                 beta,
                                 C::AbstractArray{<:Any, 3})
    batchA = size(A, 3)
    batchC = size(C, 3)

    for i in 1:batchC
        Ai = @view A[:, :, batchA == 1 ? 1 : i]
        Ci = @view C[:, :, i]
        syrk!(uplo, trans, alpha, Ai, beta, Ci)
    end

    return C
end

function syrk_batched!(uplo::Char,
                       trans::Char,
                       alpha,
                       A::AbstractVector{<:AbstractMatrix},
                       beta,
                       C::AbstractVector{<:AbstractMatrix})
    length(A) == length(C) || throw(DimensionMismatch("syrk_batched!: matrix batches must have matching lengths"))
    @warn "syrk_batched! falling back to batched gemm!" backend=string(typeof(A)) layout=:pointer
    return _syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
end

function _syrk_batched_fallback!(uplo::Char,
                                 trans::Char,
                                 alpha,
                                 A::AbstractVector{<:AbstractMatrix},
                                 beta,
                                 C::AbstractVector{<:AbstractMatrix})
    for i in eachindex(A, C)
        syrk!(uplo, trans, alpha, A[i], beta, C[i])
    end

    return C
end
