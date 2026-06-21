@inline _syrk_batched_gemm_trans(trans::Char) = trans == 'N' ? 'T' : 'N'

@inline function _syrk_batched_try_gemm!(uplo::Char,
                                         trans::Char,
                                         alpha,
                                         A,
                                         beta,
                                         C)
    uplo in ('L', 'U') || throw(ArgumentError("syrk_batched!: uplo must be 'L' or 'U'"))
    trans in ('N', 'T', 'C') || throw(ArgumentError("syrk_batched!: trans must be 'N', 'T', or 'C'"))
    gemm_batched!(trans, _syrk_batched_gemm_trans(trans), alpha, A, A, beta, C)
    return C
end

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
    backend = KernelAbstractions.get_backend(A)
    if !(backend isa KernelAbstractions.CPU)
        @warn "syrk_batched! falling back to batched gemm!" backend=string(typeof(backend)) layout=:strided maxlog=1
        try
            return _syrk_batched_try_gemm!(uplo, trans, alpha, A, beta, C)
        catch err
            @warn "syrk_batched! falling back to a loop of syrk!" backend=string(typeof(backend)) layout=:strided reason=sprint(showerror, err) maxlog=1
        end
    end
    return _syrk_batched_loop!(uplo, trans, alpha, A, beta, C)
end

function _syrk_batched_loop!(uplo::Char,
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
    isempty(A) && return C
    backend = KernelAbstractions.get_backend(first(A))
    if !(backend isa KernelAbstractions.CPU)
        @warn "syrk_batched! falling back to batched gemm!" backend=string(typeof(backend)) layout=:pointer maxlog=1
        try
            return _syrk_batched_try_gemm!(uplo, trans, alpha, A, beta, C)
        catch err
            @warn "syrk_batched! falling back to a loop of syrk!" backend=string(typeof(backend)) layout=:pointer reason=sprint(showerror, err) maxlog=1
        end
    end
    return _syrk_batched_loop!(uplo, trans, alpha, A, beta, C)
end

function _syrk_batched_loop!(uplo::Char,
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
