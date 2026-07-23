@inline tlr_orthogonalization_type(::Type{Float16}) = Float32
@inline tlr_orthogonalization_type(::Type{Float32}) = Float64
@inline tlr_orthogonalization_type(::Type{Float64}) = Float64
@inline tlr_orthogonalization_type(::Type{ComplexF32}) = ComplexF64
@inline tlr_orthogonalization_type(::Type{ComplexF64}) = ComplexF64
@inline tlr_orthogonalization_type(::Type{T}) where {T} = T

# Compatibility name for the compression workspace policy. Keeping this alias
# avoids coupling callers to a second, nominally different precision policy.
@inline _compress_accum_type(::Type{T}) where {T} = tlr_orthogonalization_type(T)

@inline _adjoint_blas_char(::Type{<:Complex}) = 'C'
@inline _adjoint_blas_char(::Type) = 'T'

"""
    _batch_views(A, r=size(A, 2)) -> Vector{<:AbstractMatrix}

Per-batch-entry views into a `[m, maxrank, n]` factor array, trimmed to the
first `r` columns. These concrete vectors are built with the workspace and
reused by pointer-batched BLAS paths.
"""
@inline _batch_views(A::AbstractArray{T,3}, r::Int=size(A, 2)) where {T} =
    [view(A, :, 1:r, k) for k in axes(A, 3)]
