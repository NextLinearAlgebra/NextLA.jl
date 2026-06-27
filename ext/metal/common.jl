const MPS = Metal.MPS
const MtlMatrixBatchView{T} = SubArray{T, 3, <:Metal.MtlArray{T, 3}, <:Any, false} where {T}

@inline NextLA.SUBGROUP_SIZE(::Type{<:Metal.MetalBackend}) = Val(64)

@inline function _supports_mps_batched_matmul(::Type{Tin}, ::Type{Tin}, ::Type{Tout}) where {Tin, Tout}
    return (Tin, Tout) in MPS.MPS_VALID_MATMUL_TYPES
end

@inline function _supports_mps_matmul(::Type{Tin}, ::Type{Tin}, ::Type{Tout}) where {Tin, Tout}
    return (Tin, Tout) in MPS.MPS_VALID_MATMUL_TYPES
end

@inline function _materialize_batch_view(A::MtlMatrixBatchView)
    dense = similar(parent(A), size(A))
    copyto!(dense, A)
    return dense
end

@inline _dense_mtl_batch(A::Metal.MtlArray{<:Any,3}) = A
@inline _dense_mtl_batch(A::MtlMatrixBatchView) = _materialize_batch_view(A)

@inline function _copyback_mtl_batch!(C::Metal.MtlArray{<:Any,3}, Cdense)
    return Cdense
end

@inline function _copyback_mtl_batch!(C::MtlMatrixBatchView, Cdense)
    copyto!(C, Cdense)
    return C
end
