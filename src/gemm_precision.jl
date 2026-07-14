export TF32

abstract type AbstractGEMMComputeMode end

"""Request ordinary GEMM accumulation in `T`."""
struct GEMMCompute{T} <: AbstractGEMMComputeMode end

"""Request TensorFloat-32 computation for FP32 GEMM operands and output."""
struct TF32 <: AbstractGEMMComputeMode end

@inline gemm_compute_mode(mode::AbstractGEMMComputeMode) = mode
@inline gemm_compute_mode(::Type{T}) where {T} = GEMMCompute{T}()
@inline gemm_compute_type(::GEMMCompute{T}) where {T} = T
@inline gemm_compute_type(::TF32) = Float32

@inline _batch_eltype(A::AbstractArray) = eltype(A)
@inline _batch_eltype(A::AbstractVector{<:AbstractArray{T}}) where {T} = T

@inline _batch_backend(A::AbstractArray) = get_backend(A)
@inline _batch_backend(A::AbstractVector) = get_backend(first(A))

"""
    _tensor_core_gemm_supported(TA, TB, TC, T)

Shared tensor-core GEMM support matrix used by the CUDA and AMDGPU backends:
native Float16, Float16 operands accumulating into Float16/Float32 under a
Float32 compute type, and native Float32/Float64.
"""
@inline function _tensor_core_gemm_supported(::Type{TA}, ::Type{TB}, ::Type{TC},
                                             ::Type{T}) where {TA,TB,TC,T}
    T === Float16 && return TA === TB === TC === Float16
    T === Float32 && return (TA === TB === Float16 && TC in (Float16, Float32)) ||
                            (TA === TB === TC === Float32)
    T === Float64 && return TA === TB === TC === Float64
    return false
end

function gemm_signature_supported(::Any, ::Type{TA}, ::Type{TB}, ::Type{TC},
                                  ::GEMMCompute{T}) where {TA,TB,TC,T}
    return TA === TB === TC === T && T in (Float32, Float64)
end

@inline gemm_signature_supported(::Any, ::Type, ::Type, ::Type, ::TF32) = false

function validate_gemm_signature(backend, ::Type{TA}, ::Type{TB}, ::Type{TC}, mode) where {TA,TB,TC}
    gemm_signature_supported(backend, TA, TB, TC, mode) && return nothing
    throw(ArgumentError(
        "unsupported GEMM precision signature on $(typeof(backend)): " *
        "$TA × $TB → $TC with compute mode $(typeof(mode))",
    ))
end

function _gemm_compute_batched!(mode::GEMMCompute, transA, transB, alpha, A, B, beta, C)
    return gemmEx_batched!(transA, transB, alpha, A, B, beta, C;
                           compute_type=gemm_compute_type(mode))
end

function _gemm_compute_batched!(::TF32, transA, transB, alpha, A, B, beta, C)
    throw(ArgumentError("TF32 GEMM is supported only on CUDA"))
end

"""
    precision_gemm_batched!(transA, transB, alpha, A, B, beta, C, mode)

Dispatch a batched GEMM from its actual operand and destination storage types plus
an explicit accumulation mode. The destination determines whether this is an
intermediate (`Tin`) or final (`Tout`) contraction stage.
"""
function precision_gemm_batched!(transA, transB, alpha, A, B, beta, C,
                                 mode::AbstractGEMMComputeMode)
    isempty(C) && return C
    backend = _batch_backend(C)
    validate_gemm_signature(
        backend, _batch_eltype(A), _batch_eltype(B), _batch_eltype(C), mode,
    )
    return _gemm_compute_batched!(mode, transA, transB, alpha, A, B, beta, C)
end
