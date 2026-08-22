const CUBLAS = CUDA.CUBLAS
const CUSOLVER = CUDA.CUSOLVER

@inline NextLA.SUBGROUP_SIZE(::Type{<:CUDA.CUDABackend}) = Val(32)
@inline NextLA.supports_pointer_batched(::Type{<:CUDA.CUDABackend}) = true

@inline _cublas_compute_type(::Type{Float16}) = CUBLAS.CUBLAS_COMPUTE_16F
@inline _cublas_compute_type(::Type{Core.BFloat16}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{Float32}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{Float64}) = CUBLAS.CUBLAS_COMPUTE_64F
@inline _cublas_compute_type(::Type{ComplexF32}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{ComplexF64}) = CUBLAS.CUBLAS_COMPUTE_64F
@inline _cublas_compute_type(::Type{Int32}) = CUBLAS.CUBLAS_COMPUTE_32I

@inline _cublas_scalar_type(::Type{Float16}) = Float16
@inline _cublas_scalar_type(::Type{Core.BFloat16}) = Float32
@inline _cublas_scalar_type(::Type{Float32}) = Float32
@inline _cublas_scalar_type(::Type{Float64}) = Float64
@inline _cublas_scalar_type(::Type{ComplexF32}) = ComplexF32
@inline _cublas_scalar_type(::Type{ComplexF64}) = ComplexF64
@inline _cublas_scalar_type(::Type{Int32}) = Int32

function NextLA.gemm_signature_supported(::CUDA.CUDABackend,
                                         ::Type{TA}, ::Type{TB}, ::Type{TC},
                                         ::NextLA.GEMMCompute{T}) where {TA,TB,TC,T}
    return NextLA._tensor_core_gemm_supported(TA, TB, TC, T)
end

@inline NextLA.supports_bfloat16_grouped_gemm(::CUDA.CUDABackend) =
    CUDA.capability(CUDA.device()) >= v"8.0"

@inline NextLA.gemm_signature_supported(::CUDA.CUDABackend,
                                        ::Type{Float32}, ::Type{Float32}, ::Type{Float32},
                                        ::NextLA.TF32) = true

@inline function _unsafe_batch_strided(batch::AbstractVector{<:CUDA.StridedCuMatrix{T}}) where {T}
    return CUDA.CuArray(pointer.(batch))
end

@inline NextLA._build_batch_ptrs(batch::AbstractVector{<:CUDA.StridedCuMatrix}) =
    _unsafe_batch_strided(batch)
