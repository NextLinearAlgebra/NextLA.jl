const CUBLAS = CUDA.CUBLAS
const CUSOLVER = CUDA.CUSOLVER
const NATIVE_GEMM_TYPES = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

@inline NextLA.SUBGROUP_SIZE(::Type{<:CUDA.CUDABackend}) = Val(32)
@inline NextLA.supports_pointer_batched(::Type{<:CUDA.CUDABackend}) = true

@inline _cublas_compute_type(::Type{Float16}) = CUBLAS.CUBLAS_COMPUTE_16F
@inline _cublas_compute_type(::Type{Float32}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{Float64}) = CUBLAS.CUBLAS_COMPUTE_64F
@inline _cublas_compute_type(::Type{ComplexF32}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{ComplexF64}) = CUBLAS.CUBLAS_COMPUTE_64F
@inline _cublas_compute_type(::Type{Int32}) = CUBLAS.CUBLAS_COMPUTE_32I

@inline _cublas_scalar_type(::Type{Float16}) = Float16
@inline _cublas_scalar_type(::Type{Float32}) = Float32
@inline _cublas_scalar_type(::Type{Float64}) = Float64
@inline _cublas_scalar_type(::Type{ComplexF32}) = ComplexF32
@inline _cublas_scalar_type(::Type{ComplexF64}) = ComplexF64
@inline _cublas_scalar_type(::Type{Int32}) = Int32

@inline _supports_native_gemm(::Type{T}, ::Type{T}, ::Type{T}) where {T<:NATIVE_GEMM_TYPES} = true
@inline _supports_native_gemm(::Type, ::Type, ::Type) = false

@inline function _unsafe_batch_strided(batch::AbstractVector{<:CUDA.StridedCuMatrix{T}}) where {T}
    return CUDA.CuArray(pointer.(batch))
end
