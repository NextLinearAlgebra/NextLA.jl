const rocBLAS = AMDGPU.rocBLAS
# rocSOLVER is the solver library rather than BLAS; as with CUSOLVER it
# arrives with the Cholesky work that first needs it.
const rocSOLVER = AMDGPU.rocSOLVER

const NATIVE_STRIDED_BATCHED_TYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

@inline NextLA.SUBGROUP_SIZE(::Type{<:AMDGPU.ROCBackend}) = Val(64)
@inline NextLA.supports_pointer_batched(::Type{<:AMDGPU.ROCBackend}) = true

@inline _rocblas_datatype(::Type{Float16}) = rocBLAS.rocblas_datatype_f16_r
@inline _rocblas_datatype(::Type{Core.BFloat16}) = rocBLAS.rocblas_datatype_bf16_r
@inline _rocblas_datatype(::Type{Float32}) = rocBLAS.rocblas_datatype_f32_r
@inline _rocblas_datatype(::Type{Float64}) = rocBLAS.rocblas_datatype_f64_r
@inline _rocblas_datatype(::Type{ComplexF32}) = rocBLAS.rocblas_datatype_f32_c
@inline _rocblas_datatype(::Type{ComplexF64}) = rocBLAS.rocblas_datatype_f64_c
@inline _rocblas_datatype(::Type{Int8}) = rocBLAS.rocblas_datatype_i8_r
@inline _rocblas_datatype(::Type{Int32}) = rocBLAS.rocblas_datatype_i32_r

@inline _supports_native_strided_batched(::Type{T}, ::Type{T}, ::Type{T}) where {T<:NATIVE_STRIDED_BATCHED_TYPES} = true
@inline _supports_native_strided_batched(::Type, ::Type, ::Type) = false

function NextLA.gemm_signature_supported(::AMDGPU.ROCBackend,
                                         ::Type{TA}, ::Type{TB}, ::Type{TC},
                                         ::NextLA.GEMMCompute{T}) where {TA,TB,TC,T}
    return NextLA._tensor_core_gemm_supported(TA, TB, TC, T)
end

@inline function _device_batch_strided(batch::AbstractVector{<:AMDGPU.StridedROCMatrix{T}}) where {T}
    return AMDGPU.ROCArray(pointer.(batch))
end

@inline NextLA._build_batch_ptrs(batch::AbstractVector{<:AMDGPU.StridedROCMatrix}) =
    _device_batch_strided(batch)

# SYRK entry points, added with src/syrk/. rocBLAS is the only backend
# with a native batched SYRK in both layouts.
@inline _rocblas_syrk_fname(::Type{Float32}, ::Val{:single}) = rocBLAS.rocblas_ssyrk_64
@inline _rocblas_syrk_fname(::Type{Float64}, ::Val{:single}) = rocBLAS.rocblas_dsyrk_64
@inline _rocblas_syrk_fname(::Type{ComplexF32}, ::Val{:single}) = rocBLAS.rocblas_csyrk_64
@inline _rocblas_syrk_fname(::Type{ComplexF64}, ::Val{:single}) = rocBLAS.rocblas_zsyrk_64
@inline _rocblas_syrk_fname(::Type{Float32}, ::Val{:batched}) = rocBLAS.rocblas_ssyrk_batched_64
@inline _rocblas_syrk_fname(::Type{Float64}, ::Val{:batched}) = rocBLAS.rocblas_dsyrk_batched_64
@inline _rocblas_syrk_fname(::Type{ComplexF32}, ::Val{:batched}) = rocBLAS.rocblas_csyrk_batched_64
@inline _rocblas_syrk_fname(::Type{ComplexF64}, ::Val{:batched}) = rocBLAS.rocblas_zsyrk_batched_64
@inline _rocblas_syrk_fname(::Type{Float32}, ::Val{:strided}) = rocBLAS.rocblas_ssyrk_strided_batched_64
@inline _rocblas_syrk_fname(::Type{Float64}, ::Val{:strided}) = rocBLAS.rocblas_dsyrk_strided_batched_64
@inline _rocblas_syrk_fname(::Type{ComplexF32}, ::Val{:strided}) = rocBLAS.rocblas_csyrk_strided_batched_64
@inline _rocblas_syrk_fname(::Type{ComplexF64}, ::Val{:strided}) = rocBLAS.rocblas_zsyrk_strided_batched_64
