module NextLAAMDGPUExt

using NextLA
using AMDGPU

const rocBLAS = AMDGPU.rocBLAS

const NATIVE_GEMM_TYPES = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}
const NATIVE_STRIDED_BATCHED_TYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

@inline NextLA.SUBGROUP_SIZE(::Type{<:AMDGPU.ROCBackend}) = Val(64)

@inline _rocblas_datatype(::Type{Float16}) = rocBLAS.rocblas_datatype_f16_r
@inline _rocblas_datatype(::Type{Float32}) = rocBLAS.rocblas_datatype_f32_r
@inline _rocblas_datatype(::Type{Float64}) = rocBLAS.rocblas_datatype_f64_r
@inline _rocblas_datatype(::Type{ComplexF32}) = rocBLAS.rocblas_datatype_f32_c
@inline _rocblas_datatype(::Type{ComplexF64}) = rocBLAS.rocblas_datatype_f64_c
@inline _rocblas_datatype(::Type{Int8}) = rocBLAS.rocblas_datatype_i8_r
@inline _rocblas_datatype(::Type{Int32}) = rocBLAS.rocblas_datatype_i32_r

@inline _supports_native_gemm(::Type{T}, ::Type{T}, ::Type{T}) where {T<:NATIVE_GEMM_TYPES} = true
@inline _supports_native_gemm(::Type, ::Type, ::Type) = false

@inline _supports_native_strided_batched(::Type{T}, ::Type{T}, ::Type{T}) where {T<:NATIVE_STRIDED_BATCHED_TYPES} = true
@inline _supports_native_strided_batched(::Type, ::Type, ::Type) = false

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

function _syrk_native!(uplo::Char,
                       trans::Char,
                       alpha,
                       A::AMDGPU.StridedROCMatrix{<:Any},
                       beta,
                       C::AMDGPU.StridedROCMatrix{<:Any})
    n, k = NextLA._syrk_dims(uplo, trans, A, C)
    lda = max(1, stride(A, 2))
    ldc = max(1, stride(C, 2))
    fname = _rocblas_syrk_fname(eltype(A), Val(:single))
    fname(rocBLAS.handle(), uplo, trans, n, k, Ref(alpha), A, lda, Ref(beta), C, ldc)
    return C
end

function _syrk_batched_native!(uplo::Char,
                               trans::Char,
                               alpha,
                               A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                               beta,
                               C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}})
    length(A) == length(C) || throw(DimensionMismatch("syrk_batched!: matrix batches must have matching lengths"))
    isempty(A) && return C
    n, k = NextLA._syrk_dims(uplo, trans, A[1], C[1])
    lda = max(1, stride(A[1], 2))
    ldc = max(1, stride(C[1], 2))
    Aptrs = rocBLAS.device_batch(A)
    Cptrs = rocBLAS.device_batch(C)
    fname = _rocblas_syrk_fname(eltype(A[1]), Val(:batched))
    fname(rocBLAS.handle(), uplo, trans, n, k, Ref(alpha), Aptrs, lda, Ref(beta), Cptrs, ldc, length(C))
    return C
end

function _syrk_batched_native!(uplo::Char,
                               trans::Char,
                               alpha,
                               A::AMDGPU.StridedROCArray{<:Any, 3},
                               beta,
                               C::AMDGPU.StridedROCArray{<:Any, 3})
    size(A, 3) == size(C, 3) || size(A, 3) == 1 ||
        throw(DimensionMismatch("syrk_batched!: A and C batch sizes are incompatible"))
    n, k = NextLA._syrk_dims(uplo, trans, @view(A[:, :, 1]), @view(C[:, :, 1]))
    lda = max(1, stride(A, 2))
    ldc = max(1, stride(C, 2))
    strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
    strideC = stride(C, 3)
    fname = _rocblas_syrk_fname(eltype(A), Val(:strided))
    fname(rocBLAS.handle(), uplo, trans, n, k, Ref(alpha), A, lda, strideA, Ref(beta), C, ldc, strideC, size(C, 3))
    return C
end

function NextLA.syrk!(uplo::Char,
                      trans::Char,
                      alpha,
                      A::AMDGPU.StridedROCMatrix{<:Any},
                      beta,
                      C::AMDGPU.StridedROCMatrix{<:Any})
    return _syrk_native!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                              beta,
                              C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}})
    return _syrk_batched_native!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AMDGPU.StridedROCArray{<:Any, 3},
                              beta,
                              C::AMDGPU.StridedROCArray{<:Any, 3})
    return _syrk_batched_native!(uplo, trans, alpha, A, beta, C)
end

# Ex verions 

function _gemm_ex!(transA::Char,
                   transB::Char,
                   alpha,
                   A::AMDGPU.ROCArray{<:Any, 2},
                   B::AMDGPU.ROCArray{<:Any, 2},
                   beta,
                   C::AMDGPU.ROCArray{<:Any, 2},
                   compute_type::Type)
    m, n, k, lda, ldb, ldc = _gemm_dims(transA, transB, A, B, C)
    scalar_type = compute_type
    alpha_ref = Ref{scalar_type}(alpha)
    beta_ref = Ref{scalar_type}(beta)
    rocBLAS.rocblas_gemm_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, A, _rocblas_datatype(eltype(A)), lda,
        B, _rocblas_datatype(eltype(B)), ldb,
        beta_ref, C, _rocblas_datatype(eltype(C)), ldc,
        C, _rocblas_datatype(eltype(C)), ldc,
        _rocblas_datatype(compute_type),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return C
end

function _gemm_batched_ex!(transA::Char,
                           transB::Char,
                           alpha,
                           A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                           B::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                           beta,
                           C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                           compute_type::Type)
    length(A) == length(B) == length(C) || throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    isempty(A) && return C

    m, n, k, lda, ldb, ldc = _gemm_dims(transA, transB, A[1], B[1], C[1])
    for i in eachindex(A, B, C)
        _gemm_dims(transA, transB, A[i], B[i], C[i]) == (m, n, k, lda, ldb, ldc) ||
            throw(DimensionMismatch("gemm_batched!: all matrices in a batch must share dimensions and strides"))
    end

    scalar_type = compute_type
    alpha_ref = Ref{scalar_type}(alpha)
    beta_ref = Ref{scalar_type}(beta)
    Aptrs = rocBLAS.device_batch(A)
    Bptrs = rocBLAS.device_batch(B)
    Cptrs = rocBLAS.device_batch(C)
    rocBLAS.rocblas_gemm_batched_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, Aptrs, _rocblas_datatype(eltype(A[1])), lda,
        Bptrs, _rocblas_datatype(eltype(B[1])), ldb,
        beta_ref, Cptrs, _rocblas_datatype(eltype(C[1])), ldc,
        Cptrs, _rocblas_datatype(eltype(C[1])), ldc,
        length(C), _rocblas_datatype(compute_type),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return C
end

function _gemm_strided_batched_ex!(transA::Char,
                                   transB::Char,
                                   alpha,
                                   A::AMDGPU.StridedROCArray{<:Any, 3},
                                   B::AMDGPU.StridedROCArray{<:Any, 3},
                                   beta,
                                   C::AMDGPU.StridedROCArray{<:Any, 3},
                                   compute_type::Type)
    batchA = size(A, 3)
    batchB = size(B, 3)
    batchC = size(C, 3)
    (batchA == batchC || batchA == 1) || throw(DimensionMismatch("gemm_batched!: A and C batch sizes are incompatible"))
    (batchB == batchC || batchB == 1) || throw(DimensionMismatch("gemm_batched!: B and C batch sizes are incompatible"))

    m, n, k, lda, ldb, ldc = _gemm_dims(transA, transB, @view(A[:, :, 1]), @view(B[:, :, 1]), @view(C[:, :, 1]))
    strideA = batchA == 1 ? 0 : stride(A, 3)
    strideB = batchB == 1 ? 0 : stride(B, 3)
    strideC = stride(C, 3)
    scalar_type = compute_type
    alpha_ref = Ref{scalar_type}(alpha)
    beta_ref = Ref{scalar_type}(beta)

    rocBLAS.rocblas_gemm_strided_batched_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, A, _rocblas_datatype(eltype(A)), lda, strideA,
        B, _rocblas_datatype(eltype(B)), ldb, strideB,
        beta_ref, C, _rocblas_datatype(eltype(C)), ldc, strideC,
        C, _rocblas_datatype(eltype(C)), ldc, strideC,
        batchC, _rocblas_datatype(compute_type),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return C
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::AMDGPU.ROCArray{<:Any, 2},
                        B::AMDGPU.ROCArray{<:Any, 2},
                        beta,
                        C::AMDGPU.ROCArray{<:Any, 2};
                        compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if compute_type == NextLA.default_compute_type(alpha, A, B, beta, C) &&
       _supports_native_gemm(eltype(A), eltype(B), eltype(C))
        return rocBLAS.gemm!(transA, transB, eltype(C)(alpha), A, B, eltype(C)(beta), C)
    end
    return _gemm_ex!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                              B::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                              beta,
                              C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}})
    return rocBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AMDGPU.StridedROCArray{<:Any, 3},
                              B::AMDGPU.StridedROCArray{<:Any, 3},
                              beta,
                              C::AMDGPU.StridedROCArray{<:Any, 3})
    if _supports_native_strided_batched(eltype(A), eltype(B), eltype(C))
        return _gemm_strided_batched_ex!(transA, transB, alpha, A, B, beta, C, eltype(C))
    elseif _supports_native_gemm(eltype(A), eltype(B), eltype(C))
        if A isa AMDGPU.ROCArray{<:Any, 3} &&
           B isa AMDGPU.ROCArray{<:Any, 3} &&
           C isa AMDGPU.ROCArray{<:Any, 3}
            return rocBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
        end
        throw(ArgumentError("AMDGPU strided batched GEMM views are supported only for eltypes with native strided-batched kernels"))
    end
    throw(ArgumentError("AMDGPU strided batched GEMM is not supported for eltypes $(eltype(A)), $(eltype(B)), and $(eltype(C))"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                                B::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                                beta,
                                C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if compute_type == NextLA.default_compute_type(alpha, A, B, beta, C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    return _gemm_batched_ex!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AMDGPU.StridedROCArray{<:Any, 3},
                                B::AMDGPU.StridedROCArray{<:Any, 3},
                                beta,
                                C::AMDGPU.StridedROCArray{<:Any, 3};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if compute_type == NextLA.default_compute_type(alpha, A, B, beta, C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    return _gemm_strided_batched_ex!(transA, transB, alpha, A, B, beta, C, compute_type)
end

end
