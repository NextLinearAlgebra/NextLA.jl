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
                               A::AbstractVector{<:AMDGPU.ROCArray{T, 2}},
                               beta,
                               C::AbstractVector{<:AMDGPU.ROCArray{T, 2}}) where {T}
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
                              A::AbstractVector{<:AMDGPU.ROCArray{T, 2}},
                              beta,
                              C::AbstractVector{<:AMDGPU.ROCArray{T, 2}}) where {T}
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
