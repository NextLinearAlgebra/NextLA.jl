@inline _rocblas_trsm_strided_batched_fname(::Type{Float32}) = rocBLAS.rocblas_strsm_strided_batched_64
@inline _rocblas_trsm_strided_batched_fname(::Type{Float64}) = rocBLAS.rocblas_dtrsm_strided_batched_64
@inline _rocblas_trsm_strided_batched_fname(::Type{ComplexF32}) = rocBLAS.rocblas_ctrsm_strided_batched_64
@inline _rocblas_trsm_strided_batched_fname(::Type{ComplexF64}) = rocBLAS.rocblas_ztrsm_strided_batched_64

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AbstractVector{<:AMDGPU.StridedROCMatrix{T}},
                              B::AbstractVector{<:AMDGPU.StridedROCMatrix{T}},
                              alpha=one(T)) where {T}
    rocBLAS.trsm_batched!(side, uplo, transa, diag, alpha, A, B)
    return B
end

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AMDGPU.StridedROCArray{T,3},
                              B::AMDGPU.StridedROCArray{T,3},
                              alpha=one(T)) where {T}
    size(A, 3) == size(B, 3) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    mA, nA = size(@view A[:, :, 1])
    m, n = size(@view B[:, :, 1])
    mA == nA || throw(DimensionMismatch("A must be square"))
    nA == (side == 'L' ? m : n) || throw(DimensionMismatch("trsm_batched!"))
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    strideA = stride(A, 3)
    strideB = stride(B, 3)
    alpha_ref = Ref{T}(alpha)
    fname = _rocblas_trsm_strided_batched_fname(T)
    fname(rocBLAS.handle(), side, uplo, transa, diag, m, n, alpha_ref, A, lda, strideA, B, ldb, strideB, size(B, 3))
    return B
end
