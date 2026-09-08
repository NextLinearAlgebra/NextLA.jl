@inline _rocsolver_potrf_strided_batched_fname(::Type{Float32}) = rocSOLVER.rocsolver_spotrf_strided_batched_64
@inline _rocsolver_potrf_strided_batched_fname(::Type{Float64}) = rocSOLVER.rocsolver_dpotrf_strided_batched_64
@inline _rocsolver_potrf_strided_batched_fname(::Type{ComplexF32}) = rocSOLVER.rocsolver_cpotrf_strided_batched_64
@inline _rocsolver_potrf_strided_batched_fname(::Type{ComplexF64}) = rocSOLVER.rocsolver_zpotrf_strided_batched_64

function _potrf_batched_amdgpu!(uplo::Char,
                                A::AMDGPU.StridedROCArray{T,3}) where {T}
    n = LinearAlgebra.checksquare(@view A[:, :, 1])
    lda = max(1, stride(A, 2))
    strideA = stride(A, 3)
    batch_count = size(A, 3)
    status = similar(A, Int32, batch_count)
    fname = _rocsolver_potrf_strided_batched_fname(T)
    fname(rocBLAS.handle(), uplo, n, A, lda, strideA, status, batch_count)
    return A, status
end

function NextLA.potrf_batched!(uplo::Char,
                               A::AMDGPU.StridedROCArray{T,3}) where {T}
    return _potrf_batched_amdgpu!(uplo, A)
end
