function _potrf_batched_oneapi!(uplo::Char,
                                A::oneAPI.oneStridedArray{T,3}) where {T}
    n = LinearAlgebra.checksquare(@view A[:, :, 1])
    lda = max(1, stride(A, 2))
    strideA = stride(A, 3)
    batch_count = size(A, 3)
    queue = oneMKL.global_queue(oneAPI.context(A), oneAPI.device(A))
    scratchpad_size = _onemkl_potrf_strided_scratchpad_fname(T)(oneAPI.sycl_queue(queue), uplo, n, lda, strideA, batch_count)
    scratchpad = oneAPI.oneVector{T}(undef, scratchpad_size)
    try
        _onemkl_potrf_strided_fname(T)(oneAPI.sycl_queue(queue), uplo, n, A, lda, strideA, batch_count, scratchpad, scratchpad_size)
    finally
        oneAPI.unsafe_free!(scratchpad)
    end
    return A
end

function NextLA.potrf_batched!(uplo::Char,
                               A::oneAPI.oneStridedArray{T,3}) where {T}
    return _potrf_batched_oneapi!(uplo, A)
end
