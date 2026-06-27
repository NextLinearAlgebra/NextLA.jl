function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AbstractVector{<:oneAPI.oneStridedMatrix{T}},
                              B::AbstractVector{<:oneAPI.oneStridedMatrix{T}},
                              alpha=one(T)) where {T}
    length(A) == length(B) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    isempty(A) && return B

    mA, nA = size(A[1])
    m, n = size(B[1])
    mA == nA || throw(DimensionMismatch("A must be square"))
    nA == (side == 'L' ? m : n) || throw(DimensionMismatch("trsm_batched!"))

    lda = max(1, stride(A[1], 2))
    ldb = max(1, stride(B[1], 2))
    Aptrs = oneAPI.unsafe_batch(A)
    Bptrs = oneAPI.unsafe_batch(B)
    queue = oneMKL.global_queue(oneAPI.context(A[1]), oneAPI.device(A[1]))
    bsize = length(A)
    m_dev = oneAPI.oneVector{Int}(fill(m, bsize))
    n_dev = oneAPI.oneVector{Int}(fill(n, bsize))
    lda_dev = oneAPI.oneVector{Int}(fill(lda, bsize))
    ldb_dev = oneAPI.oneVector{Int}(fill(ldb, bsize))
    alpha_dev = oneAPI.oneVector{T}(fill(T(alpha), bsize))
    groupsize_dev = oneAPI.oneVector{Int}(fill(1, bsize))

    try
        fname = _onemkl_trsm_fname(T, Val(:pointer))
        fname(oneAPI.sycl_queue(queue), side, uplo, transa, diag, m_dev, n_dev, alpha_dev, Aptrs, lda_dev, Bptrs, ldb_dev, bsize, groupsize_dev)
    finally
        oneAPI.unsafe_free!(groupsize_dev)
        oneAPI.unsafe_free!(alpha_dev)
        oneAPI.unsafe_free!(ldb_dev)
        oneAPI.unsafe_free!(lda_dev)
        oneAPI.unsafe_free!(n_dev)
        oneAPI.unsafe_free!(m_dev)
        oneAPI.unsafe_free!(Bptrs)
        oneAPI.unsafe_free!(Aptrs)
    end

    return B
end

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::oneAPI.oneStridedArray{T,3},
                              B::oneAPI.oneStridedArray{T,3},
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
    queue = oneMKL.global_queue(oneAPI.context(A), oneAPI.device(A))
    fname = _onemkl_trsm_fname(T, Val(:strided))
    fname(oneAPI.sycl_queue(queue), side, uplo, transa, diag, m, n, Ref(T(alpha)), A, lda, strideA, B, ldb, strideB, size(B, 3))
    return B
end
