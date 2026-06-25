module NextLAoneAPIExt

using NextLA
using oneAPI

const oneMKL = oneAPI.oneMKL
const support = oneAPI.Support

@inline NextLA.SUBGROUP_SIZE(::Type{<:oneAPI.oneAPIBackend}) = Val(32)

@inline _onemkl_syrk_fname(::Type{Float32}) = support.onemklSsyrk_batch_strided
@inline _onemkl_syrk_fname(::Type{Float64}) = support.onemklDsyrk_batch_strided
@inline _onemkl_syrk_fname(::Type{ComplexF32}) = support.onemklCsyrk_batch_strided
@inline _onemkl_syrk_fname(::Type{ComplexF64}) = support.onemklZsyrk_batch_strided
@inline _onemkl_trsm_fname(::Type{Float32}, ::Val{:pointer}) = support.onemklStrsm_batch
@inline _onemkl_trsm_fname(::Type{Float64}, ::Val{:pointer}) = support.onemklDtrsm_batch
@inline _onemkl_trsm_fname(::Type{ComplexF32}, ::Val{:pointer}) = support.onemklCtrsm_batch
@inline _onemkl_trsm_fname(::Type{ComplexF64}, ::Val{:pointer}) = support.onemklZtrsm_batch
@inline _onemkl_trsm_fname(::Type{Float32}, ::Val{:strided}) = support.onemklStrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{Float64}, ::Val{:strided}) = support.onemklDtrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{ComplexF32}, ::Val{:strided}) = support.onemklCtrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{ComplexF64}, ::Val{:strided}) = support.onemklZtrsm_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{Float32}) = support.onemklSpotrf_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{Float64}) = support.onemklDpotrf_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{ComplexF32}) = support.onemklCpotrf_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{ComplexF64}) = support.onemklZpotrf_batch_strided
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{Float32}) = support.onemklSpotrf_batch_strided_scratchpad_size
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{Float64}) = support.onemklDpotrf_batch_strided_scratchpad_size
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{ComplexF32}) = support.onemklCpotrf_batch_strided_scratchpad_size
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{ComplexF64}) = support.onemklZpotrf_batch_strided_scratchpad_size

function _syrk_strided_batched_native!(uplo::Char,
                                       trans::Char,
                                       alpha,
                                       A::oneAPI.oneStridedArray{T,3},
                                       beta,
                                       C::oneAPI.oneStridedArray{T,3}) where {T}
    size(A, 3) == size(C, 3) || size(A, 3) == 1 ||
        throw(DimensionMismatch("syrk_batched!: A and C batch sizes are incompatible"))
    n, k = NextLA._syrk_dims(uplo, trans, @view(A[:, :, 1]), @view(C[:, :, 1]))
    lda = max(1, stride(A, 2))
    ldc = max(1, stride(C, 2))
    strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
    strideC = stride(C, 3)
    queue = oneMKL.global_queue(oneAPI.context(A), oneAPI.device(A))
    fname = _onemkl_syrk_fname(T)
    fname(oneAPI.sycl_queue(queue),
        uplo,
        trans,
        n,
        k,
        Ref(T(alpha)),
        A,
        lda,
        strideA,
        Ref(T(beta)),
        C,
        ldc,
        strideC,
        size(C, 3),
    )
    return C
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::oneAPI.oneArray{<:Any, 2},
                        B::oneAPI.oneArray{<:Any, 2},
                        beta,
                        C::oneAPI.oneArray{<:Any, 2};
                        compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx! is not supported on oneAPI"))
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:oneAPI.oneArray{<:Any,2}},
                              B::AbstractVector{<:oneAPI.oneArray{<:Any,2}},
                              beta,
                              C::AbstractVector{<:oneAPI.oneArray{<:Any,2}})
    length(A) == length(B) == length(C) || throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    oneMKL.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    return C
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::oneAPI.oneStridedArray{<:Any,3},
                              B::oneAPI.oneStridedArray{<:Any,3},
                              beta,
                              C::oneAPI.oneStridedArray{<:Any,3})
    oneMKL.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
    return C
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:oneAPI.oneArray{<:Any, 2}},
                                B::AbstractVector{<:oneAPI.oneArray{<:Any, 2}},
                                beta,
                                C::AbstractVector{<:oneAPI.oneArray{<:Any, 2}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is not supported on oneAPI"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::oneAPI.oneStridedArray{<:Any, 3},
                                B::oneAPI.oneStridedArray{<:Any, 3},
                                beta,
                                C::oneAPI.oneStridedArray{<:Any, 3};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is not supported on oneAPI"))
end

function NextLA.syrk!(uplo::Char,
                      trans::Char,
                      alpha,
                      A::oneAPI.oneStridedVecOrMat{<:Any},
                      beta,
                      C::oneAPI.oneStridedMatrix{<:Any})
    oneMKL.syrk!(uplo, trans, alpha, A, beta, C)
    return C
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AbstractVector{<:oneAPI.oneArray{<:Any,2}},
                              beta,
                              C::AbstractVector{<:oneAPI.oneArray{<:Any,2}})
    @warn "syrk_batched! falling back to batched gemm!" backend = "oneAPI" layout = :pointer maxlog=1
    return NextLA.gemm_batched!(trans, NextLA._syrk_batched_gemm_trans(trans), alpha, A, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::oneAPI.oneStridedArray{<:Any,3},
                              beta,
                              C::oneAPI.oneStridedArray{<:Any,3})
    _syrk_strided_batched_native!(uplo, trans, alpha, A, beta, C)
    return C
end

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

end
