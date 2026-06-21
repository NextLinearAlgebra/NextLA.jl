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

end
