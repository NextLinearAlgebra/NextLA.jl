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
