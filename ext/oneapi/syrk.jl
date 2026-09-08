# Symmetric rank-k update on oneAPI.
#
# Source: origin/pr/19, by Alessandro <axelcar482@gmail.com>, carved out of
# that branch's monolithic ext/NextLAoneAPIExt.jl into this repo's per-backend
# ext layout. Only the @warn wording differs, for the reason in
# src/syrk/syrk.jl. The _onemkl_syrk_fname table went to common.jl beside the
# other oneMKL name tables.
#
# oneMKL has a native strided batched SYRK but no pointer-batched one, so the
# pointer layout loops single syrk! calls.

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
    @warn "syrk_batched! has no native batched SYRK here; looping syrk!" backend = "oneAPI" layout = :pointer
    return NextLA._syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
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

# recsyrk!'s Float16 updates (src/syrk/recsyrk.jl). oneAPI has no gemmEx!, so the
# multiply goes through gemm! -- mul! on the device -- as vicki-development's
# recsyrk_dev.jl did with oneMKL.gemm!. Never executed: no Intel GPU here.
NextLA._syrk_half_gemm!(alpha, A, B, beta, C::oneAPI.oneStridedMatrix) =
    NextLA.gemm!('N', 'T', alpha, A, B, beta, C)
