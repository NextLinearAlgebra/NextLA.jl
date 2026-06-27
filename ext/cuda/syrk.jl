function NextLA.syrk!(uplo::Char,
                      trans::Char,
                      alpha,
                      A::CUDA.StridedCuMatrix{<:Any},
                      beta,
                      C::CUDA.StridedCuMatrix{<:Any})
    return CUBLAS.syrk!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AbstractVector{<:CUDA.CuArray{T,2}},
                              beta,
                              C::AbstractVector{<:CUDA.CuArray{T,2}}) where {T}
    @warn "syrk_batched! falling back to batched gemm!" backend = "CUDA" layout = :pointer maxlog=1
    return NextLA.gemm_batched!(trans, NextLA._syrk_batched_gemm_trans(trans), alpha, A, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::CUDA.StridedCuArray{<:Any,3},
                              beta,
                              C::CUDA.StridedCuArray{<:Any,3})
    @warn "syrk_batched! falling back to batched gemm!" backend = "CUDA" layout = :strided maxlog=1
    return NextLA.gemm_batched!(trans, NextLA._syrk_batched_gemm_trans(trans), alpha, A, A, beta, C)
end
