# Symmetric rank-k update on CUDA.
#
# Source: origin/pr/19, by Alessandro <axelcar482@gmail.com>, carved out of
# that branch's monolithic ext/NextLACUDAExt.jl into this repo's per-backend
# ext layout. Only the @warn wording differs, for the reason in
# src/syrk/syrk.jl.
#
# CUBLAS has no batched SYRK entry point, so both batch layouts loop single
# syrk! calls rather than dispatching natively. That is a library gap, not an
# oversight here; see KNOWN_ISSUES.md.

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
                              A::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                              beta,
                              C::AbstractVector{<:CUDA.CuArray{<:Any,2}})
    @warn "syrk_batched! has no native batched SYRK here; looping syrk!" backend = "CUDA" layout = :pointer
    return NextLA._syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::CUDA.StridedCuArray{<:Any,3},
                              beta,
                              C::CUDA.StridedCuArray{<:Any,3})
    @warn "syrk_batched! has no native batched SYRK here; looping syrk!" backend = "CUDA" layout = :strided
    return NextLA._syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
end

