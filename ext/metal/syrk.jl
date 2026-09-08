# Symmetric rank-k update on Metal.
#
# Source: origin/pr/19, by Alessandro <axelcar482@gmail.com>, carved out of
# that branch's monolithic ext/NextLAMetalExt.jl into this repo's per-backend
# ext layout. Verbatim -- unlike the other three backends the @warn here is
# accurate, because Metal has no SYRK at all and both batch layouts really do
# route to gemm_batched!.
#
# The single-matrix path prefers MPS.matmul! where the element types are
# supported and otherwise falls back to mul!, which writes the full square
# rather than one triangle. Callers that rely on the untouched triangle should
# not use this backend; see KNOWN_ISSUES.md.

function NextLA.syrk!(uplo::Char,
                      trans::Char,
                      alpha,
                      A::Metal.MtlArray{<:Any, 2},
                      beta,
                      C::Metal.MtlArray{<:Any, 2})
    NextLA._syrk_dims(uplo, trans, A, C)
    if _supports_mps_matmul(eltype(A), eltype(A), eltype(C))
        return MPS.matmul!(
            C,
            A,
            A,
            alpha,
            beta,
            trans != 'N',
            trans == 'N',
        )
    end
    left = trans == 'N' ? A : trans == 'T' ? transpose(A) : trans == 'C' ? adjoint(A) :
        throw(ArgumentError("Unsupported transpose flag `$trans`"))
    right_trans = trans == 'N' ? 'T' : 'N'
    right = right_trans == 'N' ? A : transpose(A)
    return LinearAlgebra.mul!(
        C,
        left,
        right,
        alpha,
        beta,
    )
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::Metal.MtlArray{<:Any, 3},
                              beta,
                              C::Metal.MtlArray{<:Any, 3})
    NextLA._syrk_dims(uplo, trans, @view(A[:, :, 1]), @view(C[:, :, 1]))
    @warn "syrk_batched! falling back to batched gemm!" backend = "Metal" layout = :strided
    return NextLA.gemm_batched!(trans, trans == 'N' ? 'T' : 'N', alpha, A, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::MtlMatrixBatchView,
                              beta,
                              C::MtlMatrixBatchView)
    throw(ArgumentError("Metal batched SYRK does not support 3D MtlArray views; use dense MtlArray batches"))
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                              beta,
                              C::AbstractVector{<:Metal.MtlArray{<:Any, 2}})
    length(A) == length(C) || throw(DimensionMismatch("syrk_batched!: matrix batches must have matching lengths"))
    @warn "syrk_batched! falling back to batched gemm!" backend = "Metal" layout = :pointer
    isempty(A) || NextLA._syrk_dims(uplo, trans, A[1], C[1])
    return NextLA.gemm_batched!(trans, trans == 'N' ? 'T' : 'N', alpha, A, A, beta, C)
end
