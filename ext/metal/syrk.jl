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
    @warn "syrk_batched! falling back to batched gemm!" backend = "Metal" layout = :strided maxlog=1
    return NextLA.gemm_batched!(trans, trans == 'N' ? 'T' : 'N', alpha, A, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::MtlMatrixBatchView,
                              beta,
                              C::MtlMatrixBatchView)
    Adense = _dense_mtl_batch(A)
    Cdense = _dense_mtl_batch(C)
    NextLA.syrk_batched!(uplo, trans, alpha, Adense, beta, Cdense)
    return _copyback_mtl_batch!(C, Cdense)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView},
                              beta,
                              C::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView})
    Adense = _dense_mtl_batch(A)
    Cdense = _dense_mtl_batch(C)
    @warn "syrk_batched! falling back to batched gemm!" backend = "Metal" layout = :strided maxlog=1
    NextLA.gemm_batched!(trans, trans == 'N' ? 'T' : 'N', alpha, Adense, Adense, beta, Cdense)
    return _copyback_mtl_batch!(C, Cdense)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                              beta,
                              C::AbstractVector{<:Metal.MtlArray{<:Any, 2}})
    length(A) == length(C) || throw(DimensionMismatch("syrk_batched!: matrix batches must have matching lengths"))
    @warn "syrk_batched! falling back to batched gemm!" backend = "Metal" layout = :pointer maxlog=1
    isempty(A) || NextLA._syrk_dims(uplo, trans, A[1], C[1])
    return NextLA.gemm_batched!(trans, trans == 'N' ? 'T' : 'N', alpha, A, A, beta, C)
end
