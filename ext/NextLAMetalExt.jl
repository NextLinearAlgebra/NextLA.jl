module NextLAMetalExt

using NextLA
using Metal
using LinearAlgebra

const MPS = Metal.MPS
const MtlMatrixBatchView{T} = SubArray{T, 3, <:Metal.MtlArray{T, 3}, <:Any, false} where {T}

@inline NextLA.SUBGROUP_SIZE(::Type{<:Metal.MetalBackend}) = Val(64)

@inline function _supports_mps_batched_matmul(::Type{Tin}, ::Type{Tin}, ::Type{Tout}) where {Tin, Tout}
    return (Tin, Tout) in MPS.MPS_VALID_MATMUL_TYPES
end

@inline function _supports_mps_matmul(::Type{Tin}, ::Type{Tin}, ::Type{Tout}) where {Tin, Tout}
    return (Tin, Tout) in MPS.MPS_VALID_MATMUL_TYPES
end

@inline function _materialize_batch_view(A::MtlMatrixBatchView)
    dense = similar(parent(A), size(A))
    copyto!(dense, A)
    return dense
end

@inline _dense_mtl_batch(A::Metal.MtlArray{<:Any,3}) = A
@inline _dense_mtl_batch(A::MtlMatrixBatchView) = _materialize_batch_view(A)

@inline function _copyback_mtl_batch!(C::Metal.MtlArray{<:Any,3}, Cdense)
    return Cdense
end

@inline function _copyback_mtl_batch!(C::MtlMatrixBatchView, Cdense)
    copyto!(C, Cdense)
    return C
end

function _gemm_batched_mps!(transA::Char,
                            transB::Char,
                            alpha,
                            A::Metal.MtlArray{<:Any, 3},
                            B::Metal.MtlArray{<:Any, 3},
                            beta,
                            C::Metal.MtlArray{<:Any, 3})
    batchA = size(A, 3)
    batchB = size(B, 3)
    batchC = size(C, 3)
    (batchA == batchC || batchA == 1) || throw(DimensionMismatch("gemm_batched!: A and C batch sizes are incompatible"))
    (batchB == batchC || batchB == 1) || throw(DimensionMismatch("gemm_batched!: B and C batch sizes are incompatible"))

    if batchA == batchB == batchC &&
       _supports_mps_batched_matmul(eltype(A), eltype(B), eltype(C))
        return MPS.matmul!(
            C,
            A,
            B,
            alpha,
            beta,
            transA != 'N',
            transB != 'N',
        )
    end
    throw(ArgumentError("Metal batched GEMM requires equal batch sizes and an MPS-supported eltype combination"))
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::Metal.MtlArray{<:Any, 2},
                        B::Metal.MtlArray{<:Any, 2},
                        beta,
                        C::Metal.MtlArray{<:Any, 2};
                        compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx! is not supported on Metal"))
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::Metal.MtlArray{<:Any, 3},
                              B::Metal.MtlArray{<:Any, 3},
                              beta,
                              C::Metal.MtlArray{<:Any, 3})
    return _gemm_batched_mps!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::MtlMatrixBatchView,
                              B::MtlMatrixBatchView,
                              beta,
                              C::MtlMatrixBatchView)
    Adense = _dense_mtl_batch(A)
    Bdense = _dense_mtl_batch(B)
    Cdense = _dense_mtl_batch(C)
    NextLA.gemm_batched!(transA, transB, alpha, Adense, Bdense, beta, Cdense)
    return _copyback_mtl_batch!(C, Cdense)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView},
                              B::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView},
                              beta,
                              C::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView})
    Adense = _dense_mtl_batch(A)
    Bdense = _dense_mtl_batch(B)
    Cdense = _dense_mtl_batch(C)
    _gemm_batched_mps!(transA, transB, alpha, Adense, Bdense, beta, Cdense)
    return _copyback_mtl_batch!(C, Cdense)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::Metal.MtlArray{<:Any, 3},
                                B::Metal.MtlArray{<:Any, 3},
                                beta,
                                C::Metal.MtlArray{<:Any, 3};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is not supported on Metal"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::MtlMatrixBatchView,
                                B::MtlMatrixBatchView,
                                beta,
                                C::MtlMatrixBatchView;
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is not supported on Metal"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                                B::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                                beta,
                                C::AbstractVector{<:Metal.MtlArray{<:Any, 2}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx_batched! is not supported on Metal"))
end

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

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::Metal.MtlArray{<:Any,3},
                              B::Metal.MtlArray{<:Any,3},
                              alpha=one(eltype(A)))
    throw(ArgumentError("NextLA.trsm_batched! is not supported on Metal"))
end

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AbstractVector{<:Metal.MtlArray{<:Any,2}},
                              B::AbstractVector{<:Metal.MtlArray{<:Any,2}},
                              alpha=one(eltype(first(A))))
    throw(ArgumentError("NextLA.trsm_batched! is not supported on Metal"))
end

function NextLA.potrf_batched!(uplo::Char,
                               A::Metal.MtlArray{<:Any,3})
    throw(ArgumentError("NextLA.potrf_batched! is not supported on Metal"))
end

end
