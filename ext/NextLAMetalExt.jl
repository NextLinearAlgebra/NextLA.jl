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
    throw(ArgumentError("Metal batched GEMM does not support 3D MtlArray views; use dense MtlArray batches"))
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

end
