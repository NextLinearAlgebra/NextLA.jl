module NextLACUDAExt

using NextLA
using CUDA

const CUBLAS = CUDA.CUBLAS
const NATIVE_GEMM_TYPES = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

@inline _cublas_compute_type(::Type{Float16}) = CUBLAS.CUBLAS_COMPUTE_16F
@inline _cublas_compute_type(::Type{Float32}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{Float64}) = CUBLAS.CUBLAS_COMPUTE_64F
@inline _cublas_compute_type(::Type{ComplexF32}) = CUBLAS.CUBLAS_COMPUTE_32F
@inline _cublas_compute_type(::Type{ComplexF64}) = CUBLAS.CUBLAS_COMPUTE_64F
@inline _cublas_compute_type(::Type{Int32}) = CUBLAS.CUBLAS_COMPUTE_32I

@inline _cublas_scalar_type(::Type{Float16}) = Float16
@inline _cublas_scalar_type(::Type{Float32}) = Float32
@inline _cublas_scalar_type(::Type{Float64}) = Float64
@inline _cublas_scalar_type(::Type{ComplexF32}) = ComplexF32
@inline _cublas_scalar_type(::Type{ComplexF64}) = ComplexF64
@inline _cublas_scalar_type(::Type{Int32}) = Int32

@inline _supports_native_gemm(::Type{T}, ::Type{T}, ::Type{T}) where {T<:NATIVE_GEMM_TYPES} = true
@inline _supports_native_gemm(::Type, ::Type, ::Type) = false

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                              B::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                              beta,
                              C::AbstractVector{<:CUDA.CuArray{<:Any,2}})
    return CUBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::CUDA.StridedCuArray{<:Any,3},
                              B::CUDA.StridedCuArray{<:Any,3},
                              beta,
                              C::CUDA.StridedCuArray{<:Any,3})
    return CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                                B::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                                beta,
                                C::AbstractVector{<:CUDA.CuArray{<:Any,2}};
                                compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if compute_type == NextLA.default_compute_type(alpha, A, B, beta, C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    return _gemm_batched_ex!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::CUDA.StridedCuArray{<:Any,3},
                                B::CUDA.StridedCuArray{<:Any,3},
                                beta,
                                C::CUDA.StridedCuArray{<:Any,3};
                                compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if compute_type == NextLA.default_compute_type(alpha, A, B, beta, C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    return _gemm_strided_batched_ex!(transA, transB, alpha, A, B, beta, C, compute_type)
end

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
    @warn "syrk_batched! falling back to batched gemm!" backend = "CUDA" layout = :pointer
    return NextLA._syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::CUDA.StridedCuArray{<:Any,3},
                              beta,
                              C::CUDA.StridedCuArray{<:Any,3})
    @warn "syrk_batched! falling back to batched gemm!" backend = "CUDA" layout = :strided
    return NextLA._syrk_batched_fallback!(uplo, trans, alpha, A, beta, C)
end

#Ex versions 
function _gemm_ex!(transA::Char,
                   transB::Char,
                   alpha,
                   A::CUDA.CuArray{<:Any,2},
                   B::CUDA.CuArray{<:Any,2},
                   beta,
                   C::CUDA.CuArray{<:Any,2},
                   compute_type::Type)
    m, n, k, lda, ldb, ldc = _gemm_dims(transA, transB, A, B, C)
    compute_enum = _cublas_compute_type(compute_type)
    scalar_type = _cublas_scalar_type(compute_type)

    CUBLAS.cublasGemmEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        A, eltype(A), lda, B, eltype(B), ldb, CUDA.CuRef{scalar_type}(beta),
        C, eltype(C), ldc, compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return C
end

function _gemm_batched_ex!(transA::Char,
                           transB::Char,
                           alpha,
                           A::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                           B::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                           beta,
                           C::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                           compute_type::Type)
    length(A) == length(B) == length(C) || throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    isempty(A) && return C

    m, n, k, lda, ldb, ldc = _gemm_dims(transA, transB, A[1], B[1], C[1])
    for i in eachindex(A, B, C)
        _gemm_dims(transA, transB, A[i], B[i], C[i]) == (m, n, k, lda, ldb, ldc) ||
            throw(DimensionMismatch("gemm_batched!: all matrices in a batch must share dimensions and strides"))
    end

    compute_enum = _cublas_compute_type(compute_type)
    scalar_type = _cublas_scalar_type(compute_type)
    Aptrs = CUBLAS.unsafe_batch(A)
    Bptrs = CUBLAS.unsafe_batch(B)
    Cptrs = CUBLAS.unsafe_batch(C)

    try
        CUBLAS.cublasGemmBatchedEx(
            CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
            Aptrs, eltype(A[1]), lda, Bptrs, eltype(B[1]), ldb,
            CUDA.CuRef{scalar_type}(beta), Cptrs, eltype(C[1]), ldc,
            length(A), compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
        )
    finally
        CUDA.unsafe_free!(Cptrs)
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
    end

    return C
end

function _gemm_strided_batched_ex!(transA::Char,
                                   transB::Char,
                                   alpha,
                                   A::CUDA.StridedCuArray{<:Any,3},
                                   B::CUDA.StridedCuArray{<:Any,3},
                                   beta,
                                   C::CUDA.StridedCuArray{<:Any,3},
                                   compute_type::Type)
    batchA = size(A, 3)
    batchB = size(B, 3)
    batchC = size(C, 3)
    (batchA == batchC || batchA == 1) || throw(DimensionMismatch("gemm_batched!: A and C batch sizes are incompatible"))
    (batchB == batchC || batchB == 1) || throw(DimensionMismatch("gemm_batched!: B and C batch sizes are incompatible"))

    m, n, k, lda, ldb, ldc = _gemm_dims(transA, transB, @view(A[:, :, 1]), @view(B[:, :, 1]), @view(C[:, :, 1]))
    strideA = batchA == 1 ? 0 : stride(A, 3)
    strideB = batchB == 1 ? 0 : stride(B, 3)
    strideC = stride(C, 3)
    compute_enum = _cublas_compute_type(compute_type)
    scalar_type = _cublas_scalar_type(compute_type)

    CUBLAS.cublasGemmStridedBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        A, eltype(A), lda, strideA, B, eltype(B), ldb, strideB,
        CUDA.CuRef{scalar_type}(beta), C, eltype(C), ldc, strideC,
        batchC, compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return C
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::CUDA.CuArray{<:Any,2},
                        B::CUDA.CuArray{<:Any,2},
                        beta,
                        C::CUDA.CuArray{<:Any,2};
                        compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if compute_type == NextLA.default_compute_type(alpha, A, B, beta, C) &&
       _supports_native_gemm(eltype(A), eltype(B), eltype(C))
        return CUBLAS.gemm!(transA, transB, alpha, A, B, beta, C)
    end
    return _gemm_ex!(transA, transB, alpha, A, B, beta, C, compute_type)
end

end
