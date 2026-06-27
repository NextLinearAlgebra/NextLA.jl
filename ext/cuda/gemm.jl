function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:CUDA.CuArray{T,2}},
                              B::AbstractVector{<:CUDA.CuArray{T,2}},
                              beta,
                              C::AbstractVector{<:CUDA.CuArray{T,2}}) where {T}
    return CUBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              B::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              beta,
                              C::AbstractVector{<:CUDA.StridedCuMatrix{T}}) where {T}
    length(A) == length(B) == length(C) || throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    isempty(A) && return C

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A[1], B[1], C[1])

    Aptrs = _unsafe_batch_strided(A)
    Bptrs = _unsafe_batch_strided(B)
    Cptrs = _unsafe_batch_strided(C)

    try
        compute_enum = _cublas_compute_type(T)
        scalar_type = _cublas_scalar_type(T)
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

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::CUDA.StridedCuArray{<:Any,3},
                              B::CUDA.StridedCuArray{<:Any,3},
                              beta,
                              C::CUDA.StridedCuArray{<:Any,3})
    return CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched_ptrs!(transA::Char,
                                   transB::Char,
                                   alpha,
                                   Aptrs::CUDA.CuArray,
                                   Aref::AbstractMatrix{T},
                                   Bptrs::CUDA.CuArray,
                                   Bref::AbstractMatrix{T},
                                   beta,
                                   Cptrs::CUDA.CuArray,
                                   Cref::AbstractMatrix{T},
                                   batch_count::Integer) where {T}
    batch_count <= 0 && return Cptrs

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, Aref, Bref, Cref)
    compute_enum = _cublas_compute_type(T)
    scalar_type = _cublas_scalar_type(T)

    CUBLAS.cublasGemmBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        Aptrs, eltype(Aref), lda, Bptrs, eltype(Bref), ldb,
        CUDA.CuRef{scalar_type}(beta), Cptrs, eltype(Cref), ldc,
        Int(batch_count), compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return Cptrs
end

function NextLA.gemmEx_batched_ptrs!(transA::Char,
                                     transB::Char,
                                     alpha,
                                     Aptrs::CUDA.CuArray,
                                     Aref::AbstractMatrix,
                                     Bptrs::CUDA.CuArray,
                                     Bref::AbstractMatrix,
                                     beta,
                                     Cptrs::CUDA.CuArray,
                                     Cref::AbstractMatrix,
                                     batch_count::Integer;
                                     compute_type::Type=NextLA.default_compute_type(alpha, Aref, Bref, beta, Cref))
    batch_count <= 0 && return Cptrs

    NextLA._check_compute_type(compute_type)
    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, Aref, Bref, Cref)
    compute_enum = _cublas_compute_type(compute_type)
    scalar_type = _cublas_scalar_type(compute_type)

    CUBLAS.cublasGemmBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        Aptrs, eltype(Aref), lda, Bptrs, eltype(Bref), ldb,
        CUDA.CuRef{scalar_type}(beta), Cptrs, eltype(Cref), ldc,
        Int(batch_count), compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return Cptrs
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                                B::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                                beta,
                                C::AbstractVector{<:CUDA.CuArray{<:Any,2}})
    return CUBLAS.gemmBatchedEx!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:CUDA.StridedCuMatrix{<:Any}},
                                B::AbstractVector{<:CUDA.StridedCuMatrix{<:Any}},
                                beta,
                                C::AbstractVector{<:CUDA.StridedCuMatrix{<:Any}})
    return CUBLAS.gemmBatchedEx!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::CUDA.StridedCuArray{<:Any,3},
                                B::CUDA.StridedCuArray{<:Any,3},
                                beta,
                                C::CUDA.StridedCuArray{<:Any,3})
    return CUBLAS.gemmStridedBatchedEx!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::CUDA.CuArray{<:Any,2},
                        B::CUDA.CuArray{<:Any,2},
                        beta,
                        C::CUDA.CuArray{<:Any,2})
    return CUBLAS.gemmEx!(transA, transB, alpha, A, B, beta, C)
end
