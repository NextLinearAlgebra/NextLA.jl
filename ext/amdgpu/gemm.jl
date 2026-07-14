function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::AMDGPU.ROCArray{<:Any, 2},
                        B::AMDGPU.ROCArray{<:Any, 2},
                        beta,
                        C::AMDGPU.ROCArray{<:Any, 2};
                        compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if NextLA._supports_native_gemm(eltype(A), eltype(B), eltype(C)) && compute_type == eltype(C)
        return rocBLAS.gemm!(transA, transB, eltype(C)(alpha), A, B, eltype(C)(beta), C)
    end
    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A, B, C)
    alpha_ref = Ref{compute_type}(alpha)
    beta_ref = Ref{compute_type}(beta)

    rocBLAS.rocblas_gemm_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, A, _rocblas_datatype(eltype(A)), lda,
        B, _rocblas_datatype(eltype(B)), ldb,
        beta_ref, C, _rocblas_datatype(eltype(C)), ldc,
        C, _rocblas_datatype(eltype(C)), ldc,
        _rocblas_datatype(compute_type),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return C
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:AMDGPU.ROCArray{T, 2}},
                              B::AbstractVector{<:AMDGPU.ROCArray{T, 2}},
                              beta,
                              C::AbstractVector{<:AMDGPU.ROCArray{T, 2}}) where {T}
    return rocBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:AMDGPU.StridedROCMatrix{T}},
                              B::AbstractVector{<:AMDGPU.StridedROCMatrix{T}},
                              beta,
                              C::AbstractVector{<:AMDGPU.StridedROCMatrix{T}}) where {T}
    NextLA._check_batch_lengths(A, B, C)
    isempty(A) && return C

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A[1], B[1], C[1])

    scalar_type = T
    alpha_ref = Ref{scalar_type}(alpha)
    beta_ref = Ref{scalar_type}(beta)
    Aptrs = _device_batch_strided(A)
    Bptrs = _device_batch_strided(B)
    Cptrs = _device_batch_strided(C)
    rocBLAS.rocblas_gemm_batched_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, Aptrs, _rocblas_datatype(eltype(A[1])), lda,
        Bptrs, _rocblas_datatype(eltype(B[1])), ldb,
        beta_ref, Cptrs, _rocblas_datatype(eltype(C[1])), ldc,
        Cptrs, _rocblas_datatype(eltype(C[1])), ldc,
        length(C), _rocblas_datatype(T),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return C
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AMDGPU.StridedROCArray{<:Any, 3},
                              B::AMDGPU.StridedROCArray{<:Any, 3},
                              beta,
                              C::AMDGPU.StridedROCArray{<:Any, 3})
    if _supports_native_strided_batched(eltype(A), eltype(B), eltype(C))
        m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchC =
            NextLA._strided_batch_layout(transA, transB, A, B, C)
        alpha_ref = Ref{eltype(C)}(alpha)
        beta_ref = Ref{eltype(C)}(beta)

        rocBLAS.rocblas_gemm_strided_batched_ex_64(
            rocBLAS.handle(), transA, transB, m, n, k,
            alpha_ref, A, _rocblas_datatype(eltype(A)), lda, strideA,
            B, _rocblas_datatype(eltype(B)), ldb, strideB,
            beta_ref, C, _rocblas_datatype(eltype(C)), ldc, strideC,
            C, _rocblas_datatype(eltype(C)), ldc, strideC,
            batchC, _rocblas_datatype(eltype(C)),
            rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
        )
        return C
    elseif NextLA._supports_native_gemm(eltype(A), eltype(B), eltype(C))
        if A isa AMDGPU.ROCArray{<:Any, 3} &&
           B isa AMDGPU.ROCArray{<:Any, 3} &&
           C isa AMDGPU.ROCArray{<:Any, 3}
            return rocBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
        end
        throw(ArgumentError("AMDGPU strided batched GEMM views are supported only for eltypes with native strided-batched kernels"))
    end
    throw(ArgumentError("AMDGPU strided batched GEMM is not supported for eltypes $(eltype(A)), $(eltype(B)), and $(eltype(C))"))
end

function NextLA.gemm_batched_ptrs!(transA::Char,
                                   transB::Char,
                                   alpha,
                                   Aptrs::AMDGPU.ROCArray,
                                   Aref::AbstractMatrix{T},
                                   Bptrs::AMDGPU.ROCArray,
                                   Bref::AbstractMatrix{T},
                                   beta,
                                   Cptrs::AMDGPU.ROCArray,
                                   Cref::AbstractMatrix{T},
                                   batch_count::Integer) where {T}
    batch_count <= 0 && return Cptrs

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, Aref, Bref, Cref)
    scalar_type = T
    alpha_ref = Ref{scalar_type}(alpha)
    beta_ref = Ref{scalar_type}(beta)
    rocBLAS.rocblas_gemm_batched_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, Aptrs, _rocblas_datatype(eltype(Aref)), lda,
        Bptrs, _rocblas_datatype(eltype(Bref)), ldb,
        beta_ref, Cptrs, _rocblas_datatype(eltype(Cref)), ldc,
        Cptrs, _rocblas_datatype(eltype(Cref)), ldc,
        Int(batch_count), _rocblas_datatype(T),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return Cptrs
end

function NextLA.gemmEx_batched_ptrs!(transA::Char,
                                     transB::Char,
                                     alpha,
                                     Aptrs::AMDGPU.ROCArray,
                                     Aref::AbstractMatrix,
                                     Bptrs::AMDGPU.ROCArray,
                                     Bref::AbstractMatrix,
                                     beta,
                                     Cptrs::AMDGPU.ROCArray,
                                     Cref::AbstractMatrix,
                                     batch_count::Integer;
                                     compute_type::Type = NextLA.default_compute_type(alpha, Aref, Bref, beta, Cref))
    batch_count <= 0 && return Cptrs

    NextLA._check_compute_type(compute_type)
    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, Aref, Bref, Cref)
    scalar_type = compute_type
    alpha_ref = Ref{scalar_type}(alpha)
    beta_ref = Ref{scalar_type}(beta)
    rocBLAS.rocblas_gemm_batched_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, Aptrs, _rocblas_datatype(eltype(Aref)), lda,
        Bptrs, _rocblas_datatype(eltype(Bref)), ldb,
        beta_ref, Cptrs, _rocblas_datatype(eltype(Cref)), ldc,
        Cptrs, _rocblas_datatype(eltype(Cref)), ldc,
        Int(batch_count), _rocblas_datatype(compute_type),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return Cptrs
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                                B::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                                beta,
                                C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    return _gemmEx_batched_vector!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function _gemmEx_batched_vector!(transA::Char, transB::Char, alpha,
                                 A::AbstractVector{<:AMDGPU.StridedROCMatrix},
                                 B::AbstractVector{<:AMDGPU.StridedROCMatrix}, beta,
                                 C::AbstractVector{<:AMDGPU.StridedROCMatrix},
                                 compute_type::Type)
    NextLA._check_batch_lengths(A, B, C)
    isempty(A) && return C

    Aptrs = _device_batch_strided(A)
    Bptrs = _device_batch_strided(B)
    Cptrs = _device_batch_strided(C)
    NextLA.gemmEx_batched_ptrs!(
        transA, transB, alpha, Aptrs, A[1], Bptrs, B[1], beta, Cptrs, C[1], length(C);
        compute_type,
    )
    return C
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:AMDGPU.StridedROCMatrix{<:Any}},
                                B::AbstractVector{<:AMDGPU.StridedROCMatrix{<:Any}},
                                beta,
                                C::AbstractVector{<:AMDGPU.StridedROCMatrix{<:Any}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    return _gemmEx_batched_vector!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AMDGPU.StridedROCArray{<:Any, 3},
                                B::AMDGPU.StridedROCArray{<:Any, 3},
                                beta,
                                C::AMDGPU.StridedROCArray{<:Any, 3};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    if eltype(A) == eltype(B) == eltype(C) &&
       compute_type == eltype(C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchC =
        NextLA._strided_batch_layout(transA, transB, A, B, C)
    alpha_ref = Ref{compute_type}(alpha)
    beta_ref = Ref{compute_type}(beta)

    rocBLAS.rocblas_gemm_strided_batched_ex_64(
        rocBLAS.handle(), transA, transB, m, n, k,
        alpha_ref, A, _rocblas_datatype(eltype(A)), lda, strideA,
        B, _rocblas_datatype(eltype(B)), ldb, strideB,
        beta_ref, C, _rocblas_datatype(eltype(C)), ldc, strideC,
        C, _rocblas_datatype(eltype(C)), ldc, strideC,
        batchC, _rocblas_datatype(compute_type),
        rocBLAS.rocblas_gemm_algo_standard, Int32(0), rocBLAS.rocblas_gemm_flags_none,
    )
    return C
end
