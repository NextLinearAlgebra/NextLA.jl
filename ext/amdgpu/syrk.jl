# Symmetric rank-k update on AMDGPU.
#
# Source: origin/pr/19, by Alessandro <axelcar482@gmail.com>, carved out of
# that branch's monolithic ext/NextLAAMDGPUExt.jl into this repo's per-backend
# ext layout. Verbatim but for alpha and beta, which are converted to the
# element type: rocBLAS's syrk takes a pointer to exactly that type. The
# _rocblas_syrk_fname table it uses went to common.jl beside the other
# rocBLAS name tables.
#
# rocBLAS is the only vendor here with a native batched SYRK in both the
# pointer and strided layouts, so nothing on this backend falls back.

function _syrk_native!(uplo::Char,
                       trans::Char,
                       alpha,
                       A::AMDGPU.StridedROCMatrix{<:Any},
                       beta,
                       C::AMDGPU.StridedROCMatrix{<:Any})
    n, k = NextLA._syrk_dims(uplo, trans, A, C)
    lda = max(1, stride(A, 2))
    ldc = max(1, stride(C, 2))
    fname = _rocblas_syrk_fname(eltype(A), Val(:single))
    T = eltype(C)
    fname(rocBLAS.handle(), uplo, trans, n, k, Ref(T(alpha)), A, lda, Ref(T(beta)), C, ldc)
    return C
end

function _syrk_batched_native!(uplo::Char,
                               trans::Char,
                               alpha,
                               A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                               beta,
                               C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}})
    length(A) == length(C) || throw(DimensionMismatch("syrk_batched!: matrix batches must have matching lengths"))
    isempty(A) && return C
    n, k = NextLA._syrk_dims(uplo, trans, A[1], C[1])
    lda = max(1, stride(A[1], 2))
    ldc = max(1, stride(C[1], 2))
    Aptrs = rocBLAS.device_batch(A)
    Cptrs = rocBLAS.device_batch(C)
    fname = _rocblas_syrk_fname(eltype(A[1]), Val(:batched))
    T = eltype(C[1])
    fname(rocBLAS.handle(), uplo, trans, n, k, Ref(T(alpha)), Aptrs, lda, Ref(T(beta)), Cptrs, ldc, length(C))
    return C
end

function _syrk_batched_native!(uplo::Char,
                               trans::Char,
                               alpha,
                               A::AMDGPU.StridedROCArray{<:Any, 3},
                               beta,
                               C::AMDGPU.StridedROCArray{<:Any, 3})
    size(A, 3) == size(C, 3) || size(A, 3) == 1 ||
        throw(DimensionMismatch("syrk_batched!: A and C batch sizes are incompatible"))
    n, k = NextLA._syrk_dims(uplo, trans, @view(A[:, :, 1]), @view(C[:, :, 1]))
    lda = max(1, stride(A, 2))
    ldc = max(1, stride(C, 2))
    strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
    strideC = stride(C, 3)
    fname = _rocblas_syrk_fname(eltype(A), Val(:strided))
    T = eltype(C)
    fname(rocBLAS.handle(), uplo, trans, n, k, Ref(T(alpha)), A, lda, strideA, Ref(T(beta)), C, ldc, strideC, size(C, 3))
    return C
end

function NextLA.syrk!(uplo::Char,
                      trans::Char,
                      alpha,
                      A::AMDGPU.StridedROCMatrix{<:Any},
                      beta,
                      C::AMDGPU.StridedROCMatrix{<:Any})
    return _syrk_native!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}},
                              beta,
                              C::AbstractVector{<:AMDGPU.ROCArray{<:Any, 2}})
    return _syrk_batched_native!(uplo, trans, alpha, A, beta, C)
end

function NextLA.syrk_batched!(uplo::Char,
                              trans::Char,
                              alpha,
                              A::AMDGPU.StridedROCArray{<:Any, 3},
                              beta,
                              C::AMDGPU.StridedROCArray{<:Any, 3})
    return _syrk_batched_native!(uplo, trans, alpha, A, beta, C)
end
