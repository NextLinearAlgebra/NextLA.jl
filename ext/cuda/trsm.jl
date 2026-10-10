function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              B::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              alpha=one(T)) where {T}
    length(A) == length(B) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    isempty(A) && return B
    nA = size(A[1], 1)
    size(A[1], 2) == nA || throw(DimensionMismatch("trsm_batched!: A must be square"))
    m, n = size(B[1])
    nA == (side == 'L' ? m : n) || throw(DimensionMismatch("trsm_batched!"))
    Aptrs = _unsafe_batch_strided(A)
    Bptrs = _unsafe_batch_strided(B)
    fname = if T === Float32
        CUBLAS.cublasStrsmBatched_64
    elseif T === Float64
        CUBLAS.cublasDtrsmBatched_64
    elseif T === ComplexF32
        CUBLAS.cublasCtrsmBatched_64
    elseif T === ComplexF64
        CUBLAS.cublasZtrsmBatched_64
    else
        throw(ArgumentError("unsupported CUDA batched TRSM element type $T"))
    end
    try
        fname(CUBLAS.handle(), side, uplo, transa, diag, m, n,
            CUDA.CuRef{T}(alpha), Aptrs, stride(A[1], 2), Bptrs,
            stride(B[1], 2), length(A))
    finally
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
    end
    return B
end

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::CUDA.StridedCuArray{T,3},
                              B::CUDA.StridedCuArray{T,3},
                              alpha=one(T)) where {T}
    size(A, 3) == size(B, 3) || throw(DimensionMismatch("trsm_batched!: matrix batches must have matching lengths"))
    Av = [@view A[:, :, bid] for bid in axes(A, 3)]
    Bv = [@view B[:, :, bid] for bid in axes(B, 3)]
    NextLA.trsm_batched!(side, uplo, transa, diag, Av, Bv, alpha)
    return B
end
