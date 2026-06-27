function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              B::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              alpha=one(T)) where {T}
    CUBLAS.trsm_batched!(side, uplo, transa, diag, alpha, A, B)
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
