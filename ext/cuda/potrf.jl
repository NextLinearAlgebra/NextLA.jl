function NextLA.potrf_batched!(uplo::Char,
                              Av::AbstractVector{<:CUDA.StridedCuMatrix{T}}) where {T}
    return CUSOLVER.potrfBatched!(uplo, Av)
end

function NextLA.potrf_batched!(uplo::Char,
                               A::CUDA.StridedCuArray{T,3}) where {T}
    Av = [@view A[:, :, bid] for bid in axes(A, 3)]
    return CUSOLVER.potrfBatched!(uplo, Av)
end
