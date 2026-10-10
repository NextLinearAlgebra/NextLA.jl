function NextLA.potrf_batched!(uplo::Char,
                              Av::AbstractVector{<:CUDA.StridedCuMatrix{T}}) where {T}
    eltype(Av) <: CUDA.CuArray && return CUSOLVER.potrfBatched!(uplo, Av)

    # CUSOLVER's batched-pointer path (`unsafe_batch`) only accepts literal
    # CuArray elements. A CUDA.jl view collapses to one only when contiguous;
    # a non-full-width slice of a padded workspace arena (e.g. ARA's
    # fixed-width Cholesky scratch) stays a lazy SubArray, so materialize a
    # compact copy, factorize it, and write the factors back into `Av`.
    Ac = CUDA.CuArray.(Av)
    _, info = CUSOLVER.potrfBatched!(uplo, Ac)
    foreach(copyto!, Av, Ac)
    return Av, info
end

function NextLA.potrf_batched!(uplo::Char,
                               A::CUDA.StridedCuArray{T,3}) where {T}
    Av = [@view A[:, :, bid] for bid in axes(A, 3)]
    return NextLA.potrf_batched!(uplo, Av)
end
