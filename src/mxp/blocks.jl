@inline _is_block_precision(T) = T in (Float16, BFloat16, Float32, Float64) || nameof(T) === :Float8_E5M2

function _check_mixedprec_args(A::AbstractMatrix, precisions::Vector{DataType})
    isempty(A) && throw(ArgumentError("input matrix must not be empty"))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("input matrix must be square"))
    isempty(precisions) &&
        throw(ArgumentError("precisions must not be empty"))
    all(_is_block_precision, precisions) ||
        throw(ArgumentError("precisions must be Float8_E5M2, BFloat16, Float16, Float32 or Float64"))
    floatmax(precisions[end]) >= floatmax(Float32) ||
        throw(ArgumentError("the leaf precision, precisions[end], must be Float32 or Float64"))
    return nothing
end

function _quantize_mixedprec_block(B::AbstractMatrix, ::Type{T}) where {T<:AbstractFloat}
    blockamax = maximum(x -> abs(float(x)), B)
    isfinite(blockamax) || throw(ArgumentError("input matrix must be finite"))
    limit = convert(typeof(blockamax), floatmax(T))
    blockamax > limit || return (T.(B), 1f0)
    scale = Float32(blockamax / limit)
    scale * limit < blockamax && (scale = nextfloat(scale))
    isfinite(scale) ||
        throw(ArgumentError("A block of the input matrix has values too large for $T storage with a Float32 scale"))
    return (T.(B ./ convert(typeof(blockamax), scale)), scale)
end

function _requantize_mixedprec_offdiag_block!(block::AbstractMatrix{T}, W::AbstractMatrix{R}, scale::Float32) where {T<:AbstractFloat, R<:AbstractFloat}
    blockamax = maximum(abs, W)
    limit = R(floatmax(T))
    (isfinite(blockamax) && blockamax > limit) || (block .= W; return scale)
    growth = Float64(blockamax) / limit
    newscale = Float32(scale * growth)
    newscale < scale * growth && (newscale = nextfloat(newscale))
    isfinite(newscale) ||
        throw(ArgumentError("an update took a block of the input matrix past what $T storage with a Float32 scale can hold"))
    block .= W ./ R(Float64(newscale) / scale)
    return newscale
end

function _update_mixedprec_offdiag_block!(update!, block::AbstractMatrix{T}, scale::Float32) where {T<:AbstractFloat}
    floatmax(T) < floatmax(Float32) || (update!(block); return scale)
    P = similar(block, Float32, size(block))
    P .= block
    update!(P)
    return _requantize_mixedprec_offdiag_block!(block, P, scale)
end

function _solve_mixedprec_offdiag_block!(solve!, block::AbstractMatrix{T}, scale::Float32, ::Type{T_Base}) where {T<:AbstractFloat, T_Base<:AbstractFloat}
    T == T_Base && (solve!(block); return scale)
    W = similar(block, T_Base, size(block))
    W .= block
    solve!(W)
    return _requantize_mixedprec_offdiag_block!(block, W, scale)
end

function _triangular_mixedprec(::Type{C}, A::AbstractMatrix, uplo::Char, precisions::AbstractVector{DataType}) where {C}
    n = size(A, 1)
    T_Base = precisions[end]
    if length(precisions) == 1 || n == 1
        base, base_scale = _quantize_mixedprec_block(uplo == 'L' ? tril(A) : triu(A), T_Base)
        return C{T_Base}(nothing, nothing, nothing, 1f0, base_scale, base, uplo, n)
    end
    mid = _rec_split(n)
    remaining_precisions = @view precisions[2:end]
    A11 = _triangular_mixedprec(C, view(A, 1:mid, 1:mid), uplo, remaining_precisions)
    A22 = _triangular_mixedprec(C, view(A, mid+1:n, mid+1:n), uplo, remaining_precisions)
    offdiag_view = uplo == 'L' ? view(A, mid+1:n, 1:mid) : view(A, 1:mid, mid+1:n)
    OffDiag, OffDiag_scale = _quantize_mixedprec_block(offdiag_view, precisions[1])
    return C{T_Base}(A11, A22, OffDiag, OffDiag_scale, 1f0, nothing, uplo, n)
end
