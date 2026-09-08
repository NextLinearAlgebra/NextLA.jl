@inline _supported(T) = T in (Float16, Core.BFloat16, Float32, Float64) || nameof(T) === :Float8_E5M2

function _check_args(A::AbstractMatrix, precisions::Vector{DataType}, ::Type{S}) where {S}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    isempty(precisions) &&
        throw(ArgumentError("precisions must not be empty"))
    all(_supported, precisions) ||
        throw(ArgumentError("precisions must be Float8_E5M2, BFloat16, Float16, Float32 or Float64"))
    S in (Float16, Float32, Float64) ||
        throw(ArgumentError("scale_type must be Float16, Float32 or Float64"))
    isempty(A) && throw(ArgumentError("A must not be empty"))
    return nothing
end

function _store(B::AbstractMatrix, ::Type{T}, ::Type{S}) where {T<:AbstractFloat, S<:AbstractFloat}
    alpha = float(maximum(abs, B))
    isfinite(alpha) || throw(ArgumentError("A must be finite"))
    limit = convert(typeof(alpha), floatmax(T))
    alpha > limit || return (T.(B), one(S))

    scale = S(alpha / limit)
    scale * limit < alpha && (scale = nextfloat(scale))
    isfinite(scale) ||
        throw(ArgumentError("A block of A has values too large for $T storage with a $S scale"))
    return (T.(clamp.(B ./ convert(typeof(alpha), scale), -limit, limit)), scale)
end

@inline _unscaled(x, s, ::Type{T}) where {T} = T(x * T(s))

# Stores W -- a block's values over its current scale s, held in a type with more
# range than the block's -- back into block, and returns the block's scale. Values
# within floatmax of the block's type keep s; larger ones take a scale raised just
# enough that none is clipped. A non-finite W, or a scale past floatmax of its own
# type, is stored as it is under s, so such an overflow still gives Inf.
function _requantize!(block::AbstractMatrix, W::AbstractMatrix, s::S) where {S<:AbstractFloat}
    R = eltype(W)
    limit = R(floatmax(eltype(block)))
    alpha = maximum(abs, W)
    if isfinite(alpha) && alpha > limit
        s_new = S(Float64(s) * Float64(alpha) / Float64(limit))
        Float64(s_new) * Float64(limit) < Float64(s) * Float64(alpha) && (s_new = nextfloat(s_new))
        if isfinite(s_new)
            g = R(Float64(s_new) / Float64(s))
            block .= clamp.(W ./ g, -limit, limit)
            return s_new
        end
    end
    block .= W
    return s
end

# Runs update!(P) on a block's stored values and returns the block's scale. A block
# with less range than Float32 (Float16, Float8) is updated in a Float32 copy P and
# stored back through _requantize!, so an update past floatmax of its type raises its
# scale instead of leaving Inf; a wider block is updated in place and keeps its scale.
function _update_block!(update!, block::AbstractMatrix, scale)
    Float64(floatmax(eltype(block))) < Float64(floatmax(Float32)) || (update!(block); return scale)
    P = similar(block, Float32, size(block))
    P .= block
    update!(P)
    return _requantize!(block, P, scale)
end

@inline _stored(i::Int, j::Int, uplo::Char) = uplo == 'L' ? i >= j : i <= j

function _triangular_tree(::Type{C}, A::AbstractMatrix, uplo::Char, precisions::AbstractVector{DataType}, ::Type{S}) where {C, S<:AbstractFloat}
    n = size(A, 1)
    T_Base = precisions[end]
    unit_scale = one(S)
    if length(precisions) == 1 || n == 1
        base, base_scale = _store(uplo == 'L' ? tril(A) : triu(A), T_Base, S)
        return C{T_Base, S}(
            nothing, nothing, nothing,
            unit_scale, base_scale, base, uplo, n
        )
    end
    mid = _rec_split(n)
    remaining_precisions = @view precisions[2:end]
    A11 = _triangular_tree(C, view(A, 1:mid, 1:mid), uplo, remaining_precisions, S)
    A22 = _triangular_tree(C, view(A, mid+1:n, mid+1:n), uplo, remaining_precisions, S)
    offdiag = uplo == 'L' ? view(A, mid+1:n, 1:mid) : view(A, 1:mid, mid+1:n)
    OffDiag, OffDiag_scale = _store(offdiag, precisions[1], S)
    return C{T_Base, S}(
        A11, A22, OffDiag,
        OffDiag_scale, unit_scale, nothing, uplo, n
    )
end

# Solves an off-diagonal block in place against a factored diagonal block and
# returns the block's scale. The solve is linear, so it runs on the stored values
# under that scale. The diagonal block is T_Base and the solve takes one element
# type, so a narrower block is solved in a T_Base copy and stored back through
# _requantize!, which raises the scale if the solution outgrew the block's type.
function _solve_block!(solve!, block::AbstractMatrix, scale, ::Type{T_Base}) where {T_Base}
    eltype(block) == T_Base && (solve!(block); return scale)
    W = similar(block, T_Base, size(block))
    W .= block
    solve!(W)
    return _requantize!(block, W, scale)
end
