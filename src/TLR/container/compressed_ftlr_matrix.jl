"""
    CompressedFTLRPackedFactors

One packed factor allocation for a rectangular tile grid. `offsets` gives each
tile's scalar span in this factor's tile order.
"""
struct CompressedFTLRPackedFactors{AT<:AbstractVector,O<:TileOrderStyle}
    data::AT
    offsets::Vector{Int}                 # scalar offsets, 1-based, length ntile + 1
    order::O
    logical_dimensions::Vector{Int}
    qm::Int
    qn::Int
end

"""Logical-rank block-compressed storage with independently ordered factors."""
struct CompressedFTLRMatrix{BackendT<:Backend,T,AT<:AbstractVector{T},
                  OuterT<:TileOrderStyle,InnerT<:TileOrderStyle} <: AbstractTLRMatrix{T}
    backend::BackendT
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}
    outer::CompressedFTLRPackedFactors{AT,OuterT}
    inner::CompressedFTLRPackedFactors{AT,InnerT}
    ranks::Vector{Int}
    rank_multiple::Int
    resid::Vector{Float64}
    maxrank::Int
end

@inline function compressed_ftlr_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    slot = tile_linear_index(A.outer.order, A.outer.qm, A.outer.qn, i, j)
    return A.ranks[slot]
end
@inline compressed_ftlr_rank(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_rank(parent(A), j, i)
@inline function compressed_ftlr_storage_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    r = compressed_ftlr_rank(A, i, j)
    return iszero(r) || iszero(A.rank_multiple) ? r : cld(r, A.rank_multiple) * A.rank_multiple
end
@inline compressed_ftlr_storage_rank(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_storage_rank(parent(A), j, i)
@inline compressed_ftlr_logical_coords(::CompressedFTLRMatrix, i::Int, j::Int) = (i, j)
@inline compressed_ftlr_logical_coords(
    ::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) = (j, i)

@inline function compressed_ftlr_factor_view(f::CompressedFTLRPackedFactors,
                                               i::Int, j::Int, axis_index::Int,
                                               visible_r=nothing)
    logical_rows = f.logical_dimensions[axis_index]
    slot = tile_linear_index(f.order, f.qm, f.qn, i, j)
    first = f.offsets[slot]
    next = f.offsets[slot + 1]
    ld = aligned_leading_dimension(eltype(f.data), logical_rows)
    stored_r = (next - first) ÷ ld
    width = visible_r === nothing ? stored_r : visible_r

    stored_r == 0 && return reshape(view(f.data, 1:0), logical_rows, 0)
    packed = reshape(view(f.data, first:(next - 1)), ld, stored_r)
    return view(packed, 1:logical_rows, 1:width)
end

@inline compressed_ftlr_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    compressed_ftlr_factor_view(A.outer, i, j, i, compressed_ftlr_rank(A, i, j))
@inline compressed_ftlr_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    compressed_ftlr_factor_view(A.inner, i, j, j, compressed_ftlr_rank(A, i, j))
@inline compressed_ftlr_outer(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_inner(parent(A), j, i)
@inline compressed_ftlr_inner(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_outer(parent(A), j, i)
@inline compressed_ftlr_storage_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    compressed_ftlr_factor_view(A.outer, i, j, i)
@inline compressed_ftlr_storage_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    compressed_ftlr_factor_view(A.inner, i, j, j)
@inline compressed_ftlr_storage_outer(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_storage_inner(parent(A), j, i)
@inline compressed_ftlr_storage_inner(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_storage_outer(parent(A), j, i)
@inline compressed_ftlr_outer_order(A::CompressedFTLRMatrix) = A.outer.order
@inline compressed_ftlr_inner_order(A::CompressedFTLRMatrix) = A.inner.order
@inline compressed_ftlr_outer_order(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}) =
    transpose(parent(A).inner.order)
@inline compressed_ftlr_inner_order(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}) =
    transpose(parent(A).outer.order)
@inline tile_order(A::CompressedFTLRMatrix) = A.outer.order
@inline rank_multiple(A::CompressedFTLRMatrix) = A.rank_multiple
@inline rank_multiple(A::TransposeTLRMatrix) = rank_multiple(parent(A))
@inline function maximum_storage_rank(A::CompressedFTLRMatrix)
    r, multiple = maxrank(A), rank_multiple(A)
    return iszero(r) || iszero(multiple) ? r : cld(r, multiple) * multiple
end
@inline maximum_storage_rank(A::TransposeTLRMatrix) = maximum_storage_rank(parent(A))

@inline compressed_ftlr_outer_storage(A::CompressedFTLRMatrix) = A.outer
@inline compressed_ftlr_inner_storage(A::CompressedFTLRMatrix) = A.inner
@inline compressed_ftlr_outer_storage(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}) = parent(A).inner
@inline compressed_ftlr_inner_storage(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}) = parent(A).outer

"""
    compressed_ftlr_uniform_view(f::CompressedFTLRPackedFactors)

View uniformly packed factors as `[extent, width, qm*qn]` in `f.order`. This is
valid only for private regular-grid staging and throws on nonuniform extents,
capacities, or backing storage.
"""
function compressed_ftlr_uniform_view(f::CompressedFTLRPackedFactors)
    ntiles = f.qm * f.qn
    extent = f.logical_dimensions[1]

    # regular-grid extent
    all(==(extent), f.logical_dimensions) ||
        throw(ArgumentError("packed factors are not on a regular grid (nonuniform tile extent)"))

    # uniform tile capacity
    ld = aligned_leading_dimension(eltype(f.data), extent)
    stride = f.offsets[2] - f.offsets[1]
    stride % ld == 0 || throw(ArgumentError(
        "packed factor stride is not a multiple of its leading dimension"))
    capacity = stride ÷ ld
    all(k -> f.offsets[k + 1] - f.offsets[k] == stride, 1:ntiles) || throw(ArgumentError(
        "packed factors do not have a uniform per-tile width"))

    # backing size
    length(f.data) == stride * ntiles || throw(ArgumentError(
        "packed factor storage size is inconsistent with a uniform-capacity layout"))
    return view(reshape(f.data, ld, capacity, ntiles), 1:extent, :, :)
end

@inline function _compressed_ftlr_offsets(order::TileOrderStyle, qm::Int, qn::Int,
                                leading_dimension_at, rank_at)
    offsets = Vector{Int}(undef, qm * qn + 1)
    offsets[1] = 1
    @inbounds for slot in 1:(qm * qn)
        i, j = inverse_tile_index(order, qm, qn, slot)
        offsets[slot + 1] = offsets[slot] + leading_dimension_at(i, j) * rank_at(i, j)
    end
    return offsets
end

"""
    CompressedFTLRMatrix(backend, T, m, n, tile_size, ranks;
               outer_order=TileRowMajor, inner_order=TileColMajor,
               rank_multiple=0)

Allocate a finalized CompressedFTLR matrix from known logical tile ranks.
Physical factor widths equal the logical ranks when `rank_multiple == 0` and
are otherwise rounded up to that positive multiple. A regular nominal grid may
have one trailing row and/or column tile.
"""

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int,
                    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{Int};
                    outer_order=TileRowMajor, inner_order=TileColMajor,
                    rank_multiple::Int=0) where {T}
    # argument checks
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive"))
    qm, qn = cld(m, bm), cld(n, bn)
    size(ranks_in) == (qm, qn) ||
        throw(DimensionMismatch("ranks must be a $qm × $qn matrix"))
    all(>=(0), ranks_in) || throw(ArgumentError("CompressedFTLR ranks must be nonnegative"))
    rank_multiple >= 0 || throw(ArgumentError("rank_multiple must be nonnegative"))
    rowdims = [min(bm, m - (i - 1) * bm) for i in 1:qm]
    coldims = [min(bn, n - (j - 1) * bn) for j in 1:qn]
    @inbounds for j in 1:qn, i in 1:qm
        ranks_in[i, j] <= min(rowdims[i], coldims[j]) ||
            throw(ArgumentError("CompressedFTLR rank at ($i, $j) exceeds its logical tile extent"))
    end

    # packed offsets and backing storage
    outer_style = outer_order isa Type ? outer_order() : outer_order
    inner_style = inner_order isa Type ? inner_order() : inner_order
    storage_rank_at = function (i, j)
        r = ranks_in[i, j]
        return iszero(r) || iszero(rank_multiple) ? r : cld(r, rank_multiple) * rank_multiple
    end

    # leading dimensions for 16-byte-aligned factor bases, including ragged tiles
    uoffsets = _compressed_ftlr_offsets(
        outer_style, qm, qn,
        (i, j) -> aligned_leading_dimension(T, rowdims[i]), storage_rank_at)
    voffsets = _compressed_ftlr_offsets(
        inner_style, qm, qn,
        (i, j) -> aligned_leading_dimension(T, coldims[j]), storage_rank_at)
    udata = zeros(backend, T, uoffsets[end] - 1)
    vdata = zeros(backend, T, voffsets[end] - 1)
    outer = CompressedFTLRPackedFactors(udata, uoffsets, outer_style, rowdims, qm, qn)
    inner = CompressedFTLRPackedFactors(vdata, voffsets, inner_style, coldims, qm, qn)

    # diagnostic rank vector, in the outer factor's own tile order
    rankvec = Vector{Int}(undef, qm * qn)
    @inbounds for j in 1:qn, i in 1:qm
        rankvec[tile_linear_index(outer_style, qm, qn, i, j)] = ranks_in[i, j]
    end

    return CompressedFTLRMatrix{typeof(backend),T,typeof(udata),typeof(outer_style),
                      typeof(inner_style)}(
        backend, m, n, tile_size, (m % bm, n % bn), outer, inner,
        rankvec, rank_multiple, Base.zeros(Float64, qm * qn), maximum(rankvec))
end

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
                    ranks_in::AbstractMatrix{Int}; kwargs...) where {T}
    return CompressedFTLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end

@inline get_factors(A::CompressedFTLRMatrix, i::Int, j::Int) =
    (compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j))
@inline get_factors(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    (compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j))
