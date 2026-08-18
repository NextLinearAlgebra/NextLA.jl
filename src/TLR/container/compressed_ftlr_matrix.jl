"""
    CompressedFTLRMatrix

Logical-rank, block-compressed low-rank storage for a regular tile grid. Every
factor is a single packed allocation; `offsets` gives the scalar span of each
tile in that factor's own tile order.  The outer and inner factors may use
different orders, which is what permits row-packed `W` and column-packed `Z`.
"""
struct CompressedFTLRPackedFactors{AT<:AbstractVector,O<:TileOrderStyle}
    data::AT
    offsets::Vector{Int}                 # scalar offsets, 1-based, length ntile + 1
    order::O
    leading_dimensions::Vector{Int}
    logical_dimensions::Vector{Int}
    dimension_axis::Symbol
    qm::Int
    qn::Int
end

struct CompressedFTLRMatrix{BackendT<:Backend,T,AT<:AbstractVector{T},RankT<:Integer,
                  OuterT<:TileOrderStyle,InnerT<:TileOrderStyle} <: AbstractTLRMatrix{T}
    backend::BackendT
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}
    outer::CompressedFTLRPackedFactors{AT,OuterT}
    inner::CompressedFTLRPackedFactors{AT,InnerT}
    ranks::Vector{RankT}
    rank_multiple::Int
    resid::Vector{Float64}
    maxrank::Int
end

@inline function _compressed_ftlr_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    slot = tile_linear_index(A.outer.order, A.outer.qm, A.outer.qn, i, j)
    return Int(A.ranks[slot])
end
@inline _compressed_ftlr_rank(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_rank(parent(A), j, i)
@inline function _compressed_ftlr_storage_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    r = _compressed_ftlr_rank(A, i, j)
    return iszero(r) || iszero(A.rank_multiple) ? r : cld(r, A.rank_multiple) * A.rank_multiple
end
@inline _compressed_ftlr_storage_rank(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_storage_rank(parent(A), j, i)
@inline _compressed_ftlr_logical_coords(::CompressedFTLRMatrix, i::Int, j::Int) = (i, j)
@inline _compressed_ftlr_logical_coords(
    ::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) = (j, i)

@inline function _compressed_ftlr_factor_view(f::CompressedFTLRPackedFactors, stored_r::Int,
                                               visible_r::Int, i::Int, j::Int)
    axis_index = f.dimension_axis === :row ? i : j
    logical_rows = f.logical_dimensions[axis_index]
    stored_r == 0 && return reshape(view(f.data, 1:0), logical_rows, 0)
    slot = tile_linear_index(f.order, f.qm, f.qn, i, j)
    first = f.offsets[slot]
    last = f.offsets[slot + 1] - 1
    ld = f.leading_dimensions[axis_index]
    packed = reshape(view(f.data, first:last), ld, stored_r)
    return view(packed, 1:logical_rows, 1:visible_r)
end

@inline compressed_ftlr_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.outer, _compressed_ftlr_storage_rank(A, i, j),
                                 _compressed_ftlr_rank(A, i, j), i, j)
@inline compressed_ftlr_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.inner, _compressed_ftlr_storage_rank(A, i, j),
                                 _compressed_ftlr_rank(A, i, j), i, j)
@inline compressed_ftlr_outer(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_inner(parent(A), j, i)
@inline compressed_ftlr_inner(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_outer(parent(A), j, i)
@inline compressed_ftlr_storage_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.outer, _compressed_ftlr_storage_rank(A, i, j),
                                 _compressed_ftlr_storage_rank(A, i, j), i, j)
@inline compressed_ftlr_storage_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.inner, _compressed_ftlr_storage_rank(A, i, j),
                                 _compressed_ftlr_storage_rank(A, i, j), i, j)
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

@inline _compressed_ftlr_outer_storage(A::CompressedFTLRMatrix) = A.outer
@inline _compressed_ftlr_inner_storage(A::CompressedFTLRMatrix) = A.inner
@inline _compressed_ftlr_outer_storage(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}) = parent(A).inner
@inline _compressed_ftlr_inner_storage(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}) = parent(A).outer

"""
    _compressed_ftlr_uniform_view(f::CompressedFTLRPackedFactors)

Reinterpret `f`'s packed storage as a dense `[extent, width, qm*qn]` array,
where the third index is `f`'s own tile order
(`tile_linear_index(f.order, qm, qn, i, j)`). Valid only when every tile shares
the same stored capacity and logical extent along `f`'s axis — i.e. `f`
belongs to private uniform, regular-grid compressed-output staging. Throws if that
uniformity does not hold, rather than silently returning a misinterpreted view.
"""
function _compressed_ftlr_uniform_view(f::CompressedFTLRPackedFactors)
    ntiles = f.qm * f.qn
    ntiles == 0 && return reshape(view(f.data, 1:0), 0, 0, 0)
    ld = f.leading_dimensions[1]
    extent = f.logical_dimensions[1]
    (all(==(ld), f.leading_dimensions) && all(==(extent), f.logical_dimensions)) ||
        throw(ArgumentError("packed factors are not on a regular grid (nonuniform tile extent)"))
    stride = f.offsets[2] - f.offsets[1]
    stride % ld == 0 || throw(ArgumentError(
        "packed factor stride is not a multiple of its leading dimension"))
    capacity = stride ÷ ld
    all(k -> f.offsets[k + 1] - f.offsets[k] == stride, 1:ntiles) || throw(ArgumentError(
        "packed factors do not have a uniform per-tile width"))
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
               rank_multiple=0, rank_type=Int32)

Allocate a finalized CompressedFTLR matrix from known logical tile ranks.
Physical factor widths equal the logical ranks when `rank_multiple == 0` and
are otherwise rounded up to that positive multiple. A regular nominal grid may
have one trailing row and/or column tile.
"""

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int,
                    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{<:Integer};
                    outer_order=TileRowMajor, inner_order=TileColMajor,
                    rank_multiple::Integer=0,
                    rank_type::Type{<:Integer}=Int32) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive"))
    qm, qn = cld(m, bm), cld(n, bn)
    size(ranks_in) == (qm, qn) ||
        throw(DimensionMismatch("ranks must be a $qm × $qn matrix"))
    all(>=(0), ranks_in) || throw(ArgumentError("CompressedFTLR ranks must be nonnegative"))
    rank_multiple >= 0 || throw(ArgumentError("rank_multiple must be nonnegative"))
    multiple = Int(rank_multiple)
    rowdims = [min(bm, m - (i - 1) * bm) for i in 1:qm]
    coldims = [min(bn, n - (j - 1) * bn) for j in 1:qn]
    @inbounds for j in 1:qn, i in 1:qm
        ranks_in[i, j] <= min(rowdims[i], coldims[j]) ||
            throw(ArgumentError("CompressedFTLR rank at ($i, $j) exceeds its logical tile extent"))
    end

    outer_style = outer_order isa Type ? outer_order() : outer_order
    inner_style = inner_order isa Type ? inner_order() : inner_order
    storage_rank_at = function (i, j)
        r = Int(ranks_in[i, j])
        return iszero(r) || iszero(multiple) ? r : cld(r, multiple) * multiple
    end
    # Multiple-of-eight leading dimensions keep every packed factor base
    # 16-byte aligned, including ragged boundary tiles.
    uld = [cld(x, 8) * 8 for x in rowdims]
    vld = [cld(x, 8) * 8 for x in coldims]
    uoffsets = _compressed_ftlr_offsets(
        outer_style, qm, qn, (i, j) -> uld[i], storage_rank_at)
    voffsets = _compressed_ftlr_offsets(
        inner_style, qm, qn, (i, j) -> vld[j], storage_rank_at)
    udata = zeros(backend, T, uoffsets[end] - 1)
    vdata = zeros(backend, T, voffsets[end] - 1)
    outer = CompressedFTLRPackedFactors(udata, uoffsets, outer_style, uld, rowdims, :row, qm, qn)
    inner = CompressedFTLRPackedFactors(vdata, voffsets, inner_style, vld, coldims, :col, qm, qn)
    rankvec = Vector{rank_type}(undef, qm * qn)
    @inbounds for j in 1:qn, i in 1:qm
        rankvec[tile_linear_index(outer_style, qm, qn, i, j)] = rank_type(ranks_in[i, j])
    end
    resid = Base.zeros(Float64, qm * qn)
    return CompressedFTLRMatrix{typeof(backend),T,typeof(udata),rank_type,typeof(outer_style),
                      typeof(inner_style)}(
        backend, m, n, tile_size, (m % bm, n % bn), outer, inner,
        rankvec, multiple, resid, isempty(rankvec) ? 0 : maximum(rankvec))
end

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
                    ranks_in::AbstractMatrix{<:Integer}; kwargs...) where {T}
    return CompressedFTLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end

@inline function get_factors(A::CompressedFTLRMatrix, i::Int, j::Int)
    return compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j)
end
@inline get_factors(
    A::TransposeTLRMatrix{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    (compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j))
