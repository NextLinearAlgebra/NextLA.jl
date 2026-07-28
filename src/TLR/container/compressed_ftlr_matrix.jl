"""
    CompressedFTLRMatrix

Exact-rank, block-compressed low-rank storage for a regular tile grid.  Every
factor is a single packed allocation; `offsets` gives the scalar span of each
tile in that factor's own tile order.  The outer and inner factors may use
different orders, which is what permits row-packed `W` and column-packed `Z`.
"""
struct CompressedFTLRPackedFactors{AT<:AbstractVector,O<:TileOrderStyle}
    data::AT
    offsets::Vector{Int}                 # scalar offsets, 1-based, length ntile + 1
    order::O
    rows::Int
    qm::Int
    qn::Int
end

struct CompressedFTLRMatrix{BackendT<:Backend,T,AT<:AbstractVector{T},RankT<:Integer,
                  OuterT<:TileOrderStyle,InnerT<:TileOrderStyle,
                  OrderT<:TileOrderStyle} <: AbstractTLRMatrix{BackendT,T,OrderT}
    backend::BackendT
    order::OrderT                        # diagnostic rank-vector order
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}
    outer::CompressedFTLRPackedFactors{AT,OuterT}
    inner::CompressedFTLRPackedFactors{AT,InnerT}
    ranks::Vector{RankT}
    resid::Vector{Float64}
    maxrank::Int
end

@inline function _compressed_ftlr_rank(A::CompressedFTLRMatrix, i::Int, j::Int)
    return Int(A.ranks[_rank_index(A, i, j)])
end
@inline _compressed_ftlr_logical_coords(::CompressedFTLRMatrix, i::Int, j::Int) = (i, j)

@inline function _compressed_ftlr_slot(f::CompressedFTLRPackedFactors, i::Int, j::Int)
    return tile_linear_index(f.order, f.qm, f.qn, i, j)
end

@inline function _compressed_ftlr_factor_view(f::CompressedFTLRPackedFactors, r::Int, i::Int, j::Int)
    r == 0 && return reshape(view(f.data, 1:0), f.rows, 0)
    slot = _compressed_ftlr_slot(f, i, j)
    first = f.offsets[slot]
    last = f.offsets[slot + 1] - 1
    return reshape(view(f.data, first:last), f.rows, r)
end

@inline compressed_ftlr_outer(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.outer, _compressed_ftlr_rank(A, i, j), i, j)
@inline compressed_ftlr_inner(A::CompressedFTLRMatrix, i::Int, j::Int) =
    _compressed_ftlr_factor_view(A.inner, _compressed_ftlr_rank(A, i, j), i, j)
@inline compressed_ftlr_outer_order(A::CompressedFTLRMatrix) = A.outer.order
@inline compressed_ftlr_inner_order(A::CompressedFTLRMatrix) = A.inner.order

@inline function _compressed_ftlr_offsets(order::TileOrderStyle, qm::Int, qn::Int,
                                rows::Int, rank_at)
    offsets = Vector{Int}(undef, qm * qn + 1)
    offsets[1] = 1
    @inbounds for slot in 1:(qm * qn)
        i, j = inverse_tile_index(order, qm, qn, slot)
        offsets[slot + 1] = offsets[slot] + rows * rank_at(i, j)
    end
    return offsets
end

function _compressed_ftlr_rank_vector(order::TileOrderStyle, ranks_in::AbstractMatrix,
                           ::Type{RankT}) where {RankT<:Integer}
    qm, qn = size(ranks_in)
    ranks_out = Vector{RankT}(undef, qm * qn)
    @inbounds for j in 1:qn, i in 1:qm
        ranks_out[tile_linear_index(order, qm, qn, i, j)] = RankT(ranks_in[i, j])
    end
    return ranks_out
end

"""
    CompressedFTLRMatrix(backend, T, m, n, tile_size, ranks;
               outer_order=TileRowMajor, inner_order=TileColMajor,
               rank_type=Int32)

Allocate an exact-rank CompressedFTLR matrix. Initial CompressedFTLR support intentionally accepts
only a full regular tile grid; boundary tiles are a later extension.
"""
function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int,
                    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{<:Integer};
                    outer_order=TileRowMajor, inner_order=TileColMajor,
                    rank_type::Type{<:Integer}=Int32) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive"))
    m % bm == 0 && n % bn == 0 ||
        throw(ArgumentError("CompressedFTLRMatrix currently requires a full regular tile grid"))
    qm, qn = div(m, bm), div(n, bn)
    size(ranks_in) == (qm, qn) ||
        throw(DimensionMismatch("ranks must be a $qm × $qn matrix"))
    all(>=(0), ranks_in) || throw(ArgumentError("CompressedFTLR ranks must be nonnegative"))
    all(r -> r <= min(bm, bn), ranks_in) ||
        throw(ArgumentError("CompressedFTLR ranks must not exceed min(tile_size)"))

    outer_style = _order_instance(outer_order)
    inner_style = _order_instance(inner_order)
    rank_style = outer_style
    rank_at = (i, j) -> Int(ranks_in[i, j])
    uoffsets = _compressed_ftlr_offsets(outer_style, qm, qn, bm, rank_at)
    voffsets = _compressed_ftlr_offsets(inner_style, qm, qn, bn, rank_at)
    udata = zeros(backend, T, uoffsets[end] - 1)
    vdata = zeros(backend, T, voffsets[end] - 1)
    outer = CompressedFTLRPackedFactors(udata, uoffsets, outer_style, bm, qm, qn)
    inner = CompressedFTLRPackedFactors(vdata, voffsets, inner_style, bn, qm, qn)
    rankvec = _compressed_ftlr_rank_vector(rank_style, ranks_in, rank_type)
    resid = Base.zeros(Float64, qm * qn)
    return CompressedFTLRMatrix{typeof(backend),T,typeof(udata),rank_type,typeof(outer_style),
                      typeof(inner_style),typeof(rank_style)}(
        backend, rank_style, m, n, tile_size, (0, 0), outer, inner, rankvec, resid,
        isempty(rankvec) ? 0 : maximum(rankvec))
end

function CompressedFTLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
                    ranks_in::AbstractMatrix{<:Integer}; kwargs...) where {T}
    return CompressedFTLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end

"""Pack a padded, full-grid `PaddedFTLRMatrix` into exact-rank CompressedFTLR storage."""
function pack_compressed_ftlr(A::PaddedFTLRMatrix{<:Any,T}; outer_order=TileRowMajor,
                   inner_order=TileColMajor, rank_type::Type{<:Integer}=Int32) where {T}
    tail_tile_size(A) == (0, 0) ||
        throw(ArgumentError("pack_compressed_ftlr currently requires a full regular tile grid"))
    qm, qn = regular_grid_size(A)
    rank_grid = Matrix{Int}(undef, qm, qn)
    @inbounds for j in 1:qn, i in 1:qm
        rank_grid[i, j] = Int(ranks(A)[_rank_index(A, i, j)])
    end
    B = CompressedFTLRMatrix(get_backend(A), T, A.m, A.n, nominal_tile_size(A), rank_grid;
                   outer_order, inner_order, rank_type)
    @inbounds for j in 1:qn, i in 1:qm
        r = rank_grid[i, j]
        r == 0 && continue
        U, V = get_factors(A, i, j)
        copyto!(compressed_ftlr_outer(B, i, j), U)
        copyto!(compressed_ftlr_inner(B, i, j), V)
    end
    B.resid .= residuals(A)
    return B
end

@inline function get_factors(A::CompressedFTLRMatrix, i::Int, j::Int)
    return compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j)
end

@inline lowrank_regions(::CompressedFTLRMatrix) = (_INTERIOR,)
@inline outer_factors(A::CompressedFTLRMatrix, ::InteriorRegion) = A.outer
@inline inner_factors(A::CompressedFTLRMatrix, ::InteriorRegion) = A.inner
@inline region_tile_count(A::CompressedFTLRMatrix, ::InteriorRegion) = length(A.ranks)
@inline region_tile_coords(A::CompressedFTLRMatrix, ::InteriorRegion, k::Int) =
    inverse_tile_index(A.order, regular_grid_size(A)..., k)
