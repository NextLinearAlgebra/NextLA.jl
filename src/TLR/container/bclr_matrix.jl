"""
    BCLRMatrix

Exact-rank, block-compressed low-rank storage for a regular tile grid.  Every
factor is a single packed allocation; `offsets` gives the scalar span of each
tile in that factor's own tile order.  The outer and inner factors may use
different orders, which is what permits row-packed `W` and column-packed `Z`.
"""
struct BCLRPackedFactors{AT<:AbstractVector,O<:TileOrderStyle}
    data::AT
    offsets::Vector{Int}                 # scalar offsets, 1-based, length ntile + 1
    order::O
    rows::Int
    qm::Int
    qn::Int
end

struct BCLRMatrix{BackendT<:Backend,T,AT<:AbstractVector{T},RankT<:Integer,
                  OuterT<:TileOrderStyle,InnerT<:TileOrderStyle,
                  OrderT<:TileOrderStyle} <: AbstractTLRMatrix{BackendT,T,OrderT}
    backend::BackendT
    order::OrderT                        # diagnostic rank-vector order
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}
    outer::BCLRPackedFactors{AT,OuterT}
    inner::BCLRPackedFactors{AT,InnerT}
    ranks::Vector{RankT}
    resid::Vector{Float64}
    maxrank::Int
end

@inline function _bclr_rank(A::BCLRMatrix, i::Int, j::Int)
    return Int(A.ranks[_rank_index(A, i, j)])
end
@inline _bclr_logical_coords(::BCLRMatrix, i::Int, j::Int) = (i, j)

@inline function _bclr_slot(f::BCLRPackedFactors, i::Int, j::Int)
    return tile_linear_index(f.order, f.qm, f.qn, i, j)
end

@inline function _bclr_factor_view(f::BCLRPackedFactors, r::Int, i::Int, j::Int)
    r == 0 && return reshape(view(f.data, 1:0), f.rows, 0)
    slot = _bclr_slot(f, i, j)
    first = f.offsets[slot]
    last = f.offsets[slot + 1] - 1
    return reshape(view(f.data, first:last), f.rows, r)
end

@inline bclr_outer(A::BCLRMatrix, i::Int, j::Int) =
    _bclr_factor_view(A.outer, _bclr_rank(A, i, j), i, j)
@inline bclr_inner(A::BCLRMatrix, i::Int, j::Int) =
    _bclr_factor_view(A.inner, _bclr_rank(A, i, j), i, j)
@inline bclr_outer_order(A::BCLRMatrix) = A.outer.order
@inline bclr_inner_order(A::BCLRMatrix) = A.inner.order

@inline function _bclr_offsets(order::TileOrderStyle, qm::Int, qn::Int,
                                rows::Int, rank_at)
    offsets = Vector{Int}(undef, qm * qn + 1)
    offsets[1] = 1
    @inbounds for slot in 1:(qm * qn)
        i, j = inverse_tile_index(order, qm, qn, slot)
        offsets[slot + 1] = offsets[slot] + rows * rank_at(i, j)
    end
    return offsets
end

function _bclr_rank_vector(order::TileOrderStyle, ranks_in::AbstractMatrix,
                           ::Type{RankT}) where {RankT<:Integer}
    qm, qn = size(ranks_in)
    ranks_out = Vector{RankT}(undef, qm * qn)
    @inbounds for j in 1:qn, i in 1:qm
        ranks_out[tile_linear_index(order, qm, qn, i, j)] = RankT(ranks_in[i, j])
    end
    return ranks_out
end

"""
    BCLRMatrix(backend, T, m, n, tile_size, ranks;
               outer_order=TileRowMajor, inner_order=TileColMajor,
               rank_type=Int32)

Allocate an exact-rank BCLR matrix. Initial BCLR support intentionally accepts
only a full regular tile grid; boundary tiles are a later extension.
"""
function BCLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int,
                    tile_size::NTuple{2,Int}, ranks_in::AbstractMatrix{<:Integer};
                    outer_order=TileRowMajor, inner_order=TileColMajor,
                    rank_type::Type{<:Integer}=Int32) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive"))
    m % bm == 0 && n % bn == 0 ||
        throw(ArgumentError("BCLRMatrix currently requires a full regular tile grid"))
    qm, qn = div(m, bm), div(n, bn)
    size(ranks_in) == (qm, qn) ||
        throw(DimensionMismatch("ranks must be a $qm × $qn matrix"))
    all(>=(0), ranks_in) || throw(ArgumentError("BCLR ranks must be nonnegative"))
    all(r -> r <= min(bm, bn), ranks_in) ||
        throw(ArgumentError("BCLR ranks must not exceed min(tile_size)"))

    outer_style = _order_instance(outer_order)
    inner_style = _order_instance(inner_order)
    rank_style = outer_style
    rank_at = (i, j) -> Int(ranks_in[i, j])
    uoffsets = _bclr_offsets(outer_style, qm, qn, bm, rank_at)
    voffsets = _bclr_offsets(inner_style, qm, qn, bn, rank_at)
    udata = zeros(backend, T, uoffsets[end] - 1)
    vdata = zeros(backend, T, voffsets[end] - 1)
    outer = BCLRPackedFactors(udata, uoffsets, outer_style, bm, qm, qn)
    inner = BCLRPackedFactors(vdata, voffsets, inner_style, bn, qm, qn)
    rankvec = _bclr_rank_vector(rank_style, ranks_in, rank_type)
    resid = Base.zeros(Float64, qm * qn)
    return BCLRMatrix{typeof(backend),T,typeof(udata),rank_type,typeof(outer_style),
                      typeof(inner_style),typeof(rank_style)}(
        backend, rank_style, m, n, tile_size, (0, 0), outer, inner, rankvec, resid,
        isempty(rankvec) ? 0 : maximum(rankvec))
end

function BCLRMatrix(backend::Backend, ::Type{T}, m::Int, n::Int, b::Int,
                    ranks_in::AbstractMatrix{<:Integer}; kwargs...) where {T}
    return BCLRMatrix(backend, T, m, n, (b, b), ranks_in; kwargs...)
end

"""Pack a padded, full-grid `TLRMatrix` into exact-rank BCLR storage."""
function pack_bclr(A::TLRMatrix{<:Any,T}; outer_order=TileRowMajor,
                   inner_order=TileColMajor, rank_type::Type{<:Integer}=Int32) where {T}
    tail_tile_size(A) == (0, 0) ||
        throw(ArgumentError("pack_bclr currently requires a full regular tile grid"))
    qm, qn = regular_grid_size(A)
    rank_grid = Matrix{Int}(undef, qm, qn)
    @inbounds for j in 1:qn, i in 1:qm
        rank_grid[i, j] = Int(ranks(A)[_rank_index(A, i, j)])
    end
    B = BCLRMatrix(get_backend(A), T, A.m, A.n, nominal_tile_size(A), rank_grid;
                   outer_order, inner_order, rank_type)
    @inbounds for j in 1:qn, i in 1:qm
        r = rank_grid[i, j]
        r == 0 && continue
        U, V = get_factors(A, i, j)
        copyto!(bclr_outer(B, i, j), U)
        copyto!(bclr_inner(B, i, j), V)
    end
    B.resid .= residuals(A)
    return B
end

@inline function get_factors(A::BCLRMatrix, i::Int, j::Int)
    return bclr_outer(A, i, j), bclr_inner(A, i, j)
end

@inline lowrank_regions(::BCLRMatrix) = (_INTERIOR,)
@inline outer_factors(A::BCLRMatrix, ::InteriorRegion) = A.outer
@inline inner_factors(A::BCLRMatrix, ::InteriorRegion) = A.inner
@inline region_tile_count(A::BCLRMatrix, ::InteriorRegion) = length(A.ranks)
@inline region_tile_coords(A::BCLRMatrix, ::InteriorRegion, k::Int) =
    inverse_tile_index(A.order, regular_grid_size(A)..., k)
