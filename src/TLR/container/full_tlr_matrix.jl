"""
    TLRMatrix{BackendT, T, Arr3T, RankT, OrderT}

Fully tile-low-rank matrix. Every tile, including diagonal and boundary-corner
tiles, is stored through low-rank factors.

Storage is split by tile geometry:

| field      | shape                       | category |
|------------|-----------------------------|----------|
| `int_U`    | `[bm,     maxrank, q_m*q_n]`| full-size regular tiles |
| `int_V`    | `[bn,     maxrank, q_m*q_n]`| full-size regular tiles |
| `right_U`  | `[bm,     maxrank, q_m]`    | right boundary |
| `right_V`  | `[tail_n, maxrank, q_m]`    | right boundary |
| `bottom_U` | `[tail_m, maxrank, q_n]`    | bottom boundary |
| `bottom_V` | `[bn,     maxrank, q_n]`    | bottom boundary |
| `corner_U` | `[tail_m, maxrank, 1]`      | bottom-right corner |
| `corner_V` | `[tail_n, maxrank, 1]`      | bottom-right corner |

where `q_m = fld(m, bm)` and `q_n = fld(n, bn)`. Zero-depth arrays are used for
boundary categories that do not exist.
"""
struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},RankT<:Integer,OrderT<:TileOrderStyle} <: AbstractTLRMatrix{BackendT,T,OrderT}
    backend::BackendT
    order::OrderT
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int}
    tail_tile_size::NTuple{2,Int}

    int_U::Arr3T
    int_V::Arr3T
    right_U::Arr3T
    right_V::Arr3T
    bottom_U::Arr3T
    bottom_V::Arr3T
    corner_U::Arr3T
    corner_V::Arr3T

    ranks::Vector{RankT}
    resid::Vector{Float64}
    maxrank::Int
end

@inline function _tile_index(A::TLRMatrix, i::Integer, j::Integer)
    mt, nt = tilegrid_size(A)
    return tile_linear_index(A.order, mt, nt, Int(i), Int(j))
end

@inline function region_tile_coords(A::TLRMatrix, ::InteriorRegion, k::Int)
    q_m, q_n = regular_tilegrid_size(A)
    return inverse_tile_index(A.order, q_m, q_n, k)
end

@inline function region_slot(A::TLRMatrix, i::Int, j::Int)
    mt, nt = tilegrid_size(A)
    checkbounds_tile(mt, nt, i, j)
    q_m, q_n = regular_tilegrid_size(A)
    if i <= q_m && j <= q_n
        return _INTERIOR, tile_linear_index(A.order, q_m, q_n, i, j)
    elseif j == nt && tail_tile_size(A, 2) != 0 && i <= q_m
        return _RIGHT, i
    elseif i == mt && tail_tile_size(A, 1) != 0 && j <= q_n
        return _BOTTOM, j
    end
    return _CORNER, 1
end

"""
    get_factors(A::TLRMatrix, i, j) -> (U, V)

Return the low-rank factors for tile `(i, j)`, trimmed to the tile's effective
rank. Unlike dense-diagonal TLR, diagonal tiles are valid low-rank tiles.
"""
@inline function get_factors(A::TLRMatrix, i::Int, j::Int)
    region, k = region_slot(A, i, j)
    r = Int(A.ranks[_tile_index(A, i, j)])
    U = outer_factors(A, region)
    V = inner_factors(A, region)
    return view(U, :, 1:r, k), view(V, :, 1:r, k)
end

@inline outer_factors(A::TLRMatrix, ::InteriorRegion) = A.int_U
@inline inner_factors(A::TLRMatrix, ::InteriorRegion) = A.int_V
@inline outer_factors(A::TLRMatrix, ::RightRegion) = A.right_U
@inline inner_factors(A::TLRMatrix, ::RightRegion) = A.right_V
@inline outer_factors(A::TLRMatrix, ::BottomRegion) = A.bottom_U
@inline inner_factors(A::TLRMatrix, ::BottomRegion) = A.bottom_V
@inline outer_factors(A::TLRMatrix, ::CornerRegion) = A.corner_U
@inline inner_factors(A::TLRMatrix, ::CornerRegion) = A.corner_V
@inline lowrank_regions(::TLRMatrix) = (_INTERIOR, _RIGHT, _BOTTOM, _CORNER)

"""
    TLRMatrix(backend, T, m, n, tile_size, maxrank; rank_type=Int32, tile_order=TileRowMajor)

Allocate an empty fully low-rank TLR container for an `m×n` matrix with nominal
tile size `tile_size == (bm, bn)` and maximum per-tile rank `maxrank`.
"""
function TLRMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, tile_size::NTuple{2,Int}, maxrank::Int;
    rank_type::Type{<:Integer}=Int32,
    tile_order=TileRowMajor,
) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 && maxrank >= 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive; maxrank must be non-negative"))

    order = _order_instance(tile_order)
    mt, nt = cld(m, bm), cld(n, bn)
    q_m, q_n = fld(m, bm), fld(n, bn)

    tail_m = m % bm
    tail_n = n % bn
    tail_size = (tail_m, tail_n)
    has_right = tail_n != 0
    has_bottom = tail_m != 0
    has_corner = has_right && has_bottom

    tm_s = max(tail_m, 1)
    tn_s = max(tail_n, 1)

    n_int = q_m * q_n
    n_right = has_right ? q_m : 0
    n_bottom = has_bottom ? q_n : 0
    n_corner = has_corner ? 1 : 0

    int_U = zeros(backend, T, bm, maxrank, n_int)
    int_V = zeros(backend, T, bn, maxrank, n_int)
    right_U = zeros(backend, T, bm, maxrank, n_right)
    right_V = zeros(backend, T, tn_s, maxrank, n_right)
    bottom_U = zeros(backend, T, tm_s, maxrank, n_bottom)
    bottom_V = zeros(backend, T, bn, maxrank, n_bottom)
    corner_U = zeros(backend, T, tm_s, maxrank, n_corner)
    corner_V = zeros(backend, T, tn_s, maxrank, n_corner)

    ranks = Base.zeros(rank_type, mt * nt)
    resid = Base.zeros(Float64, mt * nt)

    return TLRMatrix{typeof(backend),T,typeof(int_U),rank_type,typeof(order)}(
        backend, order, m, n, tile_size, tail_size,
        int_U, int_V, right_U, right_V, bottom_U, bottom_V, corner_U, corner_V,
        ranks, resid, maxrank,
    )
end

function TLRMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, b::Int, maxrank::Int;
    kwargs...,
) where {T}
    return TLRMatrix(backend, T, m, n, (b, b), maxrank; kwargs...)
end

function TLRMatrix(A::AbstractMatrix{T}, b::Int, maxrank::Int; kwargs...) where {T}
    return TLRMatrix(get_backend(A), T, size(A, 1), size(A, 2), b, maxrank; kwargs...)
end

function TLRMatrix(A::AbstractMatrix{T}, tile_size::NTuple{2,Int}, maxrank::Int; kwargs...) where {T}
    return TLRMatrix(get_backend(A), T, size(A, 1), size(A, 2), tile_size, maxrank; kwargs...)
end
