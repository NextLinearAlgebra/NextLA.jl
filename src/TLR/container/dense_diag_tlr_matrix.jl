const _TILE_INT = UInt8(1)      # interior:        bm × bn
const _TILE_RIGHT = UInt8(2)    # right boundary:  bm × tail_n
const _TILE_BOTTOM = UInt8(3)   # bottom boundary: tail_m × bn

"""
    TLRDenseDiagMatrix{BackendT, T, Arr3T, RankT, OrderT}

Dense-diagonal TLR matrix. Diagonal tiles are dense; off-diagonal tiles are split
into regular interior, right boundary, and bottom boundary low-rank factors.

Storage is category-contiguous:

| field      | shape                         | category |
|------------|-------------------------------|----------|
| `int_U`    | `[bm,      maxrank, n_int]`   | regular non-diagonal tiles |
| `int_V`    | `[bn,      maxrank, n_int]`   | regular non-diagonal tiles |
| `right_U`  | `[bm,      maxrank, q_m]`     | right boundary |
| `right_V`  | `[tail_n,  maxrank, q_m]`     | right boundary |
| `bottom_U` | `[tail_m,  maxrank, q_n]`     | bottom boundary |
| `bottom_V` | `[bn,      maxrank, q_n]`     | bottom boundary |
| `D`        | `[bm, bn, n_full_diag]`       | full diagonal tiles |
| `D_corner` | `[tail_m, tail_n, 1 or 0]`    | final diagonal tail tile |

Ranks and residuals use the same category order: interior, right boundary,
bottom boundary. No dense-diagonal geometry table is cached; tile `(i, j)` maps
directly to a category and local slot.
"""
struct TLRDenseDiagMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},RankT<:Integer,OrderT<:TileOrderStyle} <: AbstractTLRMatrix{BackendT,T,OrderT}
    backend::BackendT
    order::OrderT
    m::Int
    n::Int
    nominal_tile_size::NTuple{2,Int} # (bm, bn)
    tail_tile_size::NTuple{2,Int}    # (tail_m, tail_n), 0 when no tail

    # Off-diagonal low-rank factors (one 3-D array per panel category):
    int_U::Arr3T        # [bm,      maxrank, n_int]
    int_V::Arr3T        # [bn,      maxrank, n_int]
    right_U::Arr3T      # [bm,      maxrank, n_right]  — zero depth when n%bn==0
    right_V::Arr3T      # [tail_n,  maxrank, n_right]
    bottom_U::Arr3T     # [tail_m,  maxrank, n_bottom] — zero depth when m%bm==0
    bottom_V::Arr3T     # [bn,      maxrank, n_bottom]

    D::Arr3T        # [bm, bn, n_full_diag] dense full diagonal tiles
    D_corner::Arr3T # [tm, tn, 1 or 0]     corner diagonal tile, if needed

    # Per-tile diagnostics — always CPU, contents written by compress!
    ranks::Vector{RankT}
    resid::Vector{Float64}  # estimated Frobenius error per off-diagonal tile
    maxrank::Int

end

@inline ndiag_tiles(A::TLRDenseDiagMatrix) = min(tilegrid_size(A)...)
@inline noffdiag_tiles(A::TLRDenseDiagMatrix) = prod(tilegrid_size(A)) - ndiag_tiles(A)
@inline dense_diag(A::TLRDenseDiagMatrix) = A.D
@inline dense_diag_corner(A::TLRDenseDiagMatrix) = A.D_corner
@inline _nfull_diag_tiles(A::TLRDenseDiagMatrix) = size(A.D, 3)

"""
    _diag_tile_view(A, tile_k)

Return the dense diagonal tile view for diagonal tile `tile_k`. Full-size
diagonal tiles live in `A.D`; a smaller final diagonal tile lives in
`A.D_corner`.
"""
@inline function _diag_tile_view(A::TLRDenseDiagMatrix, tile_k::Int)
    1 <= tile_k <= ndiag_tiles(A) || throw(BoundsError(1:ndiag_tiles(A), tile_k))
    if tile_k <= _nfull_diag_tiles(A)
        return view(A.D, :, :, tile_k)
    end
    size(A.D_corner, 3) != 0 || throw(BoundsError(1:_nfull_diag_tiles(A), tile_k))
    return view(A.D_corner, :, :, 1)
end

"""
    get_factors(A::TLRDenseDiagMatrix, i, j) -> (U, V)

Return the low-rank factors for off-diagonal tile `(i, j)`, trimmed to the
tile's effective rank. Diagonal tiles are stored densely, so `i == j` throws.
The returned views alias the underlying storage.
"""
@inline function get_factors(A::TLRDenseDiagMatrix, i::Int, j::Int)
    i == j && throw(ArgumentError("tile ($i, $j) is diagonal and stored densely"))
    cat, k = _offdiag_category_slot(A, i, j)
    r = Int(A.ranks[_rank_index(A, cat, k)])
    if cat == _TILE_INT
        return view(A.int_U, :, 1:r, k), view(A.int_V, :, 1:r, k)
    elseif cat == _TILE_RIGHT
        return view(A.right_U, :, 1:r, k), view(A.right_V, :, 1:r, k)
    else
        return view(A.bottom_U, :, 1:r, k), view(A.bottom_V, :, 1:r, k)
    end
end

"""
    _offdiag_category_slot(A, i, j) -> (category, local_slot)

Map global tile coordinates to dense-diagonal storage. Right boundary slots are
indexed by row `i`, bottom boundary slots by column `j`, and regular interior
slots use the requested tile order on the full-size interior grid.
"""
@inline function _offdiag_category_slot(A::TLRDenseDiagMatrix, i::Int, j::Int)
    mt, nt = tilegrid_size(A)
    checkbounds_tile(mt, nt, i, j)
    i == j && throw(ArgumentError("tile ($i, $j) is diagonal and stored densely"))
    if tail_tile_size(A, 2) != 0 && j == nt
        return _TILE_RIGHT, i
    elseif tail_tile_size(A, 1) != 0 && i == mt
        return _TILE_BOTTOM, j
    else
        q_m = fld(A.m, nominal_tile_size(A, 1))
        q_n = fld(A.n, nominal_tile_size(A, 2))
        return _TILE_INT, _offdiag_index(A.order, q_m, q_n, i, j)
    end
end

"""
    _rank_index(A, i, j) -> Int

Index into `A.ranks` / `A.resid` for off-diagonal tile `(i, j)`. The rank vector
is laid out as `interior; right; bottom`.
"""
@inline _rank_index(A::TLRDenseDiagMatrix, i::Int, j::Int) =
    _rank_index(A, _offdiag_category_slot(A, i, j)...)

@inline function _rank_index(A::TLRDenseDiagMatrix, cat::UInt8, k::Int)
    if cat == _TILE_INT
        return k
    elseif cat == _TILE_RIGHT
        return size(A.int_U, 3) + k
    else
        return size(A.int_U, 3) + size(A.right_U, 3) + k
    end
end

"""
    _category_coords(A, category, k) -> (tile_i, tile_j)

Inverse of the dense-diagonal category mapping for category-local slot `k`.
"""
@inline function _category_coords(A::TLRDenseDiagMatrix, cat::UInt8, k::Int)
    if cat == _TILE_INT
        q_m = fld(A.m, nominal_tile_size(A, 1))
        q_n = fld(A.n, nominal_tile_size(A, 2))
        return _offdiag_coords(A.order, q_m, q_n, k)
    elseif cat == _TILE_RIGHT
        return k, tilegrid_size(A)[2]
    else
        return tilegrid_size(A)[1], k
    end
end

"""
    _offdiag_index(order, mt, nt, tile_i, tile_j) -> Int

Return the position of off-diagonal tile `(tile_i, tile_j)` in the off-diagonal
enumeration of an explicit `mt×nt` tile grid, following `order`.
"""
@inline function _offdiag_index(order, mt::Integer, nt::Integer, tile_i::Int, tile_j::Int)
    mt_int = Int(mt)
    nt_int = Int(nt)
    checkbounds_tile(mt_int, nt_int, tile_i, tile_j)
    tile_i == tile_j && throw(ArgumentError("_offdiag_index is undefined for diagonal tiles"))
    linear = tile_linear_index(order, mt_int, nt_int, tile_i, tile_j)
    ndiag = min(mt_int, nt_int)
    diag_prefix = if _order_instance(order) isa TileColMajor
        min(ndiag, (linear + mt_int) ÷ (mt_int + 1))
    else
        min(ndiag, (linear + nt_int) ÷ (nt_int + 1))
    end
    return linear - diag_prefix
end

"""
    _offdiag_coords(order, mt, nt, ob) -> (tile_i, tile_j)

Inverse of `_offdiag_index` on an explicit `mt×nt` tile grid.
"""
@inline function _offdiag_coords(order, mt::Integer, nt::Integer, ob::Int)
    mt_int = Int(mt)
    nt_int = Int(nt)
    noff = mt_int * nt_int - min(mt_int, nt_int)
    1 <= ob <= noff || throw(BoundsError(1:noff, ob))
    return _offdiag_coords(_order_instance(order), mt_int, nt_int, ob)
end

@inline function _offdiag_coords(::TileColMajor, mt::Int, nt::Int, ob::Int)
    ndiag = min(mt, nt)
    first_cols = ndiag * (mt - 1)
    if mt > 1 && ob <= first_cols
        j0, pos0 = divrem(ob - 1, mt - 1)
        j, pos = j0 + 1, pos0 + 1
        i = pos < j ? pos : pos + 1
        return i, j
    else
        j0, i0 = divrem(ob - first_cols - 1, mt)
        i, j = i0 + 1, ndiag + j0 + 1
        return i, j
    end
end

@inline function _offdiag_coords(::TileRowMajor, mt::Int, nt::Int, ob::Int)
    ndiag = min(mt, nt)
    first_rows = ndiag * (nt - 1)
    if nt > 1 && ob <= first_rows
        i0, pos0 = divrem(ob - 1, nt - 1)
        i, pos = i0 + 1, pos0 + 1
        j = pos < i ? pos : pos + 1
        return i, j
    else
        i0, j0 = divrem(ob - first_rows - 1, nt)
        i, j = ndiag + i0 + 1, j0 + 1
        return i, j
    end
end

"""
    TLRDenseDiagMatrix(backend, T, m, n, tile_size, maxrank; rank_type=Int32, tile_order=TileColMajor)

Allocate an empty dense-diagonal TLR container for an `m×n` matrix with nominal
tile size `tile_size == (bm, bn)` and maximum per-tile rank `maxrank`.
"""
function TLRDenseDiagMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, tile_size::NTuple{2,Int}, maxrank::Int;
    rank_type::Type{<:Integer}=Int32,
    tile_order=TileColMajor,
) where {T}
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 && maxrank >= 0 ||
        throw(ArgumentError("m, n, and tile dimensions must be positive; maxrank must be non-negative"))

    order = _order_instance(tile_order)
    mt, nt = cld(m, bm), cld(n, bn)

    tail_m = m % bm
    tail_n = n % bn
    tail_size = (tail_m, tail_n)

    q_m, q_n = fld(m, bm), fld(n, bn)
    n_int = q_m * q_n - min(q_m, q_n)
    n_right = tail_n == 0 ? 0 : q_m
    n_bottom = tail_m == 0 ? 0 : q_n
    n_diag = min(mt, nt)

    # Use max(tail,1) so the leading dimension is never zero when depth is 0.
    tm_s = max(tail_m, 1)
    tn_s = max(tail_n, 1)

    int_U = zeros(backend, T, bm, maxrank, n_int)
    int_V = zeros(backend, T, bn, maxrank, n_int)
    right_U = zeros(backend, T, bm, maxrank, n_right)
    right_V = zeros(backend, T, tn_s, maxrank, n_right)
    bottom_U = zeros(backend, T, tm_s, maxrank, n_bottom)
    bottom_V = zeros(backend, T, bn, maxrank, n_bottom)
    corner_tm = n_diag == mt ? _last_dim(m, bm) : bm
    corner_tn = n_diag == nt ? _last_dim(n, bn) : bn
    has_diag_corner = n_diag > 0 && (corner_tm != bm || corner_tn != bn)
    n_full_diag = n_diag - Int(has_diag_corner)

    D = zeros(backend, T, bm, bn, n_full_diag)
    D_corner = zeros(backend, T, max(corner_tm, 1), max(corner_tn, 1), has_diag_corner ? 1 : 0)

    ranks = Base.zeros(rank_type, n_int + n_right + n_bottom)
    resid = Base.zeros(Float64, n_int + n_right + n_bottom)

    return TLRDenseDiagMatrix{typeof(backend),T,typeof(int_U),rank_type,typeof(order)}(
        backend, order, m, n, tile_size, tail_size,
        int_U, int_V, right_U, right_V, bottom_U, bottom_V,
        D, D_corner, ranks, resid, maxrank,
    )
end

function TLRDenseDiagMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, b::Int, maxrank::Int;
    kwargs...,
) where {T}
    return TLRDenseDiagMatrix(backend, T, m, n, (b, b), maxrank; kwargs...)
end

"""
    TLRDenseDiagMatrix(A, b, maxrank; kwargs...)

Allocate a dense-diagonal TLR container on the same backend as dense matrix `A`.
"""
function TLRDenseDiagMatrix(A::AbstractMatrix{T}, b::Int, maxrank::Int; kwargs...) where {T}
    return TLRDenseDiagMatrix(get_backend(A), T, size(A, 1), size(A, 2), b, maxrank; kwargs...)
end

function TLRDenseDiagMatrix(A::AbstractMatrix{T}, tile_size::NTuple{2,Int}, maxrank::Int; kwargs...) where {T}
    return TLRDenseDiagMatrix(get_backend(A), T, size(A, 1), size(A, 2), tile_size, maxrank; kwargs...)
end
