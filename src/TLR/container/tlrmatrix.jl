# Category tags for the three off-diagonal tile categories.
const _TILE_INT = UInt8(1)   # interior:        b × b
const _TILE_RIGHT = UInt8(2)   # right boundary:  b × tail_n
const _TILE_BOTTOM = UInt8(3)   # bottom boundary: tail_m × b

"""
    TLRMatrix{BackendT, T, Arr3T, RankT, OrderT}

Tile Low-Rank (TLR) matrix.

Off-diagonal tiles are partitioned into three geometry categories —
interior (b×b), right-boundary (b×tail_n), and bottom-boundary (tail_m×b) —
each stored as a separate contiguous 3-D factor array:

| field      | shape                       | category        |
|------------|-----------------------------|-----------------|
| `int_U/V`  | `[b,      maxrank, n_int]`  | interior        |
| `right_U`  | `[b,      maxrank, n_right]`| right boundary  |
| `right_V`  | `[tail_n, maxrank, n_right]`| right boundary  |
| `bottom_U` | `[tail_m, maxrank, n_bot]`  | bottom boundary |
| `bottom_V` | `[b,      maxrank, n_bot]`  | bottom boundary |
| `D`        | `[b, b,   n_full_diag]`     | full diagonal   |
| `D_corner` | `[tm, tn, 1 or 0]`          | corner diagonal |

Category membership (`category`) and the global→local index map
(`local_index`) are computed once at construction and cached as `const`
fields, so no geometry work is repeated during compression or GEMM.

`int_U/V`, `right_U/V`, `bottom_U/V` are **mutable** to support future
storage reorganisation (e.g. rank bucketing) without rebuilding the container.
"""
mutable struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},RankT<:Integer,OrderT<:TileOrderStyle}
    const backend::BackendT
    const order::OrderT
    const m::Int
    const n::Int
    const tile_m::Int
    const tile_n::Int

    # Off-diagonal low-rank factors (one 3-D array per panel category):
    int_U::Arr3T        # [b,       maxrank, n_int]
    int_V::Arr3T        # [b,       maxrank, n_int]
    right_U::Arr3T      # [b,       maxrank, n_right]  — zero depth when n%b==0
    right_V::Arr3T      # [tail_n,  maxrank, n_right]
    bottom_U::Arr3T     # [tail_m,  maxrank, n_bottom] — zero depth when m%b==0
    bottom_V::Arr3T     # [b,       maxrank, n_bottom]

    const D::Arr3T        # [b, b, n_full_diag]  dense full diagonal tiles
    const D_corner::Arr3T # [tm, tn, 1 or 0]     corner diagonal tile, if needed

    # Per-tile effective rank — always CPU, mutable after compress!
    ranks::Vector{RankT}
    const maxrank::Int

    # Cached geometry — computed once from the dense/tile dimensions:
    const obs_int::Vector{Int}     # global off-diagonal indices of interior tiles
    const obs_right::Vector{Int}   # global off-diagonal indices of right-boundary tiles
    const obs_bottom::Vector{Int}  # global off-diagonal indices of bottom-boundary tiles
    const local_index::Vector{Int} # global ob → category-local slot (1-based)
    const category::Vector{UInt8}  # global ob → _TILE_INT / _TILE_RIGHT / _TILE_BOTTOM
end

Base.eltype(::Type{<:TLRMatrix{<:Any,T}}) where {T} = T
Base.eltype(::TLRMatrix{<:Any,T}) where {T} = T
Base.size(A::TLRMatrix) = (A.m, A.n)
Base.size(A::TLRMatrix, d::Int) = size(A)[d]

"""
    ndiag_tiles(A) -> Int

Number of diagonal tiles in the tiled matrix (i.e. `min(mt, nt)`).
"""
@inline ndiag_tiles(A::TLRMatrix) = min(tilegrid_size(A)...)

"""
    noffdiag_tiles(A) -> Int

Total number of off-diagonal tiles in the matrix.
"""
@inline noffdiag_tiles(A::TLRMatrix) = prod(tilegrid_size(A)) - ndiag_tiles(A)

"""
    tilegrid_size(A::TLRMatrix) -> (mt, nt)

Return the number of tile rows (`mt`) and tile columns (`nt`)
in the matrix tiling.
"""
@inline tilegrid_size(A::TLRMatrix) = (cld(A.m, A.tile_m), cld(A.n, A.tile_n))

"""
    tile_size(A, tile_i, tile_j) -> (m_local, n_local)

Return the actual dimensions of tile `(tile_i, tile_j)`,
handling boundary tiles that may be smaller than the nominal tile size.
"""
@inline function tile_size(A::TLRMatrix, tile_i::Int, tile_j::Int)
    mt, nt = tilegrid_size(A)

    row_size = tile_i == mt ?
        A.m - (mt - 1) * A.tile_m :
        A.tile_m

    col_size = tile_j == nt ?
        A.n - (nt - 1) * A.tile_n :
        A.tile_n

    return row_size, col_size
end

@inline blocksize(A::TLRMatrix) = (A.tile_m, A.tile_n)
@inline maxrank(A::TLRMatrix) = A.maxrank
@inline ranks(A::TLRMatrix) = A.ranks

"""
    dense_diag(A)

Return the packed batch of full-size diagonal tiles. When the final diagonal
tile is smaller than `blocksize(A)`, it is stored separately in
[`dense_diag_corner`](@ref).
"""
@inline dense_diag(A::TLRMatrix) = A.D

"""
    dense_diag_corner(A)

Return the packed corner diagonal storage. When there is no smaller final
diagonal tile, this is a zero-depth array.
"""
@inline dense_diag_corner(A::TLRMatrix) = A.D_corner
@inline _nfull_diag_tiles(A::TLRMatrix) = size(A.D, 3)

@inline function _diag_tile_view(A::TLRMatrix, tile_k::Int)
    1 <= tile_k <= ndiag_tiles(A) || throw(BoundsError(1:ndiag_tiles(A), tile_k))
    if tile_k <= _nfull_diag_tiles(A)
        return view(A.D, :, :, tile_k)
    end
    size(A.D_corner, 3) != 0 || throw(BoundsError(1:_nfull_diag_tiles(A), tile_k))
    return view(A.D_corner, :, :, 1)
end

"""
Return a named tuple `(interior, right, bottom)` of the left low-rank factor
arrays for each tile category.
"""
@inline left_factors(A::TLRMatrix) =
    (interior=A.int_U, right=A.right_U, bottom=A.bottom_U)

"""
Return a named tuple `(interior, right, bottom)` of the right low-rank factor
arrays for each tile category.
"""
@inline right_factors(A::TLRMatrix) =
    (interior=A.int_V, right=A.right_V, bottom=A.bottom_V)

# ─── Per-tile factor accessors ────────────────────────────────────────────────

"""
    tile_u(A, ob) → AbstractMatrix

Left low-rank factor for off-diagonal tile `ob`, trimmed to its effective rank.
The returned view aliases the underlying storage — no allocation.
"""
@inline function tile_u(A::TLRMatrix, ob::Integer)
    k = A.local_index[ob]
    r = Int(A.ranks[ob])
    cat = A.category[ob]
    if cat == _TILE_INT
        return view(A.int_U, :, 1:r, k)
    elseif cat == _TILE_RIGHT
        return view(A.right_U, :, 1:r, k)
    else
        return view(A.bottom_U, :, 1:r, k)
    end
end

"""
    tile_v(A, ob) → AbstractMatrix

Right low-rank factor for off-diagonal tile `ob`, trimmed to its effective rank.
The returned view aliases the underlying storage — no allocation.
"""
@inline function tile_v(A::TLRMatrix, ob::Integer)
    k = A.local_index[ob]
    r = Int(A.ranks[ob])
    cat = A.category[ob]
    if cat == _TILE_INT
        return view(A.int_V, :, 1:r, k)
    elseif cat == _TILE_RIGHT
        return view(A.right_V, :, 1:r, k)
    else
        return view(A.bottom_V, :, 1:r, k)
    end
end

"""
    tile_order(A::TLRMatrix)

Return the tile traversal / storage order used by `A`
(e.g. `TileRowMajor` or `TileColMajor`).
"""
@inline tile_order(A::TLRMatrix) = A.order

@inline _tile_linear_index(A::TLRMatrix, i::Integer, j::Integer) =
    tile_linear_index(A.order, tilegrid_size(A)..., i, j)

@inline _inverse_tile_index(A::TLRMatrix, linear::Integer) =
    inverse_tile_index(A.order, tilegrid_size(A)..., linear)

"""
    _offdiag_index(A, i, j) -> Int

Return the position of off-diagonal tile `(i, j)` in the global off-diagonal
enumeration (1-based, following the matrix's tile traversal order).
This is the key used to index `A.ranks`, `A.local_index`, and `A.category`.
"""
@inline _offdiag_index(A::TLRMatrix, i::Integer, j::Integer) =
    _offdiag_index(A.order, tilegrid_size(A)..., Int(i), Int(j))

"""
    _linear_from_offdiag(A, ob) -> Int

Map a global off-diagonal index `ob` back to its linear position in the full
tile grid, accounting for the skipped diagonal tiles.
Inverse of `_offdiag_index(A, i, j)`.
"""
@inline function _linear_from_offdiag(A::TLRMatrix{<:Any,<:Any,<:Any,<:Any,TileColMajor}, ob::Int)
    mt, _ = tilegrid_size(A)
    return ob + min(ndiag_tiles(A), cld(ob, mt))
end

@inline function _linear_from_offdiag(A::TLRMatrix{<:Any,<:Any,<:Any,<:Any,TileRowMajor}, ob::Int)
    _, nt = tilegrid_size(A)
    return ob + min(ndiag_tiles(A), cld(ob, nt))
end

"""
    tile_origin_coords(A, tile_i, tile_j) -> (row0, col0)

Return the global 1-based coordinates of the top-left entry
of tile `(tile_i, tile_j)`.
"""
@inline tile_origin_coords(A::TLRMatrix, tile_i::Int, tile_j::Int) =
    ((tile_i - 1) * A.tile_m + 1, (tile_j - 1) * A.tile_n + 1)

"""
    _offdiag_index(order, mt, nt, tile_i, tile_j) -> Int

Return the position of off-diagonal tile `(tile_i, tile_j)` in the
off-diagonal enumeration (1-based, following `order`), skipping diagonal tiles.
"""
@inline function _offdiag_index(order, mt::Integer, nt::Integer, tile_i::Int, tile_j::Int)
    tile_i == tile_j && throw(ArgumentError("_offdiag_index is undefined for diagonal tiles"))
    mt_int = Int(mt)
    nt_int = Int(nt)
    linear = tile_linear_index(order, mt_int, nt_int, tile_i, tile_j)
    ndiag  = min(mt_int, nt_int)
    diag_prefix = if _order_instance(order) isa TileColMajor
        min(ndiag, (linear + mt_int) ÷ (mt_int + 1))
    else
        min(ndiag, (linear + nt_int) ÷ (nt_int + 1))
    end
    return linear - diag_prefix
end

# ─── Internal geometry helper ─────────────────────────────────────────────────

function _build_geometry(order, m::Int, n::Int, tile_m::Int, tile_n::Int)
    mt, nt = cld(m, tile_m), cld(n, tile_n)
    has_right = n % tile_n != 0
    has_bottom = m % tile_m != 0

    ndiag = min(mt, nt)
    noff = mt * nt - ndiag
    local_index = Vector{Int}(undef, noff)
    category = Vector{UInt8}(undef, noff)
    obs_int = Int[]
    obs_right = Int[]
    obs_bottom = Int[]

    for linear in 1:(mt * nt)
        i, j = inverse_tile_index(order, mt, nt, linear)
        i == j && continue
        ob = _offdiag_index(order, mt, nt, i, j)
        if has_right && j == nt
            push!(obs_right, ob)
            category[ob] = _TILE_RIGHT
            local_index[ob] = length(obs_right)
        elseif has_bottom && i == mt
            push!(obs_bottom, ob)
            category[ob] = _TILE_BOTTOM
            local_index[ob] = length(obs_bottom)
        else
            push!(obs_int, ob)
            category[ob] = _TILE_INT
            local_index[ob] = length(obs_int)
        end
    end
    return obs_int, obs_right, obs_bottom, local_index, category
end

@inline function _alloc_zeros(backend, ::Type{T}, dims...) where {T}
    a = KernelAbstractions.allocate(backend, T, dims...)
    fill!(a, zero(T))
    return a
end

"""
    TLRMatrix(backend, T, m, n, b, maxrank; rank_type=Int32, tile_order=TileColMajor)

Allocate an empty TLR container for an `m×n` matrix with tile size `b` and
maximum per-tile rank `maxrank`.
"""
function TLRMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, b::Int, maxrank::Int;
    rank_type::Type{<:Integer}=Int32,
    tile_order=TileColMajor,
) where {T}
    m > 0 && n > 0 && b > 0 && maxrank >= 0 ||
        throw(ArgumentError("m, n, b must be positive; maxrank must be non-negative"))

    order = _order_instance(tile_order)
    mt, nt = cld(m, b), cld(n, b)

    tail_m = m % b   # logical height of last tile row (== b when m%b==0)
    tail_n = n % b   # logical width  of last tile col (== b when n%b==0)

    obs_int, obs_right, obs_bottom, local_index, category = _build_geometry(order, m, n, b, b)

    n_int = length(obs_int)
    n_right = length(obs_right)
    n_bottom = length(obs_bottom)
    n_diag = min(mt, nt)

    # Use max(tail,1) so the leading dimension is never zero when depth is 0.
    tm_s = max(tail_m, 1)
    tn_s = max(tail_n, 1)

    int_U = _alloc_zeros(backend, T, b, maxrank, n_int)
    int_V = _alloc_zeros(backend, T, b, maxrank, n_int)
    right_U = _alloc_zeros(backend, T, b, maxrank, n_right)
    right_V = _alloc_zeros(backend, T, tn_s, maxrank, n_right)
    bottom_U = _alloc_zeros(backend, T, tm_s, maxrank, n_bottom)
    bottom_V = _alloc_zeros(backend, T, b, maxrank, n_bottom)
    corner_tm = n_diag == mt ? (m - (mt - 1) * b) : b
    corner_tn = n_diag == nt ? (n - (nt - 1) * b) : b
    has_diag_corner = n_diag > 0 && (corner_tm != b || corner_tn != b)
    n_full_diag = n_diag - Int(has_diag_corner)

    D = _alloc_zeros(backend, T, b, b, n_full_diag)
    D_corner = _alloc_zeros(backend, T, max(corner_tm, 1), max(corner_tn, 1), has_diag_corner ? 1 : 0)

    ranks = zeros(rank_type, mt * nt - n_diag)

    return TLRMatrix{typeof(backend),T,typeof(int_U),rank_type,typeof(order)}(
        backend, order, m, n, b, b,
        int_U, int_V, right_U, right_V, bottom_U, bottom_V,
        D, D_corner, ranks, maxrank,
        obs_int, obs_right, obs_bottom, local_index, category,
    )
end

"""
    TLRMatrix(A, b, maxrank; kwargs...)

Allocate a TLR container on the same backend as dense matrix `A`.
"""
function TLRMatrix(A::AbstractMatrix{T}, b::Int, maxrank::Int; kwargs...) where {T}
    return TLRMatrix(get_backend(A), T, size(A, 1), size(A, 2), b, maxrank; kwargs...)
end
