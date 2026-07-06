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

Category membership (`category`), the global→local index map (`local_index`),
and each off-diagonal tile's grid coordinates (`coords`) are computed once at
construction and cached as `const` fields, so no geometry work is repeated
during compression or GEMM.

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

    # Per-tile diagnostics — always CPU, contents written by compress!
    ranks::Vector{RankT}
    const resid::Vector{Float64}  # estimated Frobenius error per off-diagonal tile
    const maxrank::Int

    # Cached geometry — computed once from the dense/tile dimensions:
    const obs_int::Vector{Int}     # global off-diagonal indices of interior tiles
    const obs_right::Vector{Int}   # global off-diagonal indices of right-boundary tiles
    const obs_bottom::Vector{Int}  # global off-diagonal indices of bottom-boundary tiles
    const local_index::Vector{Int} # global ob → category-local slot (1-based)
    const category::Vector{UInt8}  # global ob → _TILE_INT / _TILE_RIGHT / _TILE_BOTTOM
    const coords::Vector{Tuple{Int,Int}} # global ob → (tile_i, tile_j)
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
@inline KernelAbstractions.get_backend(A::TLRMatrix) = A.backend

"""
    residuals(A) -> Vector{Float64}

Estimated Frobenius-norm error of each off-diagonal tile's low-rank
approximation, indexed like [`ranks`](@ref) and written by [`compress!`](@ref).

The estimate includes the range-capture error of the randomized sketch
(the Yu–Gu–Li error indicator), so a tile whose spectrum does not fit within
`maxrank` reports its true residual instead of silently claiming convergence:
such a tile has `ranks(A)[ob] == maxrank(A)` and `residuals(A)[ob]` above the
requested tolerance, and should be treated as a candidate for dense storage
or a higher-rank second pass.
"""
@inline residuals(A::TLRMatrix) = A.resid

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

"""
    _offdiag_index(A, i, j) -> Int

Return the position of off-diagonal tile `(i, j)` in the global off-diagonal
enumeration (1-based, following the matrix's tile traversal order).
This is the key used to index `A.ranks`, `A.local_index`, and `A.category`.
"""
@inline _offdiag_index(A::TLRMatrix, i::Integer, j::Integer) =
    _offdiag_index(A.order, tilegrid_size(A)..., Int(i), Int(j))

"""
    tile_origin_coords(A, tile_i, tile_j) -> (row0, col0)

Return the global 1-based coordinates of the top-left entry
of tile `(tile_i, tile_j)`.
"""
@inline tile_origin_coords(A::TLRMatrix, tile_i::Int, tile_j::Int) =
    ((tile_i - 1) * A.tile_m + 1, (tile_j - 1) * A.tile_n + 1)

"""
    _offdiag_coords(A, ob) -> (tile_i, tile_j)

Logical tile coordinates of off-diagonal slot `ob`, for any tile order. Inverse
of [`_offdiag_index`](@ref), served from the precomputed `A.coords` table (built
by [`_build_geometry`](@ref)) — an O(1) lookup, no grid arithmetic per call.
"""
@inline _offdiag_coords(A::TLRMatrix, ob::Integer) = @inbounds A.coords[ob]

"""
    _dense_tile_view(dense, A, tile_i, tile_j) -> SubArray

Zero-copy view of the `(tile_i, tile_j)` block of dense matrix `dense`, sized to
the tile's actual (possibly boundary-truncated) extent. This is the single
primitive behind every "carve out the dense tile" site in compress, uncompress,
and gemm.
"""
@inline function _dense_tile_view(dense::AbstractMatrix, A::TLRMatrix, tile_i::Integer, tile_j::Integer)
    p0, q0 = tile_origin_coords(A, Int(tile_i), Int(tile_j))
    tm, tn = tile_size(A, Int(tile_i), Int(tile_j))
    return view(dense, p0:(p0 + tm - 1), q0:(q0 + tn - 1))
end

"""
    _offdiag_tile_view(dense, A, ob) -> SubArray

Dense view of the off-diagonal tile at slot `ob`. Convenience combiner:
`_dense_tile_view(dense, A, _offdiag_coords(A, ob)...)`.
"""
@inline _offdiag_tile_view(dense::AbstractMatrix, A::TLRMatrix, ob::Integer) =
    _dense_tile_view(dense, A, _offdiag_coords(A, ob)...)

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
    coords = Vector{Tuple{Int,Int}}(undef, noff)
    obs_int = Int[]
    obs_right = Int[]
    obs_bottom = Int[]

    for linear in 1:(mt * nt)
        i, j = inverse_tile_index(order, mt, nt, linear)
        i == j && continue
        ob = _offdiag_index(order, mt, nt, i, j)
        coords[ob] = (i, j)
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
    return obs_int, obs_right, obs_bottom, local_index, category, coords
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

    obs_int, obs_right, obs_bottom, local_index, category, coords = _build_geometry(order, m, n, b, b)

    n_int = length(obs_int)
    n_right = length(obs_right)
    n_bottom = length(obs_bottom)
    n_diag = min(mt, nt)

    # Use max(tail,1) so the leading dimension is never zero when depth is 0.
    tm_s = max(tail_m, 1)
    tn_s = max(tail_n, 1)

    int_U = zeros(backend, T, b, maxrank, n_int)
    int_V = zeros(backend, T, b, maxrank, n_int)
    right_U = zeros(backend, T, b, maxrank, n_right)
    right_V = zeros(backend, T, tn_s, maxrank, n_right)
    bottom_U = zeros(backend, T, tm_s, maxrank, n_bottom)
    bottom_V = zeros(backend, T, b, maxrank, n_bottom)
    corner_tm = n_diag == mt ? (m - (mt - 1) * b) : b
    corner_tn = n_diag == nt ? (n - (nt - 1) * b) : b
    has_diag_corner = n_diag > 0 && (corner_tm != b || corner_tn != b)
    n_full_diag = n_diag - Int(has_diag_corner)

    D = zeros(backend, T, b, b, n_full_diag)
    D_corner = zeros(backend, T, max(corner_tm, 1), max(corner_tn, 1), has_diag_corner ? 1 : 0)

    ranks = Base.zeros(rank_type, mt * nt - n_diag)
    resid = Base.zeros(Float64, mt * nt - n_diag)

    return TLRMatrix{typeof(backend),T,typeof(int_U),rank_type,typeof(order)}(
        backend, order, m, n, b, b,
        int_U, int_V, right_U, right_V, bottom_U, bottom_V,
        D, D_corner, ranks, resid, maxrank,
        obs_int, obs_right, obs_bottom, local_index, category, coords,
    )
end

"""
    TLRMatrix(A, b, maxrank; kwargs...)

Allocate a TLR container on the same backend as dense matrix `A`.
"""
function TLRMatrix(A::AbstractMatrix{T}, b::Int, maxrank::Int; kwargs...) where {T}
    return TLRMatrix(get_backend(A), T, size(A, 1), size(A, 2), b, maxrank; kwargs...)
end
