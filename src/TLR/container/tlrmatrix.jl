# Category tags for the three off-diagonal tile categories.
const _TILE_INT = UInt8(1)   # interior:        b × b
const _TILE_RIGHT = UInt8(2)   # right boundary:  b × tail_n
const _TILE_BOTTOM = UInt8(3)   # bottom boundary: tail_m × b

"""
    TLRMatrix{BackendT, T, Arr3T, RankT}

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
| `D`        | `[b, b,   n_diag]`          | diagonal        |

Category membership (`category`) and the global→local index map
(`local_index`) are computed once at construction and cached as `const`
fields, so no geometry work is repeated during compression or GEMM.

`int_U/V`, `right_U/V`, `bottom_U/V` are **mutable** to support future
storage reorganisation (e.g. rank bucketing) without rebuilding the container.
"""
mutable struct TLRMatrix{BackendT<:Backend,T,Arr3T<:AbstractArray{T,3},RankT<:Integer}
    const backend::BackendT
    const layout::TileMap

    # Off-diagonal low-rank factors (one 3-D array per panel category):
    int_U::Arr3T        # [b,       maxrank, n_int]
    int_V::Arr3T        # [b,       maxrank, n_int]
    right_U::Arr3T      # [b,       maxrank, n_right]  — zero depth when n%b==0
    right_V::Arr3T      # [tail_n,  maxrank, n_right]
    bottom_U::Arr3T     # [tail_m,  maxrank, n_bottom] — zero depth when m%b==0
    bottom_V::Arr3T     # [b,       maxrank, n_bottom]

    const D::Arr3T      # [b, b, n_diag]  dense diagonal tiles

    # Per-tile effective rank — always CPU, mutable after compress!
    ranks::Vector{RankT}
    const maxrank::Int
    const compress_diag::Bool

    # Cached geometry — computed once from layout, never recomputed:
    const obs_int::Vector{Int}     # off-diagonal batch indices of interior tiles
    const obs_right::Vector{Int}   # off-diagonal batch indices of right-boundary tiles
    const obs_bottom::Vector{Int}  # off-diagonal batch indices of bottom-boundary tiles
    const local_index::Vector{Int} # global ob → category-local slot (1-based)
    const category::Vector{UInt8}  # global ob → _TILE_INT / _TILE_RIGHT / _TILE_BOTTOM
end

Base.eltype(::Type{<:TLRMatrix{<:Any,T}}) where {T} = T
Base.eltype(::TLRMatrix{<:Any,T}) where {T} = T
Base.size(A::TLRMatrix) = (A.layout.m, A.layout.n)
Base.size(A::TLRMatrix, d::Int) = size(A)[d]

@inline tile_linear_index(A::TLRMatrix, i::Integer, j::Integer) =
    tile_linear_index(A.layout.order, i, j)

@inline tile_storage_index(A::TLRMatrix, i::Integer, j::Integer) =
    offdiag_batch_index(A.layout, Int(i), Int(j))

# ─── Internal geometry helper ─────────────────────────────────────────────────

function _build_geometry(layout::TileMap)
    mt, nt = size(layout)
    has_right = layout.n % layout.tile_n != 0
    has_bottom = layout.m % layout.tile_m != 0

    noff = noffdiag_tiles(layout)
    local_index = Vector{Int}(undef, noff)
    category = Vector{UInt8}(undef, noff)
    obs_int = Int[]
    obs_right = Int[]
    obs_bottom = Int[]

    for linear in 1:prod(size(layout))
        i, j = inverse_tile_index(layout, linear)
        i == j && continue
        ob = offdiag_batch_index(layout, i, j)
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
    TLRMatrix(backend, T, m, n, b, maxrank; compress_diag=false, rank_type=Int32, tile_order=TileColMajor)

Allocate an empty TLR container for an `m×n` matrix with tile size `b` and
maximum per-tile rank `maxrank`.
"""
function TLRMatrix(
    backend::Backend, ::Type{T},
    m::Int, n::Int, b::Int, maxrank::Int;
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:TileOrder}=TileColMajor,
) where {T}
    m > 0 && n > 0 && b > 0 && maxrank >= 0 ||
        throw(ArgumentError("m, n, b must be positive; maxrank must be non-negative"))

    layout = TileMap(tile_order(cld(m, b), cld(n, b)), b, b, m, n)
    mt, nt = size(layout)

    tail_m = m - (mt - 1) * b   # logical height of last tile row (== b when m%b==0)
    tail_n = n - (nt - 1) * b   # logical width  of last tile col (== b when n%b==0)

    obs_int, obs_right, obs_bottom, local_index, category = _build_geometry(layout)

    n_int = length(obs_int)
    n_right = length(obs_right)
    n_bottom = length(obs_bottom)
    n_diag = ndiag_tiles(layout)

    # Use max(tail,1) so the leading dimension is never zero when depth is 0.
    tm_s = max(tail_m, 1)
    tn_s = max(tail_n, 1)

    int_U = _alloc_zeros(backend, T, b, maxrank, n_int)
    int_V = _alloc_zeros(backend, T, b, maxrank, n_int)
    right_U = _alloc_zeros(backend, T, b, maxrank, n_right)
    right_V = _alloc_zeros(backend, T, tn_s, maxrank, n_right)
    bottom_U = _alloc_zeros(backend, T, tm_s, maxrank, n_bottom)
    bottom_V = _alloc_zeros(backend, T, b, maxrank, n_bottom)
    D = _alloc_zeros(backend, T, b, b, n_diag)

    ranks = zeros(rank_type, noffdiag_tiles(layout))

    return TLRMatrix{typeof(backend),T,typeof(int_U),rank_type}(
        backend, layout,
        int_U, int_V, right_U, right_V, bottom_U, bottom_V,
        D, ranks, maxrank, compress_diag,
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
