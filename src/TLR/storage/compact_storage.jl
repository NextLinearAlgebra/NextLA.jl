"""
    CompactTileStorage

Compact TLR storage where every off-diagonal tile is stored at its effective
rank. Both U and V factors are laid out as a **single flat buffer** in
column-major order; per-tile access uses 1-indexed offset arrays.

Layout of `U_data`:
  tile ob occupies `U_data[U_offsets[ob] : U_offsets[ob] + U_nrows[ob]*ranks[ob] - 1]`
  reshaped as `[U_nrows[ob], ranks[ob]]` in column-major order.

The flat buffers (`U_data`, `V_data`, `D`) live on the same device as the
source `TLRMatrix` was compressed on.  The offset / metadata arrays
(`U_offsets`, `V_offsets`, `U_nrows`, `V_nrows`, `ranks`) are always
**CPU arrays** for efficient host-side access.

Created by [`compact!`](@ref).
"""
struct CompactTileStorage{
    T,
    RankT     <: Integer,
    DataStore <: AbstractVector{T},   # flat buffer: CPU or GPU
    DiagStore <: AbstractArray{T,3},  # diagonal tiles: CPU or GPU
} <: AbstractTLRStorage{T}
    U_data::DataStore
    V_data::DataStore
    U_offsets::Vector{Int}   # CPU: U_offsets[ob] = 1-based start in U_data
    V_offsets::Vector{Int}   # CPU: V_offsets[ob] = 1-based start in V_data
    U_nrows::Vector{Int}     # CPU: actual row count per tile for U (tile_m)
    V_nrows::Vector{Int}     # CPU: actual row count per tile for V (tile_n)
    D::DiagStore
    ranks::Vector{RankT}     # CPU: effective rank per tile
    compress_diag::Bool
end

Base.eltype(::Type{<:CompactTileStorage{T}}) where {T} = T
Base.eltype(::CompactTileStorage{T}) where {T} = T

@inline maxrank(s::CompactTileStorage)         = isempty(s.ranks) ? 0 : Int(maximum(s.ranks))
@inline compress_diag(s::CompactTileStorage)   = s.compress_diag
@inline ranks(s::CompactTileStorage)           = s.ranks
@inline dense_diag(s::CompactTileStorage)      = s.D
@inline left_factors(s::CompactTileStorage)    = s.U_data    # flat buffer
@inline right_factors(s::CompactTileStorage)   = s.V_data    # flat buffer

@inline stored_tile_count(::CompactTileStorage, layout::TileMap) = noffdiag_tiles(layout)

@inline function tile_storage_index(::CompactTileStorage, layout::TileMap,
                                    i::Integer, j::Integer)
    offdiag_batch_index(layout, Int(i), Int(j))
end

# ─── Per-tile factor accessors ────────────────────────────────────────────────

@inline function tile_u(s::CompactTileStorage{T}, ::TileMap, ob::Integer) where {T}
    off   = s.U_offsets[ob]::Int
    nrows = s.U_nrows[ob]::Int
    r_ob  = Int(s.ranks[ob])
    len   = nrows * r_ob
    reshape(view(s.U_data, off : off + len - 1), nrows, r_ob)
end

@inline function tile_v(s::CompactTileStorage{T}, ::TileMap, ob::Integer) where {T}
    off   = s.V_offsets[ob]::Int
    nrows = s.V_nrows[ob]::Int
    r_ob  = Int(s.ranks[ob])
    len   = nrows * r_ob
    reshape(view(s.V_data, off : off + len - 1), nrows, r_ob)
end
