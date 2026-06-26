"""
    UniformTileStorage

Concrete TLR storage using per-tile low-rank factors `U` and `V`, plus dense
diagonal tiles `D` when `compress_diag == false`.
"""
struct UniformTileStorage{
    T,
    RankT<:Integer,
    UStore<:AbstractArray{T,3},
    VStore<:AbstractArray{T,3},
    DiagStore<:AbstractArray{T,3},
    RanksStore<:AbstractVector{RankT},
} <: AbstractTLRStorage{T}
    U::UStore
    V::VStore
    D::DiagStore
    ranks::RanksStore
    maxrank::Int
    compress_diag::Bool
end

Base.eltype(::Type{<:UniformTileStorage{T}}) where {T} = T
Base.eltype(storage::UniformTileStorage{T}) where {T} = T

"""Return the nominal tile size of `layout`."""
@inline blocksize(layout::TileMap) = layout.tile_m
"""Return the maximum rank storable in `storage`."""
@inline maxrank(storage::UniformTileStorage) = storage.maxrank
"""Return whether diagonal tiles are stored in compressed form."""
@inline compress_diag(storage::UniformTileStorage) = storage.compress_diag
"""Return the rank buffer for `storage`."""
@inline ranks(storage::UniformTileStorage) = storage.ranks
"""Return the dense diagonal tile storage."""
@inline dense_diag(storage::UniformTileStorage) = storage.D
"""Return the left low-rank factors."""
@inline left_factors(storage::UniformTileStorage) = storage.U
"""Return the right low-rank factors."""
@inline right_factors(storage::UniformTileStorage) = storage.V

"""
    tile_storage_index(storage, layout, i, j)

Return the storage slot used by logical tile `(i, j)` in `storage`.
"""
@inline function tile_storage_index(storage::UniformTileStorage, layout::TileMap, i::Integer, j::Integer)
    return storage.compress_diag ? tile_linear_index(layout.order, i, j) : offdiag_batch_index(layout, Int(i), Int(j))
end

"""Return the number of low-rank tile slots owned by `storage`."""
@inline stored_tile_count(storage::UniformTileStorage, layout::TileMap) =
    storage.compress_diag ? prod(size(layout)) : noffdiag_tiles(layout)

"""
    allocate_storage(backend, T, layout, max_rank; compress_diag=false, rank_type=Int32)

Allocate a `UniformTileStorage` matching `layout` on `backend`.
"""
function allocate_storage(
    backend::Backend,
    ::Type{T},
    layout::TileMap,
    max_rank::Int;
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
) where {T}
    ntiles = compress_diag ? prod(size(layout)) : noffdiag_tiles(layout)
    U = allocate(backend, T, layout.tile_m, max_rank, ntiles)
    V = allocate(backend, T, layout.tile_n, max_rank, ntiles)
    D = allocate(backend, T, layout.tile_m, layout.tile_n, compress_diag ? 0 : ndiag_tiles(layout))
    rs = allocate(backend, rank_type, ntiles)
    fill!(rs, zero(rank_type))
    return UniformTileStorage(U, V, D, rs, max_rank, compress_diag)
end
