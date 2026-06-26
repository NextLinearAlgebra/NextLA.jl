"""Return the nominal tile size of `A`."""
@inline blocksize(A::TLRMatrix) = blocksize(A.layout)
"""Return the maximum tile rank representable in `A`."""
@inline maxrank(A::TLRMatrix) = maxrank(A.storage)
"""Return whether diagonal tiles are compressed in `A`."""
@inline compress_diag(A::TLRMatrix) = compress_diag(A.storage)
"""Return the rank buffer owned by `A.storage`."""
@inline ranks(A::TLRMatrix) = ranks(A.storage)
"""Return the dense diagonal tile storage owned by `A.storage`."""
@inline dense_diag(A::TLRMatrix) = dense_diag(A.storage)
"""Return the left low-rank factors owned by `A.storage`."""
@inline left_factors(A::TLRMatrix) = left_factors(A.storage)
"""Return the right low-rank factors owned by `A.storage`."""
@inline right_factors(A::TLRMatrix) = right_factors(A.storage)
"""Return the number of low-rank tile slots owned by `A.storage`."""
@inline nstored_tiles(A::TLRMatrix) = stored_tile_count(A.storage, A.layout)

# ─── Storage-agnostic per-tile factor accessors ───────────────────────────────

"""
    tile_u(A_tlr, ob) → AbstractMatrix

Return the left factor for off-diagonal tile `ob`, trimmed to its effective
rank.  Works for both `UniformTileStorage` and `CompactTileStorage`.
"""
@inline tile_u(A::TLRMatrix, ob::Integer) = tile_u(A.storage, A.layout, ob)

"""
    tile_v(A_tlr, ob) → AbstractMatrix

Return the right factor for off-diagonal tile `ob`, trimmed to its effective
rank.  Works for both `UniformTileStorage` and `CompactTileStorage`.
"""
@inline tile_v(A::TLRMatrix, ob::Integer) = tile_v(A.storage, A.layout, ob)

# Padded storage: trim both the tile height and effective rank.
# The returned SubArray still inherits the parent slot stride, so boundary
# tiles in UniformTileStorage remain physically column-strided by the nominal
# block size rather than densely repacked to their logical height.
@inline function tile_u(s::UniformTileStorage, layout::TileMap, ob::Integer)
    lin = offdiag_linear_index(layout, Int(ob))
    tile_i, tile_j = inverse_tile_index(layout, lin)
    tile_m, _ = tile_sizes(layout, tile_i, tile_j)
    view(s.U, 1:tile_m, 1:Int(s.ranks[ob]), ob)
end
@inline function tile_v(s::UniformTileStorage, layout::TileMap, ob::Integer)
    lin = offdiag_linear_index(layout, Int(ob))
    tile_i, tile_j = inverse_tile_index(layout, lin)
    _, tile_n = tile_sizes(layout, tile_i, tile_j)
    view(s.V, 1:tile_n, 1:Int(s.ranks[ob]), ob)
end
