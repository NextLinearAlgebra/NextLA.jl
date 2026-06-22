"""
    allocate_packed_dense_tiles(backend, T, layout)

Allocate the internal packed dense-tile workspace used by dense compression.
"""
@inline function allocate_packed_dense_tiles(
    backend::Backend,
    ::Type{T},
    layout::TileMap,
) where {T}
    offdiag = allocate(backend, T, layout.tile_m, layout.tile_n, noffdiag_tiles(layout))
    diag = allocate(backend, T, layout.tile_m, layout.tile_n, ndiag_tiles(layout))
    return PackedDenseTileStorage(offdiag, diag, layout)
end
