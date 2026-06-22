"""Map an active batch index back to the full logical tile traversal index."""
@inline function _full_tile_linear_index(layout::TileMap, compress_diag::Bool, batch::Int)
    return compress_diag ? batch : offdiag_linear_index(layout, batch)
end

"""
    materialize_diag_tiles!(diag, op, layout; backend)

Materialize exact diagonal tiles of matrix-free operator `op` into dense tile
storage `diag`.
"""
function materialize_diag_tiles!(
    diag::AbstractArray{T,3},
    op,
    layout::TileMap;
    backend,
) where {T}
    size(diag, 1) == layout.tile_m || throw(DimensionMismatch("diag row size mismatch"))
    size(diag, 2) == layout.tile_n || throw(DimensionMismatch("diag column size mismatch"))
    size(diag, 3) == ndiag_tiles(layout) || throw(DimensionMismatch("diag batch size mismatch"))

    X = allocate(backend, T, layout.n, layout.tile_n)
    Y = allocate(backend, T, layout.m, layout.tile_n)

    for tile in 1:ndiag_tiles(layout)
        fill!(X, zero(T))
        tile_i, tile_j = inverse_tile_index(layout, tile_linear_index(layout.order, tile, tile))
        p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
        tile_m, tile_n = tile_sizes(layout, tile_i, tile_j)

        for local_col in 1:tile_n
            @inbounds X[q0 + local_col - 1, local_col] = one(T)
        end

        mul!(Y, op, X)
        fill!(@view(diag[:, :, tile]), zero(T))
        @views diag[1:tile_m, 1:tile_n, tile] .= Y[p0:(p0 + tile_m - 1), 1:tile_n]
    end

    return diag
end
