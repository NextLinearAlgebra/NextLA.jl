"""
    TileMap(order, tile_m, tile_n, m, n)

Logical description of how a dense `m × n` matrix is partitioned into uniform
tiles of nominal size `tile_m × tile_n`.

`TileMap` owns geometry and traversal metadata only. It does not own any factor
or tile payload storage.
"""
struct TileMap{O<:TileOrder}
    order::O
    tile_m::Int
    tile_n::Int
    m::Int
    n::Int

    function TileMap(
        order::O,
        tile_m::Integer,
        tile_n::Integer,
        m::Integer,
        n::Integer,
    ) where {O<:TileOrder}
        tile_m > 0 || throw(ArgumentError("tile_m must be positive"))
        tile_n > 0 || throw(ArgumentError("tile_n must be positive"))
        m > 0 || throw(ArgumentError("m must be positive"))
        n > 0 || throw(ArgumentError("n must be positive"))

        layout = new{O}(order, Int(tile_m), Int(tile_n), Int(m), Int(n))
        size(order) == expected_tile_grid(layout) ||
            throw(DimensionMismatch("tile_order size must match the dense matrix tile grid"))
        return layout
    end
end

Base.size(layout::TileMap) = size(layout.order)

"""Return the tile-grid dimensions implied by the dense matrix shape."""
@inline expected_tile_grid(layout::TileMap) = (cld(layout.m, layout.tile_m), cld(layout.n, layout.tile_n))
"""Return the number of diagonal tiles in `layout`."""
@inline ndiag_tiles(layout::TileMap) = min(size(layout)...)
"""Return the number of off-diagonal tiles in `layout`."""
@inline noffdiag_tiles(layout::TileMap) = prod(size(layout)) - ndiag_tiles(layout)

"""Return the logical tile coordinate corresponding to `linear` in `layout`."""
@inline inverse_tile_index(layout::TileMap, linear::Int) = inverse_tile_index(layout.order, linear)

"""
    tile_origin_coords(layout, tile_i, tile_j)

Return the dense starting coordinates `(p0, q0)` of logical tile `(tile_i, tile_j)`.
"""
@inline function tile_origin_coords(layout::TileMap, tile_i::Int, tile_j::Int)
    p0 = (tile_i - 1) * layout.tile_m + 1
    q0 = (tile_j - 1) * layout.tile_n + 1
    return p0, q0
end

"""
    tile_sizes(layout, tile_i, tile_j)

Return the valid dense extent `(tile_m, tile_n)` of logical tile `(tile_i, tile_j)`.
"""
@inline function tile_sizes(layout::TileMap, tile_i::Int, tile_j::Int)
    p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
    tile_m = min(layout.tile_m, layout.m - p0 + 1)
    tile_n = min(layout.tile_n, layout.n - q0 + 1)
    return tile_m, tile_n
end

"""
    offdiag_batch_index(layout, tile_i, tile_j)

Map an off-diagonal logical tile to its compact off-diagonal batch index.
"""
@inline function offdiag_batch_index(layout::TileMap, tile_i::Int, tile_j::Int)
    tile_i == tile_j && throw(ArgumentError("off-diagonal index is undefined for diagonal tiles"))
    linear = tile_linear_index(layout.order, tile_i, tile_j)
    ndiag = ndiag_tiles(layout)
    diag_prefix = if layout.order isa TileOrder{ColMajor}
        min(ndiag, (linear + layout.order.mt) ÷ (layout.order.mt + 1))
    else
        min(ndiag, (linear + layout.order.nt) ÷ (layout.order.nt + 1))
    end
    return linear - diag_prefix
end

"""Inverse of [`offdiag_batch_index`](@ref) for column-major tile order."""
@inline function offdiag_linear_index(layout::TileMap{TileOrder{ColMajor}}, batch::Int)
    return batch + min(ndiag_tiles(layout), cld(batch, layout.order.mt))
end

"""Inverse of [`offdiag_batch_index`](@ref) for row-major tile order."""
@inline function offdiag_linear_index(layout::TileMap{TileOrder{RowMajor}}, batch::Int)
    return batch + min(ndiag_tiles(layout), cld(batch, layout.order.nt))
end
