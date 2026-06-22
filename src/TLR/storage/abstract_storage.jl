"""
    AbstractTLRStorage{T}

Abstract storage backend for a TLR matrix representation of element type `T`.
"""
abstract type AbstractTLRStorage{T} end

"""
    PackedDenseTileStorage(offdiag, diag, layout)

Internal workspace storage holding a dense matrix materialized into a packed
tile batch. This is used during compression; it is not a final TLR format.
"""
struct PackedDenseTileStorage{T,Offdiag<:AbstractArray{T,3},Diag<:AbstractArray{T,3},L<:TileMap}
    offdiag::Offdiag
    diag::Diag
    layout::L

    function PackedDenseTileStorage(
        offdiag::Offdiag,
        diag::Diag,
        layout::L,
    ) where {T,Offdiag<:AbstractArray{T,3},Diag<:AbstractArray{T,3},L<:TileMap}
        size(offdiag, 1) == layout.tile_m ||
            throw(DimensionMismatch("packed off-diagonal tiles must have first dimension $(layout.tile_m)"))
        size(offdiag, 2) == layout.tile_n ||
            throw(DimensionMismatch("packed off-diagonal tiles must have second dimension $(layout.tile_n)"))
        size(diag, 1) == layout.tile_m ||
            throw(DimensionMismatch("packed diagonal tiles must have first dimension $(layout.tile_m)"))
        size(diag, 2) == layout.tile_n ||
            throw(DimensionMismatch("packed diagonal tiles must have second dimension $(layout.tile_n)"))
        size(offdiag, 3) == noffdiag_tiles(layout) ||
            throw(DimensionMismatch("packed off-diagonal batch size must be $(noffdiag_tiles(layout))"))
        size(diag, 3) == ndiag_tiles(layout) ||
            throw(DimensionMismatch("packed diagonal batch size must be $(ndiag_tiles(layout))"))
        return new{T,Offdiag,Diag,L}(offdiag, diag, layout)
    end
end
