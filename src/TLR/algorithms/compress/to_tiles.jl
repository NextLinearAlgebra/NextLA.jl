"""
    materialize_tiles_generic_kernel!

Kernel that copies a dense matrix into padded tile storage, optionally splitting
diagonal and off-diagonal tiles into separate batches.
"""
@kernel function materialize_tiles_generic_kernel!(
    offdiag_or_all::AbstractArray{T,3},
    diag::Union{Nothing,AbstractArray{T,3}},
    A::AbstractMatrix{T},
    layout::Layout,
    ::Val{ROWS_PER_WORKGROUP},
    nwork::Int,
    launch_workgroups::Int,
) where {T,Layout<:TileMap,ROWS_PER_WORKGROUP}
    thread_row = @index(Local, Linear)
    workgroup_id = @index(Group, Linear)

    for work in workgroup_id:launch_workgroups:nwork
        tile_col = ((work - 1) % layout.tile_n) + 1
        linear = ((work - 1) ÷ layout.tile_n) + 1

        tile_i, tile_j = inverse_tile_index(layout, linear)
        p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
        tile_m, tile_n = tile_sizes(layout, tile_i, tile_j)

        is_diag = (diag !== nothing) && (tile_i == tile_j)
        dst = is_diag ? diag : offdiag_or_all
        idx = is_diag ? tile_i : (diag === nothing ? linear : offdiag_batch_index(layout, tile_i, tile_j))

        for row_in_tile in thread_row:ROWS_PER_WORKGROUP:layout.tile_m
            value = zero(T)
            if row_in_tile <= tile_m && tile_col <= tile_n
                @inbounds value = A[p0 + row_in_tile - 1, q0 + tile_col - 1]
            end
            @inbounds dst[row_in_tile, tile_col, idx] = value
        end
    end
end

"""
    to_tiles!(storage, A; kwargs...)

Materialize dense matrix `A` into internal packed tile workspace `storage`.
"""
function to_tiles!(
    storage::PackedDenseTileStorage{T,Offdiag,Diag,<:TileMap},
    A::AbstractMatrix{T};
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
    backend=KernelAbstractions.get_backend(A),
) where {T,Offdiag<:AbstractArray{T,3},Diag<:AbstractArray{T,3}}
    nwork = storage.layout.tile_n * prod(size(storage.layout))
    launch_workgroups = min(NWORKGROUPS, max(1, nwork))

    kernel! = materialize_tiles_generic_kernel!(backend, ROWS_PER_WORKGROUP)
    kernel!(
        storage.offdiag, storage.diag, A, storage.layout, Val(ROWS_PER_WORKGROUP), nwork, launch_workgroups;
        ndrange=ROWS_PER_WORKGROUP * launch_workgroups,
    )
    return storage
end

"""
    to_tiles!(tiles, A, layout; kwargs...)

Materialize dense matrix `A` into a single dense tile batch following `layout`.
"""
function to_tiles!(
    tiles::AbstractArray{T,3},
    A::AbstractMatrix{T},
    layout::TileMap;
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
    backend=KernelAbstractions.get_backend(A),
) where {T}
    size(tiles) == (layout.tile_m, layout.tile_n, prod(size(layout))) ||
        throw(DimensionMismatch("tile batch size mismatch"))

    nwork = layout.tile_n * prod(size(layout))
    launch_workgroups = min(NWORKGROUPS, max(1, nwork))

    kernel! = materialize_tiles_generic_kernel!(backend, ROWS_PER_WORKGROUP)
    kernel!(
        tiles, nothing, A, layout, Val(ROWS_PER_WORKGROUP), nwork, launch_workgroups;
        ndrange=ROWS_PER_WORKGROUP * launch_workgroups,
    )
    return tiles
end
