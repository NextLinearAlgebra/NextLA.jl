"""Operator-input sampling state and its reusable work buffers."""
struct OperatorSamplingState{Op,L,XT,YT}
    op::Op
    layout::L
    compress_diag::Bool
    Xwork::XT
    Ywork::YT
end

"""Allocate the reusable work buffers needed by operator-based ARA sampling."""
function OperatorSamplingState(
    backend,
    ::Type{T},
    op,
    layout::TileMap,
    compress_diag::Bool,
    max_rank::Int,
) where {T}
    xwork = allocate(backend, T, layout.n, max_rank)
    ywork = allocate(backend, T, layout.m, max_rank)
    return OperatorSamplingState(op, layout, compress_diag, xwork, ywork)
end

function _sample_range!(
    Y_current::AbstractArray{T,3},
    source::OperatorSamplingState,
    ws::ARAWorkspace,
    active_idx,
    n_active::Int,
    sample_size::Int,
    dense::Bool,
    backend,
) where {T}
    Omega_active = @view ws.Omega[:, 1:sample_size, 1:n_active]
    Random.randn!(Omega_active)

    for p in 1:n_active
        batch = Int(@inbounds active_idx[p])
        linear = _full_tile_linear_index(source.layout, source.compress_diag, batch)
        tile_i, tile_j = inverse_tile_index(source.layout, linear)
        p0, q0 = tile_origin_coords(source.layout, tile_i, tile_j)
        tile_m, tile_n = tile_sizes(source.layout, tile_i, tile_j)

        fill!(source.Xwork, zero(T))
        @views source.Xwork[q0:(q0 + tile_n - 1), 1:sample_size] .= Omega_active[1:tile_n, 1:sample_size, p]
        mul!(source.Ywork, source.op, source.Xwork)
        fill!(@view(Y_current[:, :, p]), zero(T))
        @views Y_current[1:tile_m, 1:sample_size, p] .= source.Ywork[p0:(p0 + tile_m - 1), 1:sample_size]
    end

    return Y_current
end

function _sample_corange!(
    V::AbstractArray{T,3},
    source::OperatorSamplingState,
    U::AbstractArray{T,3},
    backend,
) where {T}
    for batch in 1:size(V, 3)
        linear = _full_tile_linear_index(source.layout, source.compress_diag, batch)
        tile_i, tile_j = inverse_tile_index(source.layout, linear)
        p0, q0 = tile_origin_coords(source.layout, tile_i, tile_j)
        tile_m, tile_n = tile_sizes(source.layout, tile_i, tile_j)
        rank = size(U, 2)

        fill!(source.Ywork, zero(T))
        @views source.Ywork[p0:(p0 + tile_m - 1), 1:rank] .= U[1:tile_m, 1:rank, batch]
        mul!(source.Xwork, adjoint(source.op), source.Ywork)
        fill!(@view(V[:, :, batch]), zero(T))
        @views V[1:tile_n, 1:rank, batch] .= source.Xwork[q0:(q0 + tile_n - 1), 1:rank]
    end

    return V
end

"""
    ara_batched_operator!(U, V, ranks, op, layout, max_rank, block_size, eps; kwargs...)

Compress a matrix-free operator into per-tile low-rank factors over `layout`.
"""
function ara_batched_operator!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    op,
    layout::TileMap,
    max_rank::Int,
    block_size::Int,
    eps;
    compress_diag::Bool=false,
    backend=get_backend(U),
) where {T,RankT<:Integer}
    source = OperatorSamplingState(backend, T, op, layout, compress_diag, max_rank)
    batch_size = size(U, 3)
    return _ara_batched_impl!(
        U,
        V,
        ranks,
        source,
        size(U, 1),
        size(V, 1),
        batch_size,
        max_rank,
        block_size,
        eps,
        nothing;
        backend,
    )
end
