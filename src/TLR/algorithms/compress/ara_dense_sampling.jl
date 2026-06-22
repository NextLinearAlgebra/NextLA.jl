"""Dense-input sampling state backed by packed dense tiles."""
struct DenseSamplingState{A<:AbstractArray}
    tiles::A
end

function _sample_range!(
    Y_current::AbstractArray{T,3},
    source::DenseSamplingState,
    ws::ARAWorkspace,
    active_idx,
    n_active::Int,
    sample_size::Int,
    dense::Bool,
    backend,
) where {T}
    Omega_active = @view ws.Omega[:, 1:sample_size, 1:n_active]
    Random.randn!(Omega_active)

    _ara_bgemm!(
        'N', 'N', one(T), backend,
        source.tiles, size(source.tiles, 2), ws.M_ptrs, ws.Mcompact, active_idx,
        Omega_active, ws.Omega_ptrs,
        zero(T),
        Y_current, ws.Y_ptrs,
        n_active, dense,
    )
    return Y_current
end

function _sample_corange!(
    V::AbstractArray{T,3},
    source::DenseSamplingState,
    U::AbstractArray{T,3},
    backend,
) where {T}
    transchar = T <: Real ? 'T' : 'C'
    gemm_batched!(transchar, 'N', one(T), source.tiles, U, zero(T), V)
    return V
end

"""
    ara_batched!(U, V, ranks, M, max_rank, block_size, eps)

Compress a dense packed tile batch `M` into low-rank factors `U` and `V`.
"""
function ara_batched!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    M::AbstractArray{T,3},
    max_rank::Int,
    block_size::Int,
    eps,
) where {T,RankT<:Integer}
    get_backend(U) == get_backend(V) == get_backend(M) ||
        throw(ArgumentError("U, V, and M must have the same backend"))

    source = DenseSamplingState(M)
    return _ara_batched_impl!(
        U,
        V,
        ranks,
        source,
        size(M, 1),
        size(M, 2),
        size(M, 3),
        max_rank,
        block_size,
        eps,
        nothing;
        backend=get_backend(M),
    )
end
