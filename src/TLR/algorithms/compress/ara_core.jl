using Random
using KernelAbstractions

"""
    ARAWorkspace

Internal workspace for the adaptive randomized algorithm. It owns the temporary
sample, orthogonalization, compaction, and pointer-batch buffers needed during
compression.
"""
struct ARAWorkspace{AT,ST,IT,FT,PT}
    Omega::AT
    Y::AT
    Yscratch::AT
    Z::AT
    G::AT
    Gscratch::AT
    Mcompact::AT
    Ucompact::AT
    potrf_status::FT
    active_idx::IT
    next_active_idx::IT
    survivor_pos::IT
    survivor_flags::FT
    small_sample_count::FT
    sample_scale::ST
    M_ptrs::PT
    U_ptrs::PT
    Omega_ptrs::PT
    Y_ptrs::PT
    Z_ptrs::PT
end

"""Allocate the internal workspace required by one batched ARA run."""
function _allocate_ara_workspace(
    backend,
    ptr_type::Type,
    ::Type{T},
    ::Type{RankT},
    m::Int,
    n::Int,
    batch_size::Int,
    max_rank::Int,
    block_size::Int,
) where {T,RankT<:Integer}
    RT = typeof(real(zero(T)))
    IdxT = Int
    max_block = min(block_size, max_rank)
    alloc(typ, dims...) = allocate(backend, typ, dims...)
    return ARAWorkspace(
        alloc(T, n, max_block, batch_size),
        alloc(T, m, max_block, batch_size),
        alloc(T, m, max_block, batch_size),
        alloc(T, max_rank, max_block, batch_size),
        alloc(T, max_block, max_block, batch_size),
        alloc(T, max_block, max_block, batch_size),
        alloc(T, m, n, batch_size),
        alloc(T, m, max_rank, batch_size),
        alloc(Int32, batch_size),
        alloc(IdxT, batch_size),
        alloc(IdxT, batch_size),
        alloc(IdxT, batch_size),
        alloc(Int32, batch_size),
        alloc(Int32, batch_size),
        alloc(RT, batch_size),
        alloc(ptr_type, batch_size),
        alloc(ptr_type, batch_size),
        alloc(ptr_type, batch_size),
        alloc(ptr_type, batch_size),
        alloc(ptr_type, batch_size),
    )
end

"""Gather the currently active batch items into a compact 3D buffer."""
@kernel function gather_active_blocks3d_kernel!(
    dst,
    src,
    active_idx,
    nrows::Int,
    ncols::Int,
    n_active::Int,
)
    row, col, p = @index(Global, NTuple)
    if p <= n_active && row <= nrows && col <= ncols
        tile = Int(@inbounds active_idx[p])
        @inbounds dst[row, col, p] = src[row, col, tile]
    end
end

"""Initialize the active-set index vector to `1:batch_size`."""
@kernel function initialize_active_idx_kernel!(active_idx, batch_size::Int)
    p = @index(Global)
    if p <= batch_size
        @inbounds active_idx[p] = p
    end
end

"""Initialize the per-tile convergence scale from the first sampled block."""
@kernel function initialize_ara_scale_active_kernel!(
    sample_scale,
    G_block,
    active_idx,
    sample_size::Int,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        tile = @inbounds active_idx[p]
        scale = zero(eltype(sample_scale))
        @inbounds for local_col in 1:sample_size
            scale = max(scale, abs(real(G_block[local_col, local_col, p])))
        end
        @inbounds sample_scale[tile] = max(sample_scale[tile], scale)
    end
end

"""Update convergence state and mark tiles that have reached the target rank."""
@kernel function convergence_ara_active_kernel!(
    ranks,
    small_sample_count,
    sample_scale,
    survivor_flags,
    G_block,
    active_idx,
    eps,
    j_before::Int,
    sample_size::Int,
    required_small::Int,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        tile = @inbounds active_idx[p]
        threshold = eps * max(one(eltype(sample_scale)), @inbounds(sample_scale[tile]))
        count = @inbounds small_sample_count[tile]
        survived = one(eltype(survivor_flags))

        @inbounds for local_col in 1:sample_size
            dR = abs(real(G_block[local_col, local_col, p]))
            if dR <= threshold
                count += 1
                if count >= required_small
                    ranks[tile] = convert(eltype(ranks), max(0, j_before + local_col - required_small))
                    survived = zero(eltype(survivor_flags))
                    break
                end
            else
                count = 0
            end
        end

        @inbounds small_sample_count[tile] = count
        @inbounds survivor_flags[p] = survived
    end
end

"""Compact the active-set index vector after a convergence pass."""
@kernel function compact_active_idx_kernel!(
    next_active_idx,
    survivor_pos,
    active_idx,
    survivor_flags,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        if @inbounds survivor_flags[p] != zero(eltype(survivor_flags))
            slot = Int(@inbounds survivor_pos[p])
            tile = @inbounds active_idx[p]
            @inbounds next_active_idx[slot] = tile
        end
    end
end

"""Compact active batch data into the front of a scratch buffer."""
@kernel function compact_blocks3d_kernel!(
    dst,
    src,
    survivor_pos,
    nrows::Int,
    ncols::Int,
    n_active::Int,
)
    row, col, p = @index(Global, NTuple)
    if p <= n_active && row <= nrows && col <= ncols
        slot = @inbounds survivor_pos[p]
        if slot != zero(eltype(survivor_pos))
            @inbounds dst[row, col, Int(slot)] = src[row, col, p]
        end
    end
end

"""Scatter a newly accepted `U` block back into the full factor storage."""
@kernel function scatter_U_block_kernel!(
    U,
    Y_block,
    active_idx,
    j_before::Int,
    sample_size::Int,
    n_active::Int,
)
    row, col, p = @index(Global, NTuple)
    if p <= n_active && row <= size(Y_block, 1) && col <= sample_size
        tile = @inbounds active_idx[p]
        @inbounds U[row, j_before + col, tile] = Y_block[row, col, p]
    end
end

"""Assign the final rank to tiles that remain active at loop exit."""
@kernel function finalize_active_ranks_kernel!(
    ranks,
    active_idx,
    j::Int,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        tile = @inbounds active_idx[p]
        @inbounds ranks[tile] = convert(eltype(ranks), j)
    end
end

"""Build raw pointer batches for backends that support pointer-batched GEMM."""
@kernel function build_batch_ptrs_kernel!(
    ptrs,
    base_ptr,
    batch_stride_bytes::Int,
    active_idx,
    scattered::Bool,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        tile = scattered ? Int(@inbounds(active_idx[p])) : p
        offset = (tile - 1) * batch_stride_bytes
        @inbounds ptrs[p] = convert(eltype(ptrs), reinterpret(UInt, base_ptr) + UInt(offset))
    end
end

"""Set up a pointer batch for the first `n_active` tiles of `A`."""
@inline function _setup_ptrs!(ptrs, A::AbstractArray{T,3}, active_idx, n_active::Int, backend; scattered::Bool=false) where {T}
    n_active > 0 || return ptrs
    build_batch_ptrs_kernel!(backend)(
        ptrs, pointer(A), stride(A, 3) * sizeof(T), active_idx, scattered, n_active;
        ndrange=n_active,
    )
    return ptrs
end

"""Compact the active tile index vector and return the new active count."""
@inline function _compact_active_idx!(next_active_idx, survivor_pos, active_idx, survivor_flags, n_active::Int, backend)
    n_active > 0 || return 0
    survivor_pos_view = @view survivor_pos[1:n_active]
    survivor_flags_view = @view survivor_flags[1:n_active]
    accumulate!(+, survivor_pos_view, survivor_flags_view)
    next_n_active = Int(Array(@view(survivor_pos[n_active:n_active]))[1])
    next_n_active > 0 || return 0
    compact_active_idx_kernel!(backend)(
        next_active_idx, survivor_pos, active_idx, survivor_flags, n_active;
        ndrange=n_active,
    )
    return next_n_active
end

"""Gather active tiles into a contiguous buffer when strided access is no longer valid."""
@inline function _gather_active_blocks!(dst, src::AbstractArray{T,3}, active_idx, nrows::Int, ncols::Int, n_active::Int, backend) where {T}
    n_active > 0 || return dst
    gather_active_blocks3d_kernel!(backend)(
        dst, src, active_idx, nrows, ncols, n_active;
        ndrange=(nrows, ncols, n_active),
    )
    return dst
end

"""
    _ara_bgemm!

Internal GEMM helper that switches between strided-batched, pointer-batched,
and gathered execution depending on whether the active set is still dense.
"""
@inline function _ara_bgemm!(
    transA::Char,
    transB::Char,
    alpha,
    backend,
    A::AbstractArray{T,3},
    a_cols::Int,
    A_ptrs,
    A_compact,
    active_idx,
    B::AbstractArray{T,3},
    B_ptrs,
    beta,
    C::AbstractArray{T,3},
    C_ptrs,
    n_active::Int,
    dense::Bool,
) where {T}
    n_active > 0 || return C
    A_use = a_cols == size(A, 2) ? A : @view(A[:, 1:a_cols, :])
    if dense
        gemm_batched!(transA, transB, alpha, A_use, B, beta, C)
    elseif supports_pointer_batched(backend)
        _setup_ptrs!(A_ptrs, A_use, active_idx, n_active, backend; scattered=true)
        _setup_ptrs!(B_ptrs, B, active_idx, n_active, backend; scattered=false)
        _setup_ptrs!(C_ptrs, C, active_idx, n_active, backend; scattered=false)
        gemm_batched_ptrs!(
            transA,
            transB,
            alpha,
            A_ptrs, @view(A_use[:, :, 1]),
            B_ptrs, @view(B[:, :, 1]),
            beta,
            C_ptrs, @view(C[:, :, 1]),
            n_active,
        )
    else
        _gather_active_blocks!(A_compact, A_use, active_idx, size(A_use, 1), size(A_use, 2), n_active, backend)
        gemm_batched!(transA, transB, alpha, @view(A_compact[:, 1:a_cols, 1:n_active]), B, beta, C)
    end
    return C
end

"""Mutable state carried across iterations of the adaptive compression loop."""
mutable struct ARAIterationState{IT,YT,GT}
    j::Int
    n_active::Int
    active_idx::IT
    spare_idx::IT
    Y_buffer::YT
    Y_spare::YT
    G_buffer::GT
    G_spare::GT
    dense::Bool
end

"""Run one adaptive sampling / orthogonalization / compaction step."""
function _ara_step!(
    U,
    ranks,
    source,
    ws::ARAWorkspace,
    state::ARAIterationState,
    sample_size::Int,
    eps_rt,
    required_small::Int,
    backend,
)
    T = eltype(U)
    transchar = T <: Real ? 'T' : 'C'
    j_before = state.j

    Y_current = @view state.Y_buffer[:, 1:sample_size, 1:state.n_active]
    G_raw = @view state.G_buffer[1:sample_size, 1:sample_size, 1:state.n_active]

    _sample_range!(
        Y_current,
        source,
        ws,
        state.active_idx,
        state.n_active,
        sample_size,
        state.dense,
        backend,
    )

    @unroll for pass in 1:2
        if j_before > 0
            Z_active = @view ws.Z[1:j_before, 1:sample_size, 1:state.n_active]

            _ara_bgemm!(
                transchar, 'N', one(T), backend,
                U, j_before, ws.U_ptrs, ws.Ucompact, state.active_idx,
                Y_current, ws.Y_ptrs,
                zero(T),
                Z_active, ws.Z_ptrs,
                state.n_active, state.dense,
            )

            _ara_bgemm!(
                'N', 'N', -one(T), backend,
                U, j_before, ws.U_ptrs, ws.Ucompact, state.active_idx,
                Z_active, ws.Z_ptrs,
                one(T),
                Y_current, ws.Y_ptrs,
                state.n_active, state.dense,
            )
        end

        syrk_batched!('L', transchar, one(T), Y_current, zero(T), G_raw)

        if pass == 1
            if j_before == 0
                initialize_ara_scale_active_kernel!(backend)(
                    ws.sample_scale, G_raw, state.active_idx, sample_size, state.n_active;
                    ndrange=state.n_active,
                )
            end

            convergence_ara_active_kernel!(backend)(
                ranks, ws.small_sample_count, ws.sample_scale, ws.survivor_flags,
                G_raw, state.active_idx, eps_rt, j_before, sample_size, required_small, state.n_active;
                ndrange=state.n_active,
            )

            state.n_active = _compact_active_idx!(
                state.spare_idx, ws.survivor_pos, state.active_idx, ws.survivor_flags, state.n_active, backend,
            )

            state.n_active == 0 && return nothing

            compacted = state.n_active != size(Y_current, 3)
            if compacted
                compact_blocks3d_kernel!(backend)(
                    state.Y_spare, state.Y_buffer, ws.survivor_pos, size(state.Y_buffer, 1), sample_size, size(Y_current, 3);
                    ndrange=(size(state.Y_buffer, 1), sample_size, size(Y_current, 3)),
                )
                compact_blocks3d_kernel!(backend)(
                    state.G_spare, state.G_buffer, ws.survivor_pos, sample_size, sample_size, size(Y_current, 3);
                    ndrange=(sample_size, sample_size, size(Y_current, 3)),
                )

                state.Y_buffer, state.Y_spare = state.Y_spare, state.Y_buffer
                state.G_buffer, state.G_spare = state.G_spare, state.G_buffer
                state.active_idx, state.spare_idx = state.spare_idx, state.active_idx
                state.dense = false
            end

            Y_current = @view state.Y_buffer[:, 1:sample_size, 1:state.n_active]
            G_raw = @view state.G_buffer[1:sample_size, 1:sample_size, 1:state.n_active]
        end

        potrf_status = @view ws.potrf_status[1:state.n_active]
        _, potrf_status_tmp = potrf_batched!('L', G_raw)
        copyto!(potrf_status, potrf_status_tmp)
        trsm_batched!('R', 'L', transchar, 'N', G_raw, Y_current)
    end

    scatter_U_block_kernel!(backend)(
        U, Y_current, state.active_idx, j_before, sample_size, state.n_active;
        ndrange=(size(Y_current, 1), sample_size, state.n_active),
    )

    state.j += sample_size
    return nothing
end

"""
    _ara_batched_impl!

Shared adaptive randomized compression loop used by both dense and operator
input paths.
"""
function _ara_batched_impl!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    source,
    m::Int,
    n::Int,
    batch_size::Int,
    max_rank::Int,
    block_size::Int,
    eps,
    nblocks_ref::Union{Nothing,Base.RefValue{Int}};
    backend=get_backend(U),
) where {T,RankT<:Integer}
    size(U, 1) == m || throw(DimensionMismatch("U must have $m rows"))
    size(V, 1) == n || throw(DimensionMismatch("V must have $n rows"))
    size(U, 2) >= max_rank || throw(DimensionMismatch("U must have at least $max_rank columns"))
    size(V, 2) >= max_rank || throw(DimensionMismatch("V must have at least $max_rank columns"))
    size(U, 3) == batch_size || throw(DimensionMismatch("U batch size must match the source"))
    size(V, 3) == batch_size || throw(DimensionMismatch("V batch size must match the source"))
    length(ranks) == batch_size || throw(DimensionMismatch("ranks length must match the batch size"))

    max_rank > 0 || throw(ArgumentError("max_rank must be positive"))
    block_size > 0 || throw(ArgumentError("block_size must be positive"))

    RT = typeof(real(zero(T)))
    eps_rt = RT(eps)^2
    required_small = block_size
    ptr_type = typeof(pointer(U))
    ws = _allocate_ara_workspace(backend, ptr_type, T, RankT, m, n, batch_size, max_rank, block_size)

    fill!(ranks, zero(RankT))
    fill!(ws.small_sample_count, Int32(0))
    fill!(ws.sample_scale, zero(RT))

    initialize_active_idx_kernel!(backend)(ws.active_idx, batch_size; ndrange=batch_size)

    state = ARAIterationState(
        0,
        batch_size,
        ws.active_idx,
        ws.next_active_idx,
        ws.Y,
        ws.Yscratch,
        ws.G,
        ws.Gscratch,
        true,
    )

    nblocks_ref === nothing || (nblocks_ref[] = 0)

    while state.n_active > 0 && state.j < max_rank
        nblocks_ref === nothing || (nblocks_ref[] += 1)
        sample_size = min(block_size, max_rank - state.j)

        _ara_step!(
            U,
            ranks,
            source,
            ws,
            state,
            sample_size,
            eps_rt,
            required_small,
            backend,
        )
    end

    if state.n_active > 0
        finalize_active_ranks_kernel!(backend)(ranks, state.active_idx, state.j, state.n_active; ndrange=state.n_active)
    end

    if state.j > 0
        V_active = @view V[:, 1:state.j, :]
        U_active = @view U[:, 1:state.j, :]
        _sample_corange!(V_active, source, U_active, backend)
    end

    return U, V, ranks
end
