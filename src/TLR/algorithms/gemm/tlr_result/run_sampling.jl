@kernel function _scatter_run_factor_kernel!(dest, src, ranks_slot,
                                             slot_to_member)
    i, k, slot = @index(Global, NTuple)
    @inbounds dest[i, k, Int(slot_to_member[slot])] =
        k <= Int(ranks_slot[slot]) ? src[i, k, slot] : zero(eltype(dest))
end

@kernel function _scatter_run_diagnostic_kernel!(ranks, err_sq, ranks_slot,
                                                 err_slot, slot_to_member)
    slot = @index(Global, Linear)
    @inbounds begin
        member = Int(slot_to_member[slot])
        ranks[member] = ranks_slot[slot]
        err_sq[member] = err_slot[slot]
    end
end

"""
    range_find_column_run!(U, V, ranks, err_sq, ops, rows, j; ...)

Fixed-column packed-run RangeFind. The tiles `(rows, j)` share one `H_ℓj` per sampling
pass and occupy a physically contiguous active prefix. Members that converge
are swap-removed into a retired suffix; the complete run is truncated as one
batch and scattered once back to the caller's original row order.

The canonical row-major API uses this family for the right-selected `NT` case,
where a fixed logical-B column shares `H_ℓj`. The `NN` right path and the
left-sampling paths use the symmetric fixed-row driver below.
"""
function range_find_column_run!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                                ranks, err_sq,
                                ops::LogicalTLROperands, rows, j::Integer;
                                alpha, beta=false, C=nothing,
                                eps_rel::Real, r_required::Int=10,
                                tol::Real, rel::Bool=true,
                                block::Int=32,
                                rA::Int=rankdim(ops.au),
                                rB::Int=rankdim(ops.bv),
                                compute=nothing, arena=nothing,
                                ranks_slot=nothing, err_slot=nothing,
                                slot_to_member=nothing) where {T}
    row_ids = collect(Int, rows)
    nmember = length(row_ids)
    bm, maxrank, countU = size(U)
    bn, maxrankV, countV = size(V)
    countU == countV == nmember == length(ranks) == length(err_sq) ||
        throw(DimensionMismatch("run outputs and diagnostics must match length(rows)"))
    maxrank == maxrankV ||
        throw(DimensionMismatch("U and V must have the same rank capacity"))
    nmember == 0 && return (; passes=0, active_counts=Int[], member_ids=Int[])
    fill!(U, zero(T)); fill!(V, zero(T))

    if maxrank == 0
        fill!(ranks, zero(eltype(ranks)))
        fill!(err_sq, 0.0)
        return (; passes=0, active_counts=Int[], member_ids=collect(1:nmember))
    end

    _arena_reset!(arena)
    backend = get_backend(U)
    blk = min(block, max(maxrank, 1))
    run = ColumnRunCoupling(ops, row_ids, j;
                            alpha, beta, C, block=blk, maxrank,
                            rA, rB, compute, arena)
    ws = ARAWorkspace(T, backend, bm, maxrank, nmember; block=blk, arena)
    member_ids = collect(1:nmember)
    sampler = function (Y, sketch_width, active_ids)
        apply_right_run!(Y, run, sketch_width, length(active_ids); beta, compute)
    end
    basis = ara_build_basis_packed!(
        ws, sampler, member_ids; eps_rel, r_required, compute,
        swap_member! = (p, q) -> _swap_column_run_members!(run, p, q),
    )
    sQ = size(basis.Q, 2)

    if sQ == 0
        fill!(ranks, zero(eltype(ranks)))
        fill!(err_sq, 0.0)
        return (; passes=basis.passes, active_counts=basis.active_counts,
                member_ids=basis.member_ids)
    end

    _arena_reset_phase!(arena)
    tarena = _run_t_arena(arena)
    G = _workspace_array!(tarena, backend, T, rA, sQ, run.qk, nmember)
    Wbuf = _workspace_array!(tarena, backend, T, rB, run.qk, sQ, nmember)
    rC = run.betaV === nothing ? 0 : size(first(run.betaV), 2)
    beta_tmp = _workspace_array!(tarena, backend, T, rC, sQ, nmember)
    Z = _workspace_array!(tarena, backend, T, bn, sQ, nmember)
    apply_left_run!(
        Z, run, basis.Q, sQ; beta, compute, G, Wbuf, beta_tmp)
    Uh = _workspace_array!(tarena, backend, T, bm, sQ, nmember)
    Vh = _workspace_array!(tarena, backend, T, bn, sQ, nmember)
    ranks_slot === nothing &&
        (ranks_slot = allocate(backend, eltype(ranks), nmember))
    err_slot === nothing &&
        (err_slot = allocate(backend, eltype(err_sq), nmember))
    ara_truncate!(Uh, Vh, ranks_slot, err_slot, basis.Q, Z;
                  tol, relative=rel, maxrank=sQ, compute)

    slot_to_member === nothing &&
        (slot_to_member = allocate(backend, Int32, nmember))
    copyto!(view(slot_to_member, 1:nmember), Int32.(basis.member_ids))
    _scatter_run_factor_kernel!(backend)(
        view(U, :, 1:sQ, :), Uh, ranks_slot, slot_to_member;
        ndrange=(bm, sQ, nmember),
    )
    _scatter_run_factor_kernel!(backend)(
        view(V, :, 1:sQ, :), Vh, ranks_slot, slot_to_member;
        ndrange=(bn, sQ, nmember),
    )
    _scatter_run_diagnostic_kernel!(backend)(
        ranks, err_sq, ranks_slot, err_slot, slot_to_member; ndrange=nmember,
    )
    return (; passes=basis.passes, active_counts=basis.active_counts,
            member_ids=basis.member_ids)
end

@inline function _validate_row_run_outputs(U, V, ranks, err_sq, cols)
    nmember = length(cols)
    size(U, 3) == size(V, 3) == nmember == length(ranks) == length(err_sq) ||
        throw(DimensionMismatch("run outputs and diagnostics must match length(cols)"))
    size(U, 2) == size(V, 2) ||
        throw(DimensionMismatch("U and V must have the same rank capacity"))
    return nmember
end

function _scatter_range_run!(U, V, ranks, err_sq, Uh, Vh, ranks_slot, err_slot,
                             member_ids, sQ, slot_to_member=nothing)
    backend = get_backend(U)
    nmember = length(member_ids)
    slot_to_member === nothing &&
        (slot_to_member = allocate(backend, Int32, nmember))
    copyto!(view(slot_to_member, 1:nmember), Int32.(member_ids))
    _scatter_run_factor_kernel!(backend)(
        view(U, :, 1:sQ, :), Uh, ranks_slot, slot_to_member;
        ndrange=(size(U, 1), sQ, nmember),
    )
    _scatter_run_factor_kernel!(backend)(
        view(V, :, 1:sQ, :), Vh, ranks_slot, slot_to_member;
        ndrange=(size(V, 1), sQ, nmember),
    )
    _scatter_run_diagnostic_kernel!(backend)(
        ranks, err_sq, ranks_slot, err_slot, slot_to_member; ndrange=nmember,
    )
    return nothing
end

"""
    range_find_row_right_run!(...)

Fixed-output-row packed-run path using repeated `XΩ` samples. This is the
canonical `NN` path: A's logical tile row is the zero-copy terminal stack.
"""
function range_find_row_right_run!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                                   ranks, err_sq, ops::LogicalTLROperands,
                                   i::Integer, cols;
                                   alpha, beta=false, C=nothing,
                                   eps_rel::Real, r_required::Int=10,
                                   tol::Real, rel::Bool=true,
                                   block::Int=32,
                                   rA::Int=rankdim(ops.au),
                                   rB::Int=rankdim(ops.bv),
                                   compute=nothing, arena=nothing,
                                   ranks_slot=nothing, err_slot=nothing,
                                   slot_to_member=nothing) where {T}
    col_ids = collect(Int, cols)
    nmember = _validate_row_run_outputs(U, V, ranks, err_sq, col_ids)
    nmember == 0 &&
        return (; passes=0, active_counts=Int[], member_ids=Int[], side=:right)
    fill!(U, zero(T)); fill!(V, zero(T))
    maxrank = size(U, 2)
    if maxrank == 0
        fill!(ranks, zero(eltype(ranks))); fill!(err_sq, 0.0)
        return (; passes=0, active_counts=Int[], member_ids=collect(1:nmember),
                side=:right)
    end

    _arena_reset!(arena)
    backend = get_backend(U)
    blk = min(block, max(maxrank, 1))
    run = RowRightRunCoupling(
        ops, i, col_ids; alpha, beta, C, block=blk, maxrank,
        rA, rB, compute, arena, index_scratch=slot_to_member,
    )
    ws = ARAWorkspace(
        T, backend, size(U, 1), maxrank, nmember; block=blk, arena)
    member_ids = collect(1:nmember)
    sampler = function (Y, sketch_width, active_ids)
        apply_right_row_run!(Y, run, sketch_width, length(active_ids); beta, compute)
    end
    basis = ara_build_basis_packed!(
        ws, sampler, member_ids; eps_rel, r_required, compute,
        swap_member! = (p, q) -> _swap_row_right_members!(run, p, q),
    )
    sQ = size(basis.Q, 2)
    sQ == 0 &&
        return (; passes=basis.passes, active_counts=basis.active_counts,
                member_ids=basis.member_ids, side=:right)

    _arena_reset_phase!(arena)
    tarena = _run_t_arena(arena)
    G = _workspace_array!(tarena, backend, T, rA, sQ, run.qk, nmember)
    Wbuf = _workspace_array!(tarena, backend, T, rB, run.qk, sQ, nmember)
    rC = run.betaV === nothing ? 0 : size(first(run.betaV), 2)
    beta_tmp = _workspace_array!(tarena, backend, T, rC, sQ, nmember)
    Z = _workspace_array!(tarena, backend, T, size(V, 1), sQ, nmember)
    apply_left_row_run!(
        Z, run, basis.Q, sQ; beta, compute, G, Wbuf, beta_tmp)
    Uh = _workspace_array!(tarena, backend, T, size(U, 1), sQ, nmember)
    Vh = _workspace_array!(tarena, backend, T, size(V, 1), sQ, nmember)
    ranks_slot === nothing &&
        (ranks_slot = allocate(backend, eltype(ranks), nmember))
    err_slot === nothing &&
        (err_slot = allocate(backend, eltype(err_sq), nmember))
    ara_truncate!(Uh, Vh, ranks_slot, err_slot, basis.Q, Z;
                  tol, relative=rel, maxrank=sQ, compute)
    _scatter_range_run!(U, V, ranks, err_sq, Uh, Vh, ranks_slot, err_slot,
                        basis.member_ids, sQ, slot_to_member)
    return (; passes=basis.passes, active_counts=basis.active_counts,
            member_ids=basis.member_ids, side=:right)
end

"""
    range_find_row_left_run!(...)

Fixed-output-row packed-run path using repeated `XᵀΩ` samples. The ARA basis
is the right output basis; truncation is applied to `Xᵀ` and its factors are
swapped back to the conventional `X ≈ U Vᵀ` orientation.
"""
function range_find_row_left_run!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                                  ranks, err_sq, ops::LogicalTLROperands,
                                  i::Integer, cols;
                                  alpha, beta=false, C=nothing,
                                  eps_rel::Real, r_required::Int=10,
                                  tol::Real, rel::Bool=true,
                                  block::Int=32,
                                  rA::Int=rankdim(ops.au),
                                  rB::Int=rankdim(ops.bv),
                                  compute=nothing, arena=nothing,
                                  ranks_slot=nothing, err_slot=nothing,
                                  slot_to_member=nothing) where {T}
    col_ids = collect(Int, cols)
    nmember = _validate_row_run_outputs(U, V, ranks, err_sq, col_ids)
    nmember == 0 &&
        return (; passes=0, active_counts=Int[], member_ids=Int[], side=:left)
    fill!(U, zero(T)); fill!(V, zero(T))
    maxrank = size(U, 2)
    if maxrank == 0
        fill!(ranks, zero(eltype(ranks))); fill!(err_sq, 0.0)
        return (; passes=0, active_counts=Int[], member_ids=collect(1:nmember),
                side=:left)
    end

    _arena_reset!(arena)
    backend = get_backend(U)
    blk = min(block, max(maxrank, 1))
    run = RowLeftRunCoupling(
        ops, i, col_ids; alpha, beta, C, block=blk, maxrank,
        rA, rB, compute, arena,
    )
    ws = ARAWorkspace(
        T, backend, size(V, 1), maxrank, nmember; block=blk, arena)
    member_ids = collect(1:nmember)
    sampler = function (Y, sketch_width, active_ids)
        apply_left_row_run!(Y, run, sketch_width, length(active_ids); beta, compute)
    end
    basis = ara_build_basis_packed!(
        ws, sampler, member_ids; eps_rel, r_required, compute,
        swap_member! = (p, q) -> _swap_row_left_members!(run, p, q),
    )
    sQ = size(basis.Q, 2)
    sQ == 0 &&
        return (; passes=basis.passes, active_counts=basis.active_counts,
                member_ids=basis.member_ids, side=:left)

    _arena_reset_phase!(arena)
    tarena = _run_t_arena(arena)
    H = _workspace_array!(tarena, backend, T, rB, sQ, run.qk, nmember)
    Tbuf = _workspace_array!(tarena, backend, T, rA, run.qk, sQ, nmember)
    rC = run.betaU === nothing ? 0 : size(first(run.betaU), 2)
    beta_tmp = _workspace_array!(tarena, backend, T, rC, sQ, nmember)
    L = _workspace_array!(tarena, backend, T, size(U, 1), sQ, nmember)
    apply_right_row_run!(
        L, run, basis.Q, sQ; beta, compute, H, Tbuf, beta_tmp)
    Uh = _workspace_array!(tarena, backend, T, size(U, 1), sQ, nmember)
    Vh = _workspace_array!(tarena, backend, T, size(V, 1), sQ, nmember)
    ranks_slot === nothing &&
        (ranks_slot = allocate(backend, eltype(ranks), nmember))
    err_slot === nothing &&
        (err_slot = allocate(backend, eltype(err_sq), nmember))
    # Xᵀ ≈ Q_R Lᵀ. Truncating the transpose returns (V, U).
    ara_truncate!(Vh, Uh, ranks_slot, err_slot, basis.Q, L;
                  tol, relative=rel, maxrank=sQ, compute)
    _scatter_range_run!(U, V, ranks, err_sq, Uh, Vh, ranks_slot, err_slot,
                        basis.member_ids, sQ, slot_to_member)
    return (; passes=basis.passes, active_counts=basis.active_counts,
            member_ids=basis.member_ids, side=:left)
end
