function _retirement_outputs(ws_owner, count::Int)
    U = view(ws_owner.U, :, :, 1:count)
    V = view(ws_owner.V, :, :, 1:count)
    rr = view(ws_owner.ranks_slot, 1:count)
    ee = view(ws_owner.errors_slot, 1:count)
    fill!(U, zero(eltype(U)))
    fill!(V, zero(eltype(V)))
    return U, V, rr, ee
end

"""
    _finalize_wave!(Cout, run, ara_ws, slots, member_ids, progress, fixed,
                    arena, ws_owner; beta, tol, rel, compute)

Truncate one retired wave's accumulated ARA basis into low-rank factors and
scatter them into `Cout`. Dispatched on `run`'s fixed axis: fixing a column
needs `proj`/`sketch` scratch shaped like a fixed-row run's own fields (and
vice versa, fixing a row needs column-shaped scratch) since this computes
the *complementary* basis to whichever one `run`'s hot sampler builds
directly -- see [`apply_corange!`](@ref). The `ara_truncate!` argument order
(`Uh,Vh` vs `Vh,Uh`) is likewise swapped between the two: which of `apply_run!`'s
direct sample vs. this co-range apply yields the U- vs V-factor flips with the
sampling direction.
"""
function _finalize_wave!(Cout, run::RunCoupling{:column}, ara_ws,
                         slots::UnitRange{Int}, member_ids,
                         progress, j::Int, arena, ws_owner;
                         beta, tol, rel, compute)
    count = length(slots)
    count == 0 && return
    U, V, rr, ee = _retirement_outputs(ws_owner, count)
    s = maximum(view(progress, slots))
    logical = view(member_ids, slots)
    if s > 0
        _arena_reset_phase!(arena)
        a = _run_t_arena(arena)
        T = eltype(run.S)
        backend = get_backend(run.S)
        rA, rB, qk = size(run.S, 1), size(run.S, 2), run.qk
        Q = view(ara_ws.Q, :, 1:s, slots)
        proj = _workspace_array!(a, backend, T, rA, s, qk, count)
        sketch = _workspace_array!(a, backend, T, rB, qk, s, count)
        rC = run.betaV === nothing ? 0 : size(first(run.betaV), 2)
        bt = _workspace_array!(a, backend, T, rC, s, count)
        Z = _workspace_array!(a, backend, T, size(V, 1), s, count)
        apply_corange!(
            Z, run, Q, s; beta, compute, proj, sketch, beta_tmp=bt,
            slot0=first(slots))
        Uh = _workspace_array!(a, backend, T, size(U, 1), s, count)
        Vh = _workspace_array!(a, backend, T, size(V, 1), s, count)
        ara_truncate!(
            Uh, Vh, rr, ee, Q, Z; tol, relative=rel,
            maxrank=min(s, size(U, 2)), compute)
        copyto!(view(U, :, 1:s, :), Uh)
        copyto!(view(V, :, 1:s, :), Vh)
    else
        fill!(rr, zero(eltype(rr)))
        fill!(ee, 0.0)
    end
    qm = grid_size(Cout)[1]
    qn = grid_size(Cout)[2]
    outslots = view(ws_owner.output_slots, 1:count)
    inslots = view(ws_owner.output_slots_inner, 1:count)
    @inbounds for p in 1:count
        outslots[p] = j + (logical[p] - 1) * qn      # tile_linear_index(TileRowMajor, ...)
        inslots[p] = logical[p] + (j - 1) * qm        # tile_linear_index(TileColMajor, ...)
    end
    _store_tlr_run!(
        Cout, U, V, rr, ee, outslots, inslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function _finalize_wave!(Cout, run::RunCoupling{:row}, ara_ws,
                         slots::UnitRange{Int}, member_ids, progress,
                         i::Int, arena, ws_owner;
                         beta, tol, rel, compute)
    count = length(slots)
    count == 0 && return
    U, V, rr, ee = _retirement_outputs(ws_owner, count)
    s = maximum(view(progress, slots))
    logical = view(member_ids, slots)
    if s > 0
        _arena_reset_phase!(arena)
        a = _run_t_arena(arena)
        T = eltype(run.S)
        backend = get_backend(run.S)
        rA, rB, qk = size(run.S, 1), size(run.S, 2), run.qk
        Q = view(ara_ws.Q, :, 1:s, slots)
        proj = _workspace_array!(a, backend, T, rB, s, qk, count)
        sketch = _workspace_array!(a, backend, T, rA, qk, s, count)
        rC = run.betaU === nothing ? 0 : size(first(run.betaU), 2)
        bt = _workspace_array!(a, backend, T, rC, s, count)
        L = _workspace_array!(a, backend, T, size(U, 1), s, count)
        apply_corange!(
            L, run, Q, s; beta, compute, proj, sketch, beta_tmp=bt,
            slot0=first(slots))
        Uh = _workspace_array!(a, backend, T, size(U, 1), s, count)
        Vh = _workspace_array!(a, backend, T, size(V, 1), s, count)
        ara_truncate!(
            Vh, Uh, rr, ee, Q, L; tol, relative=rel,
            maxrank=min(s, size(U, 2)), compute)
        copyto!(view(U, :, 1:s, :), Uh)
        copyto!(view(V, :, 1:s, :), Vh)
    else
        fill!(rr, zero(eltype(rr)))
        fill!(ee, 0.0)
    end
    qm = grid_size(Cout)[1]
    qn = grid_size(Cout)[2]
    outslots = view(ws_owner.output_slots, 1:count)
    inslots = view(ws_owner.output_slots_inner, 1:count)
    @inbounds for p in 1:count
        outslots[p] = logical[p] + (i - 1) * qn        # tile_linear_index(TileRowMajor, ...)
        inslots[p] = i + (logical[p] - 1) * qm          # tile_linear_index(TileColMajor, ...)
    end
    _store_tlr_run!(
        Cout, U, V, rr, ee, outslots, inslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function _rolling_lane_loop!(Cout, run, ara_ws, allmembers::AbstractVector{Int},
                             fixed::Int, ops, arena, ws_owner;
                             beta, eps_rel, r_required, tol, rel, compute)
    cap = ws_owner.key.capacity
    member_ids = ws_owner.member_ids
    progress = ws_owner.progress
    initial = min(cap, length(allmembers))
    copyto!(member_ids, 1, allmembers, 1, initial)
    fill!(progress, 0)
    pending = initial + 1
    nactive = initial
    ara_reset!(ara_ws.state, ara_ws.block, size(ara_ws.Q, 2))
    copyto!(ara_ws.state.samples_host, ara_ws.state.samples)
    fill!(ara_ws.Q, zero(eltype(ara_ws.Q)))

    sampler = (Y, width, _) -> apply_run!(Y, run, width, nactive; beta, compute)
    swapper = (p, q) -> _swap_run_members!(run, p, q)

    while nactive > 0 || pending <= length(allmembers)
        if nactive == 0
            nnew = min(cap, length(allmembers) - pending + 1)
            slots = 1:nnew
            ids = view(member_ids, slots)
            copyto!(ids, view(allmembers, pending:(pending + nnew - 1)))
            _arena_reset_phase!(arena)
            admit_wave!(
                run, ops, ids, slots, fixed, arena;
                C=iszero(beta) ? nothing : Cout, compute)
            ara_reset_slots!(ara_ws, 1, nnew)
            fill!(view(progress, slots), 0)
            pending += nnew
            nactive = nnew
            ara_ws = rebind_ara_phase(ara_ws, arena)
            rebind_sampling_scratch!(run, arena)
        end

        info = ara_packed_pass!(
            ara_ws, sampler, nactive, member_ids, progress;
            eps_rel, r_required, compute, swap_member! = swapper)
        nactive = info.nactive
        retired = info.retired
        isempty(retired) && continue

        _finalize_wave!(
            Cout, run, ara_ws, retired, member_ids, progress, fixed,
            arena, ws_owner; beta, tol, rel, compute)

        nnew = min(cap - nactive, max(length(allmembers) - pending + 1, 0))
        if nnew > 0
            slots = (nactive + 1):(nactive + nnew)
            ids = view(member_ids, slots)
            copyto!(ids, view(allmembers, pending:(pending + nnew - 1)))
            _arena_reset_phase!(arena)
            admit_wave!(
                run, ops, ids, slots, fixed, arena;
                C=iszero(beta) ? nothing : Cout, compute)
            ara_reset_slots!(ara_ws, first(slots), nnew)
            fill!(view(progress, slots), 0)
            pending += nnew
            nactive += nnew
        end
        if nactive > 0
            ara_ws = rebind_ara_phase(ara_ws, arena)
            rebind_sampling_scratch!(run, arena)
        end
    end
    return Cout
end
