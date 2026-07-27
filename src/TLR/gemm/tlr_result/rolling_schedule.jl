mutable struct TLRGemmScheduleStats
    active_counts::Vector{Int}
    pending_counts::Vector{Int}
    retirement_waves::Vector{Int}
    admissions::Int
    passes::Int
    padded_projection_columns::Int
    discarded_terminal_columns::Int
    sampling_ns::Int
    orthogonalization_ns::Int
    finalization_ns::Int
    admission_ns::Int
end

TLRGemmScheduleStats() = TLRGemmScheduleStats(
    Int[], Int[], Int[], 0, 0, 0, 0, 0, 0, 0, 0)

@inline function _stats_sync(stats, x)
    stats === nothing || KernelAbstractions.synchronize(get_backend(x))
end

@inline function _record_pass!(stats, nactive, pending, info)
    stats === nothing && return
    push!(stats.active_counts, nactive)
    push!(stats.pending_counts, pending)
    stats.passes += 1
    stats.padded_projection_columns +=
        sum(max(info.projection_width - p, 0) for p in info.progress_before)
    stats.discarded_terminal_columns += info.discarded
end

function _retirement_outputs(ws_owner, count::Int)
    U = view(ws_owner.U, :, :, 1:count)
    V = view(ws_owner.V, :, :, 1:count)
    rr = view(ws_owner.ranks_slot, 1:count)
    ee = view(ws_owner.errors_slot, 1:count)
    fill!(U, zero(eltype(U)))
    fill!(V, zero(eltype(V)))
    return U, V, rr, ee
end

function _finalize_column_wave!(Cout, run::ColumnRunCoupling, ara_ws,
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
        G = _workspace_array!(a, backend, T, rA, s, qk, count)
        W = _workspace_array!(a, backend, T, rB, qk, s, count)
        rC = run.betaV === nothing ? 0 : size(first(run.betaV), 2)
        bt = _workspace_array!(a, backend, T, rC, s, count)
        Z = _workspace_array!(a, backend, T, size(V, 1), s, count)
        apply_left_run!(
            Z, run, Q, s; beta, compute, G, Wbuf=W, beta_tmp=bt,
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
    outslots = view(ws_owner.output_slots, 1:count)
    @inbounds for p in 1:count
        outslots[p] = j + (logical[p] - 1) * grid_size(Cout)[2]
    end
    _store_tlr_run!(
        Cout, U, V, rr, ee, outslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function _finalize_row_wave!(Cout, run::RowRightRunCoupling, ara_ws,
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
        G = _workspace_array!(a, backend, T, rA, s, qk, count)
        W = _workspace_array!(a, backend, T, rB, qk, s, count)
        rC = run.betaV === nothing ? 0 : size(first(run.betaV), 2)
        bt = _workspace_array!(a, backend, T, rC, s, count)
        Z = _workspace_array!(a, backend, T, size(V, 1), s, count)
        apply_left_row_run!(
            Z, run, Q, s; beta, compute, G, Wbuf=W, beta_tmp=bt,
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
    outslots = view(ws_owner.output_slots, 1:count)
    @inbounds for p in 1:count
        outslots[p] = logical[p] + (i - 1) * grid_size(Cout)[2]
    end
    _store_tlr_run!(
        Cout, U, V, rr, ee, outslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function _finalize_row_wave!(Cout, run::RowLeftRunCoupling, ara_ws,
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
        H = _workspace_array!(a, backend, T, rB, s, qk, count)
        Tb = _workspace_array!(a, backend, T, rA, qk, s, count)
        rC = run.betaU === nothing ? 0 : size(first(run.betaU), 2)
        bt = _workspace_array!(a, backend, T, rC, s, count)
        L = _workspace_array!(a, backend, T, size(U, 1), s, count)
        apply_right_row_run!(
            L, run, Q, s; beta, compute, H, Tbuf=Tb, beta_tmp=bt,
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
    outslots = view(ws_owner.output_slots, 1:count)
    @inbounds for p in 1:count
        outslots[p] = logical[p] + (i - 1) * grid_size(Cout)[2]
    end
    _store_tlr_run!(
        Cout, U, V, rr, ee, outslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function _rolling_lane_loop!(Cout, run, ara_ws, allmembers::AbstractVector{Int},
                             fixed::Int, ops, arena, ws_owner;
                             beta, eps_rel, r_required, tol, rel, compute,
                             side::Symbol, stats=nothing)
    cap = ws_owner.key.capacity
    member_ids = ws_owner.member_ids
    progress = ws_owner.progress
    initial = min(cap, length(allmembers))
    copyto!(member_ids, 1, allmembers, 1, initial)
    fill!(progress, 0)
    pending = initial + 1
    nactive = initial
    stats === nothing || (stats.admissions += initial)
    ara_reset!(ara_ws.state, ara_ws.block, size(ara_ws.Q, 2))
    copyto!(ara_ws.state.samples_host, ara_ws.state.samples)
    fill!(ara_ws.Q, zero(eltype(ara_ws.Q)))

    raw_sampler = if run isa ColumnRunCoupling
        (Y, width, _) -> apply_right_run!(
            Y, run, width, nactive; beta, compute)
    elseif run isa RowRightRunCoupling
        (Y, width, _) -> apply_right_row_run!(
            Y, run, width, nactive; beta, compute)
    else
        (Y, width, _) -> apply_left_row_run!(
            Y, run, width, nactive; beta, compute)
    end
    sampler = if stats === nothing
        raw_sampler
    else
        function (Y, width, ids)
            _stats_sync(stats, Y)
            ts = time_ns()
            raw_sampler(Y, width, ids)
            _stats_sync(stats, Y)
            stats.sampling_ns += time_ns() - ts
            return Y
        end
    end
    swapper = run isa ColumnRunCoupling ?
        ((p, q) -> _swap_column_run_members!(run, p, q)) :
        run isa RowRightRunCoupling ?
        ((p, q) -> _swap_row_right_members!(run, p, q)) :
        ((p, q) -> _swap_row_left_members!(run, p, q))

    while nactive > 0 || pending <= length(allmembers)
        if nactive == 0
            nnew = min(cap, length(allmembers) - pending + 1)
            slots = 1:nnew
            ids = view(member_ids, slots)
            copyto!(ids, view(allmembers, pending:(pending + nnew - 1)))
            _arena_reset_phase!(arena)
            admit_wave!(
                run, ops, ids, slots, fixed, arena;
                C=iszero(beta) ? nothing : logical_operand(Cout), compute)
            ara_reset_slots!(ara_ws, 1, nnew)
            fill!(view(progress, slots), 0)
            pending += nnew
            nactive = nnew
            stats === nothing || (stats.admissions += nnew)
            ara_ws = rebind_ara_phase(ara_ws, arena)
            rebind_sampling_scratch!(run, arena)
        end

        progress_before = copy(view(progress, 1:nactive))
        active_before = nactive
        _stats_sync(stats, ara_ws.Q)
        t0 = time_ns()
        sampling_before = stats === nothing ? 0 : stats.sampling_ns
        info = ara_packed_pass!(
            ara_ws, sampler, nactive, member_ids, progress;
            eps_rel, r_required, compute, swap_member! = swapper)
        _stats_sync(stats, ara_ws.Q)
        stats === nothing || (stats.orthogonalization_ns +=
            time_ns() - t0 - (stats.sampling_ns - sampling_before))
        nactive = info.nactive
        pending_count = max(length(allmembers) - pending + 1, 0)
        _record_pass!(stats, active_before, pending_count,
                      (; info..., progress_before))
        retired = info.retired
        isempty(retired) && continue
        stats === nothing || push!(stats.retirement_waves, length(retired))

        _stats_sync(stats, ara_ws.Q)
        t0 = time_ns()
        if run isa ColumnRunCoupling
            _finalize_column_wave!(
                Cout, run, ara_ws, retired, member_ids, progress, fixed,
                arena, ws_owner; beta, tol, rel, compute)
        else
            _finalize_row_wave!(
                Cout, run, ara_ws, retired, member_ids, progress, fixed,
                arena, ws_owner; beta, tol, rel, compute)
        end
        _stats_sync(stats, ara_ws.Q)
        stats === nothing || (stats.finalization_ns += time_ns() - t0)

        nnew = min(cap - nactive, max(length(allmembers) - pending + 1, 0))
        if nnew > 0
            slots = (nactive + 1):(nactive + nnew)
            ids = view(member_ids, slots)
            copyto!(ids, view(allmembers, pending:(pending + nnew - 1)))
            _stats_sync(stats, ara_ws.Q)
            t0 = time_ns()
            _arena_reset_phase!(arena)
            admit_wave!(
                run, ops, ids, slots, fixed, arena;
                C=iszero(beta) ? nothing : logical_operand(Cout), compute)
            ara_reset_slots!(ara_ws, first(slots), nnew)
            fill!(view(progress, slots), 0)
            pending += nnew
            nactive += nnew
            stats === nothing || begin
                _stats_sync(stats, ara_ws.Q)
                stats.admissions += nnew
                stats.admission_ns += time_ns() - t0
            end
        end
        if nactive > 0
            ara_ws = rebind_ara_phase(ara_ws, arena)
            rebind_sampling_scratch!(run, arena)
        end
    end
    return stats
end
