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
    _capped_capture_sq!(capture, run, Q, s, progress, cap, block, Gbuf, Ebuf;
                        beta, compute, proj, sketch, beta_tmp, slot0)
        -> capture or nothing

Squared range-capture error `‖X - QQᴴX‖_F²` of the members whose basis stopped
at the ARA cap (`progress[p] >= cap`), and zero for the others, for
`ara_truncate!(...; capture)`, which adds it to the reported residual and
leaves the rank choice as it was. Returns `nothing`, computing nothing, when no
member reached the cap.

ARA's stopping test (`r_required` consecutive negligible samples) bounds what `Q`
misses only for members that converged. A member stopped by the cap can miss
much more, and without this its residual would be the truncation error alone,
which is zero when the cap holds the whole basis. The miss is estimated a
posteriori from fresh Gaussian vectors (Halko, Martinsson and Tropp, SIAM Rev.
53(2), 2011, §4.3): for `G` with `w` columns of independent N(0, 1) entries,
`E‖Xᴴ(I - QQᴴ)G‖_F² = w‖(I - QQᴴ)X‖_F²`, with relative standard deviation at most
`√(2/w)` (the Gaussian trace estimator, Avron and Toledo, J. ACM 58(2), 2011).
Here `w = min(block, s)`.

`Gbuf` (the rows of `Q`) and `Ebuf` (the rows of `Xᴴ`) are the caller's
truncation outputs before they are written: `Gbuf` holds `G`, and `Ebuf` first
`QᴴG`, then `Xᴴ(I - QQᴴ)G`.
"""
function _capped_capture_sq!(capture, run, Q, s::Int, progress, cap::Int,
                             block::Int, Gbuf, Ebuf; beta, compute, proj,
                             sketch, beta_tmp, slot0::Int)
    capped = [progress[p] >= cap for p in eachindex(progress)]
    any(capped) || return nothing
    T = eltype(Q)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    w = min(block, s)
    G = view(Gbuf, :, 1:w, :)
    QG = view(Ebuf, 1:s, 1:w, :)
    E = view(Ebuf, :, 1:w, :)

    # Gaussian block projected off the basis
    Random.randn!(G)
    precision_gemm_batched!(adjoint_blas_char(T), 'N', one(T), Q, G,
                            zero(T), QG, mode)
    precision_gemm_batched!('N', 'N', -one(T), Q, QG, one(T), G, mode)

    # operator samples and the estimate
    apply_corange!(E, run, G, w; beta, compute, proj, sketch, beta_tmp, slot0)
    batch_frobenius_norms_sq!(capture, E)
    est = Array(capture)
    @inbounds for p in eachindex(est)
        est[p] = capped[p] ? est[p] / w : 0.0
    end
    copyto!(capture, est)
    return capture
end

"""
    _finalize_wave!(Cout, run, ara_ws, slots, member_ids, progress, fixed,
                    arena, ws_owner; beta, tol, rel, compute)

Compute the complementary basis for a retired wave, truncate it, and scatter
the factors into `Cout`. Fixed-column and fixed-row runs swap scratch shapes and
`ara_truncate!` factor order because direct and co-range sampling exchange the
outer/inner roles; see [`apply_corange!`](@ref).
"""
function _finalize_wave!(Cout, run::RunCoupling{:column}, ara_ws,
                         slots::UnitRange{Int}, member_ids,
                         progress, j::Int, arena, ws_owner;
                         beta, tol, rel, compute)
    count = length(slots)
    count == 0 && return

    # retired buffers and basis width
    U, V, rr, ee = _retirement_outputs(ws_owner, count)
    s = maximum(view(progress, slots))
    logical = view(member_ids, slots)

    # complementary basis and truncation
    if s > 0
        arena_reset_phase!(arena)
        a = run_t_arena(arena)
        T = eltype(run.S)
        backend = get_backend(run.S)
        rA, rB, qk = size(run.S, 1), size(run.S, 2), run.qk
        Q = view(ara_ws.Q, :, 1:s, slots)
        proj = workspace_array!(a, backend, T, rA, s, qk, count)
        sketch = workspace_array!(a, backend, T, rB, qk, s, count)
        rC = run.betaV === nothing ? 0 : size(first(run.betaV), 2)
        bt = workspace_array!(a, backend, T, rC, s, count)
        Z = workspace_array!(a, backend, T, size(V, 1), s, count)

        apply_corange!(
            Z, run, Q, s; beta, compute, proj, sketch, beta_tmp=bt,
            slot0=first(slots))

        Uh = workspace_array!(a, backend, T, size(U, 1), s, count)
        Vh = workspace_array!(a, backend, T, size(V, 1), s, count)
        capture = _capped_capture_sq!(
            ee, run, Q, s, view(progress, slots), size(ara_ws.Q, 2),
            ara_ws.block, Uh, Vh; beta, compute, proj, sketch, beta_tmp=bt,
            slot0=first(slots))
        ara_truncate!(
            Uh, Vh, rr, ee, Q, Z; tol, relative=rel,
            maxrank=min(s, size(U, 2)), capture, compute)

        copyto!(view(U, :, 1:s, :), Uh)
        copyto!(view(V, :, 1:s, :), Vh)
    else
        fill!(rr, zero(eltype(rr)))
        fill!(ee, 0.0)
    end

    # canonical factor and diagnostic slots
    qm = grid_size(Cout)[1]
    qn = grid_size(Cout)[2]
    outslots = view(ws_owner.output_slots, 1:count)
    inslots = view(ws_owner.output_slots_inner, 1:count)
    @inbounds for p in 1:count
        outslots[p] = j + (logical[p] - 1) * qn      # tile_linear_index(TileRowMajor, ...)
        inslots[p] = logical[p] + (j - 1) * qm        # tile_linear_index(TileColMajor, ...)
    end
    store_tlr_run!(
        Cout, U, V, rr, ee, outslots, inslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function _finalize_wave!(Cout, run::RunCoupling{:row}, ara_ws,
                         slots::UnitRange{Int}, member_ids, progress,
                         i::Int, arena, ws_owner;
                         beta, tol, rel, compute)
    count = length(slots)
    count == 0 && return

    # retired buffers and basis width
    U, V, rr, ee = _retirement_outputs(ws_owner, count)
    s = maximum(view(progress, slots))
    logical = view(member_ids, slots)

    # complementary basis and truncation
    if s > 0
        arena_reset_phase!(arena)
        a = run_t_arena(arena)
        T = eltype(run.S)
        backend = get_backend(run.S)
        rA, rB, qk = size(run.S, 1), size(run.S, 2), run.qk
        Q = view(ara_ws.Q, :, 1:s, slots)
        proj = workspace_array!(a, backend, T, rB, s, qk, count)
        sketch = workspace_array!(a, backend, T, rA, qk, s, count)
        rC = run.betaU === nothing ? 0 : size(first(run.betaU), 2)
        bt = workspace_array!(a, backend, T, rC, s, count)
        L = workspace_array!(a, backend, T, size(U, 1), s, count)

        apply_corange!(
            L, run, Q, s; beta, compute, proj, sketch, beta_tmp=bt,
            slot0=first(slots))

        Uh = workspace_array!(a, backend, T, size(U, 1), s, count)
        Vh = workspace_array!(a, backend, T, size(V, 1), s, count)
        capture = _capped_capture_sq!(
            ee, run, Q, s, view(progress, slots), size(ara_ws.Q, 2),
            ara_ws.block, Vh, Uh; beta, compute, proj, sketch, beta_tmp=bt,
            slot0=first(slots))
        ara_truncate!(
            Vh, Uh, rr, ee, Q, L; tol, relative=rel,
            maxrank=min(s, size(U, 2)), capture, compute)

        copyto!(view(U, :, 1:s, :), Uh)
        copyto!(view(V, :, 1:s, :), Vh)
    else
        fill!(rr, zero(eltype(rr)))
        fill!(ee, 0.0)
    end

    # canonical factor and diagnostic slots
    qm = grid_size(Cout)[1]
    qn = grid_size(Cout)[2]
    outslots = view(ws_owner.output_slots, 1:count)
    inslots = view(ws_owner.output_slots_inner, 1:count)
    @inbounds for p in 1:count
        outslots[p] = logical[p] + (i - 1) * qn        # tile_linear_index(TileRowMajor, ...)
        inslots[p] = i + (logical[p] - 1) * qm          # tile_linear_index(TileColMajor, ...)
    end
    store_tlr_run!(
        Cout, U, V, rr, ee, outslots, inslots, ws_owner.ranks_global,
        ws_owner.errors_global, ws_owner.indices, ws_owner.indices_host)
end

function rolling_lane_loop!(Cout, run, ara_ws, allmembers::AbstractVector{Int},
                             fixed::Int, ops, arena, ws_owner;
                             beta, eps_rel, r_required, tol, rel, compute)
    # initial active-slot prefix
    cap = ws_owner.capacity
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

    # hot samplers, reused every pass below
    sampler = (Y, width, _) -> apply_run!(Y, run, width, nactive; beta, compute)
    swapper = (p, q) -> swap_run_members!(run, p, q)

    while nactive > 0 || pending <= length(allmembers)
        # cold-start admission
        if nactive == 0
            nnew = min(cap, length(allmembers) - pending + 1)
            slots = 1:nnew
            ids = view(member_ids, slots)
            copyto!(ids, view(allmembers, pending:(pending + nnew - 1)))

            arena_reset_phase!(arena)
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

        # packed ARA pass and retirements
        info = ara_packed_pass!(
            ara_ws, sampler, nactive, member_ids, progress;
            eps_rel, r_required, compute, swap_member! = swapper)
        nactive = info.nactive
        retired = info.retired
        isempty(retired) && continue

        # retired-wave truncation and scatter
        _finalize_wave!(
            Cout, run, ara_ws, retired, member_ids, progress, fixed,
            arena, ws_owner; beta, tol, rel, compute)

        # freed-slot backfill
        nnew = min(cap - nactive, max(length(allmembers) - pending + 1, 0))
        if nnew > 0
            slots = (nactive + 1):(nactive + nnew)
            ids = view(member_ids, slots)
            copyto!(ids, view(allmembers, pending:(pending + nnew - 1)))

            arena_reset_phase!(arena)
            admit_wave!(
                run, ops, ids, slots, fixed, arena;
                C=iszero(beta) ? nothing : Cout, compute)

            ara_reset_slots!(ara_ws, first(slots), nnew)
            fill!(view(progress, slots), 0)
            pending += nnew
            nactive += nnew
        end

        # active sampling views
        if nactive > 0
            ara_ws = rebind_ara_phase(ara_ws, arena)
            rebind_sampling_scratch!(run, arena)
        end
    end

    return Cout
end
