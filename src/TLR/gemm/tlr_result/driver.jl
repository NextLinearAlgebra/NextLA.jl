# gemm! entry point for the canonical TLR-output GEMM: produces a
# compressed TLR C via ARA sampling (single_tile_coupling.jl/run_coupling.jl,
# single_tile_sampling.jl/rolling_schedule.jl). Sampling-side selection, run
# scatter, and the reusable-workspace-driven traversal loop live here.

@inline _active_rank_cap(A::TLRMatrix) =
    isempty(A.ranks) ? 0 : min(maxrank(A), maximum(Int, A.ranks))

@inline function _right_sampling_workspace_elems(qm::Int, qk::Int,
                                                 rA::Int, rB::Int,
                                                 block::Int, rmaxC::Int)
    qk * (block * (rB + qm * rA) +
          rmaxC * qm * (rA + rB) +
          qm * rA * rB)
end

@inline function _left_sampling_workspace_elems(qn::Int, qk::Int,
                                                rA::Int, rB::Int,
                                                block::Int, rmaxC::Int)
    qk * (block * (rA + qn * rB) +
          rmaxC * qn * (rA + rB) +
          qn * rA * rB)
end

"""
    choose_tlr_sampling_side(LA, LB, rmaxC, block, rA, rB) -> Symbol

Choose the repeated ARA apply from the logical layouts. A logical row-major A
permits zero-copy right sampling; a logical column-major B permits zero-copy
left sampling. When both are available, compare the peak retained core
workspace of one fixed-column right run with one fixed-row left run.
"""
function choose_tlr_sampling_side(LA::LogicalTLROperand,
                                  LB::LogicalTLROperand,
                                  rmaxC::Int, block::Int,
                                  rA::Int, rB::Int)
    can_right = tile_order(LA) isa TileRowMajor
    can_left = tile_order(LB) isa TileColMajor
    can_right || can_left || throw(ArgumentError(
        "canonical TLR GEMM does not yet support transA='T', transB='N': " *
        "neither contraction stack is contiguous; run-level packing/reduction " *
        "is deferred to the general-storage API"))
    can_right && !can_left && return :right
    can_left && !can_right && return :left

    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    right = _right_sampling_workspace_elems(
        qm, qk, rA, rB, block, rmaxC)
    left = _left_sampling_workspace_elems(
        qn, qk, rA, rB, block, rmaxC)
    return right < left ? :right : :left
end

@kernel function _store_tlr_run_factor_kernel!(dest, src, slots)
    i, k, p = @index(Global, NTuple)
    @inbounds dest[i, k, Int(slots[p])] = src[i, k, p]
end

@kernel function _store_tlr_run_diagnostic_kernel!(ranks, err_sq,
                                                   ranks_run, err_run, slots)
    p = @index(Global, Linear)
    slot = Int(@inbounds slots[p])
    @inbounds begin
        ranks[slot] = ranks_run[p]
        err_sq[slot] = err_run[p]
    end
end

"""
    _store_tlr_run!(C, U, V, ranks_run, err_run, slots, ranks_dev, err_dev,
                    slots_dev, slots_host)

Scatter one run's factors and diagnostics into `C`'s canonical storage.
`slots_dev` is caller-owned scratch (sized to at least `length(slots)`,
reused across the driver's traversal of `C`'s rows/columns) rather than
allocated here, since every run in that traversal needs an identically-sized
buffer.
"""
function _store_tlr_run!(C::TLRMatrix, U, V, ranks_run, err_run,
                         slots::AbstractVector{Int}, ranks_dev, err_dev,
                         slots_dev, slots_host)
    backend = get_backend(C)
    count = length(slots)
    @inbounds for p in 1:count
        slots_host[p] = Int32(slots[p])
    end
    copyto!(slots_dev, slots_host)
    sd = view(slots_dev, 1:count)
    _store_tlr_run_factor_kernel!(backend)(
        C.int_U, U, sd; ndrange=size(U),
    )
    _store_tlr_run_factor_kernel!(backend)(
        C.int_V, V, sd; ndrange=size(V),
    )
    _store_tlr_run_diagnostic_kernel!(backend)(
        ranks_dev, err_dev, ranks_run, err_run, sd; ndrange=count,
    )
    return nothing
end

function _validate_canonical_tlr_gemm(C::TLRMatrix,
                                      A::TLRMatrix,
                                      B::TLRMatrix,
                                      LA::LogicalTLROperand,
                                      LB::LogicalTLROperand)
    all(tile_order(X) isa TileRowMajor for X in (C, A, B)) ||
        throw(ArgumentError(
            "canonical TLR gemm! requires physical TileRowMajor storage for C, A, and B"))
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("logical contraction tile sizes must agree"))
    nominal_tile_size(C) ==
        (nominal_tile_size(LA, 1), nominal_tile_size(LB, 2)) ||
        throw(DimensionMismatch("C's tile shape must match the logical output tile"))
    any(!iszero, tail_tile_size(X, d) for X in (C, LA, LB) for d in 1:2) &&
        throw(ArgumentError(
            "canonical TLR gemm! currently requires regular-grid tiling"))
    return nothing
end

function _tlr_gemm_workspace_spec(C::TLRMatrix{BackendT,T},
                                  A::TLRMatrix{BackendT,T},
                                  B::TLRMatrix{BackendT,T};
                                  transA::Char='N', transB::Char='N',
                                  block::Int=32) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_canonical_tlr_gemm(C, A, B, LA, LB)
    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    rA, rB = _active_rank_cap(A), _active_rank_cap(B)
    blk = min(block, max(maxrank(C), 1))
    side = choose_tlr_sampling_side(LA, LB, maxrank(C), blk, rA, rB)
    family = side === :right && tile_order(LB) isa TileColMajor ?
        :column : (side === :right ? :row_right : :row_left)
    nmember = family === :column ? qm : qn
    bm = nominal_tile_size(C, 1)
    bn = nominal_tile_size(C, 2)
    Thi = tlr_orthogonalization_type(T)
    arena_bytes = ara_run_workspace_bytes(
        family, rA, rB, qk, nmember, blk, maxrank(C), bm, bn, T, Thi)
    key = (
        backend=typeof(get_backend(C)), T=T, rankT=eltype(C.ranks),
        family=family, qm=qm, qk=qk, qn=qn, nmember=nmember,
        rA=rA, rB=rB, block=blk, maxrank=maxrank(C), bm=bm, bn=bn,
    )
    return (; LA, LB, side, family, nmember, arena_bytes, key, Thi)
end

function _prepare_tlr_gemm_workspace(C, A, B, workspace;
                                     transA::Char, transB::Char, block::Int)
    spec = _tlr_gemm_workspace_spec(C, A, B; transA, transB, block)
    if workspace === nothing
        return TLRGemmWorkspace(C, A, B; transA, transB, block), spec
    elseif workspace isa Integer
        workspace >= 0 ||
            throw(ArgumentError("workspace bytes must be nonnegative"))
        required = tlr_gemm_minimum_workspace_bytes(
            C, A, B; transA, transB, block)
        workspace >= required || throw(ArgumentError(
            "workspace has $workspace bytes; at least $required bytes are required"))
        ws = TLRGemmWorkspace(
            C, A, B; bytes=Int(workspace), transA, transB, block)
        return ws, spec
    elseif workspace isa TLRGemmWorkspace
        workspace.key.operation == spec.key || throw(ArgumentError(
            "TLRGemmWorkspace geometry, backend, or element type does not match this operation"))
        return workspace, spec
    end
    throw(ArgumentError(
        "workspace must be nothing, an integer byte count, or TLRGemmWorkspace"))
end

"""
    gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
          alpha=true, beta=false, transA='N', transB='N',
          tol=0, rel=false, eps_rel=nothing, r_required=10, block=32,
          compute=nothing, workspace=nothing) -> C

Canonical physical-row-major TLR result GEMM:

    C := alpha * op(A) * op(B) + beta * C

`C`, `A`, and `B` must all be physically `TileRowMajor`. `NN` uses right
sampling, `TT` uses left sampling, and `NT` chooses between both zero-copy
terminal stacks from rank-derived intermediate workspace. `TN` is deliberately
unsupported until the general-storage packing/reduction path is implemented.

`tol` controls final Frobenius truncation. `eps_rel` controls adaptive range
capture and defaults to `max(tol, ara_stopping_floor(promoted_type))`.
`workspace` accepts a byte count or a reusable `TLRGemmWorkspace`; omitting
it constructs one temporary workspace for convenience.
"""
function _gemm_tlr!(C::TLRMatrix{BackendT,T},
                    A::TLRMatrix{BackendT,T},
                    B::TLRMatrix{BackendT,T};
               alpha=true, beta=false,
               transA::Char='N', transB::Char='N',
               tol::Real=0.0, rel::Bool=false,
               eps_rel=nothing, r_required::Int=10, block::Int=32,
               compute=nothing, workspace=nothing,
               stats=nothing) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_canonical_tlr_gemm(C, A, B, LA, LB)
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    r_required >= 1 || throw(ArgumentError("r_required must be positive"))
    block >= 1 || throw(ArgumentError("block must be positive"))

    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    validate_tlr_gemm_precision(get_backend(C), T, T, mode)
    ScalarT = gemm_compute_type(mode)
    α, β = ScalarT(alpha), ScalarT(beta)
    floor_rel = ara_stopping_floor(tlr_orthogonalization_type(T))
    sample_tol = eps_rel === nothing ? max(Float64(tol), floor_rel) :
                 Float64(eps_rel)
    sample_tol >= floor_rel || throw(ArgumentError(
        "eps_rel=$sample_tol is below the supported floor $floor_rel"))

    ws_owner, workspace_spec = _prepare_tlr_gemm_workspace(
        C, A, B, workspace; transA, transB, block)
    ops = logical_operands(LA, LB)
    LC = logical_operand(C)
    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    rA, rB = _active_rank_cap(A), _active_rank_cap(B)
    blk = min(block, max(maxrank(C), 1))
    side = workspace_spec.side
    backend = get_backend(C)
    ranks_dev = ws_owner.ranks_global
    err_dev = ws_owner.errors_global
    bm_tile = nominal_tile_size(C, 1)
    bn_tile = nominal_tile_size(C, 2)

    if side === :right && tile_order(LB) isa TileColMajor
        # NT, right choice: fixed columns share H. `arena` and `ws_owner`'s
        # traversal/diagnostic buffers are sized once for this family/shape
        # and reused (reset) across every column below; each iteration
        # constructs a fresh `ColumnRunCoupling`/`ARAWorkspace` from the
        # arena and drives it to convergence via `_rolling_lane_loop!`,
        # which rolls pending members into released slots as active ones
        # retire instead of running the whole column as one fixed batch.
        arena = ws_owner.arena
        cap = ws_owner.key.capacity
        for j in 1:qn
            _arena_reset!(arena)
            initial = 1:min(cap, qm)
            run = ColumnRunCoupling(
                ops, initial, j; alpha=α, beta=β, C=LC,
                block=blk, maxrank=maxrank(C), rA, rB, compute=mode, arena)
            ara_ws = ARAWorkspace(
                T, backend, bm_tile, maxrank(C), cap; block=blk, arena,
                state_storage=ws_owner.ara_state)
            _rolling_lane_loop!(
                C, run, ara_ws, 1:qm, j, ops, arena, ws_owner;
                beta=β, eps_rel=sample_tol, r_required, tol, rel,
                compute=mode, side=:right, stats,
            )
        end
    else
        # NN right sampling, or NT/TT left sampling: fixed output rows.
        arena = ws_owner.arena
        cap = ws_owner.key.capacity
        for i in 1:qm
            _arena_reset!(arena)
            initial = 1:min(cap, qn)
            if side === :right
                run = RowRightRunCoupling(
                    ops, i, initial; alpha=α, beta=β, C=LC,
                    block=blk, maxrank=maxrank(C), rA, rB, compute=mode,
                    arena, index_scratch=ws_owner.indices,
                )
                ara_ws = ARAWorkspace(
                    T, backend, bm_tile, maxrank(C), cap; block=blk, arena,
                    state_storage=ws_owner.ara_state)
            else
                run = RowLeftRunCoupling(
                    ops, i, initial; alpha=α, beta=β, C=LC,
                    block=blk, maxrank=maxrank(C), rA, rB, compute=mode,
                    arena,
                )
                ara_ws = ARAWorkspace(
                    T, backend, bn_tile, maxrank(C), cap; block=blk, arena,
                    state_storage=ws_owner.ara_state)
            end
            _rolling_lane_loop!(
                C, run, ara_ws, 1:qn, i, ops, arena, ws_owner;
                beta=β, eps_rel=sample_tol, r_required, tol, rel,
                compute=mode, side, stats,
            )
        end
    end

    ranks_host = Array(ranks_dev)
    err_host = Array(err_dev)
    copyto!(C.ranks, ranks_host)
    @inbounds for k in eachindex(C.resid)
        C.resid[k] = sqrt(max(err_host[k], 0.0))
    end
    return C
end

function gemm!(C::TLRMatrix{BackendT,T},
               A::TLRMatrix{BackendT,T},
               B::TLRMatrix{BackendT,T};
               alpha=true, beta=false,
               transA::Char='N', transB::Char='N',
               tol::Real=0.0, rel::Bool=false,
               eps_rel=nothing, r_required::Int=10, block::Int=32,
               compute=nothing, workspace=nothing) where {BackendT,T}
    return _gemm_tlr!(
        C, A, B; alpha, beta, transA, transB, tol, rel, eps_rel,
        r_required, block, compute, workspace)
end

"""
    _tlr_gemm_schedule_stats!(C, A, B; kwargs...) -> TLRGemmScheduleStats

Internal profiling entry point for the R4a scheduler. It intentionally keeps
instrumentation out of the public `gemm!` keyword contract.
"""
function _tlr_gemm_schedule_stats!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
                                   kwargs...)
    stats = TLRGemmScheduleStats()
    _gemm_tlr!(C, A, B; kwargs..., stats)
    return stats
end
