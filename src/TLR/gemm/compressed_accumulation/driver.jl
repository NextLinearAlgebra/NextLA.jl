# Allocation-returning compressed-output GEMM via ARA sampling
# (run_coupling.jl/rolling_schedule.jl).
# Sampling-side selection, run
# scatter, and the reusable-workspace-driven traversal loop live here.

@inline _active_rank_cap(A::CompressedFTLRMatrix) =
    isempty(A.ranks) ? 0 : min(maxrank(A), maximum(Int, A.ranks))

"""
    _require_complementary_packing(X, name)

The compressed-output zero-copy paths need `X`'s *logical* outer order to be
`TileRowMajor` and inner order `TileColMajor` — the default complementary
packing every `CompressedFTLRMatrix` here is constructed with. Under that
packing this holds for both `'N'` and `'T'` (transposing the tile order
with itself returns the original order), which is what lets one code path
serve all four transpose combinations instead of gating on which side is
zero-copy for a given transpose flag.
"""
@inline function _require_complementary_packing(X, name::String)
    compressed_ftlr_outer_order(X) isa TileRowMajor &&
        compressed_ftlr_inner_order(X) isa TileColMajor || throw(ArgumentError(
        "canonical TLR gemm! requires complementary packing (outer row-major, " *
        "inner col-major) for $name"))
    return nothing
end

"""
    choose_tlr_sampling_side(LA, LB, rmaxC, block, rA, rB) -> Symbol

Choose the repeated ARA apply from the logical layouts. Under the
complementary packing `_require_complementary_packing` requires, both a
zero-copy right sampling stack (from `A`) and a zero-copy left sampling stack
(from `B`) are always available, for any transpose combination — so this is a
pure cost comparison: the peak retained core workspace of one fixed-column
right run vs. one fixed-row left run.
"""
function choose_tlr_sampling_side(LA::AbstractTLRMatrix,
                                  LB::AbstractTLRMatrix,
                                  rmaxC::Int, block::Int,
                                  rA::Int, rB::Int)
    _require_complementary_packing(LA, "A")
    _require_complementary_packing(LB, "B")

    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    right = qk * (block * (rB + qm * rA) +
                    rmaxC * qm * (rA + rB) + qm * rA * rB)
    left = qk * (block * (rA + qn * rB) +
                   rmaxC * qn * (rA + rB) + qn * rA * rB)
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
    _store_tlr_run!(C, U, V, ranks_run, err_run, outer_slots, inner_slots,
                    ranks_dev, err_dev, slots_dev, slots_host)

Scatter one run's factors and diagnostics into `C`'s canonical storage.
`C.outer` (row-major) and `C.inner` (col-major) linearize the same logical
`(i,j)` tile to *different* scalar slots, so two slot maps are needed —
`outer_slots` also drives `ranks_dev`/`err_dev`, since `C`'s own diagnostic
rank-vector order matches `C.outer`'s (both `TileRowMajor` by construction).
`slots_dev` is caller-owned scratch (sized to at least `length(outer_slots)`,
reused across the driver's traversal of `C`'s rows/columns) rather than
allocated here, since every run in that traversal needs an identically-sized
buffer.
"""
function _store_tlr_run!(C::CompressedFTLRMatrix, U, V, ranks_run, err_run,
                         outer_slots::AbstractVector{Int},
                         inner_slots::AbstractVector{Int},
                         ranks_dev, err_dev, slots_dev, slots_host)
    backend = get_backend(C)
    count = length(outer_slots)
    @inbounds for p in 1:count
        slots_host[p] = Int32(outer_slots[p])
    end
    copyto!(slots_dev, slots_host)
    sd_outer = view(slots_dev, 1:count)
    _store_tlr_run_factor_kernel!(backend)(
        _compressed_ftlr_uniform_view(C.outer), U, sd_outer; ndrange=size(U),
    )
    _store_tlr_run_diagnostic_kernel!(backend)(
        ranks_dev, err_dev, ranks_run, err_run, sd_outer; ndrange=count,
    )
    @inbounds for p in 1:count
        slots_host[p] = Int32(inner_slots[p])
    end
    copyto!(slots_dev, slots_host)
    sd_inner = view(slots_dev, 1:count)
    _store_tlr_run_factor_kernel!(backend)(
        _compressed_ftlr_uniform_view(C.inner), V, sd_inner; ndrange=size(V),
    )
    return nothing
end

function _validate_canonical_tlr_gemm(C::CompressedFTLRMatrix,
                                      LA::AbstractTLRMatrix,
                                      LB::AbstractTLRMatrix)
    _require_complementary_packing(C, "C")
    _require_complementary_packing(LA, "A")
    _require_complementary_packing(LB, "B")
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

function _tlr_gemm_workspace_spec(C::CompressedFTLRMatrix{BackendT,T},
                                  A::CompressedFTLRMatrix{BackendT,T},
                                  B::CompressedFTLRMatrix{BackendT,T};
                                  transA::Char='N', transB::Char='N',
                                  block::Int=32) where {BackendT,T}
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    _validate_canonical_tlr_gemm(C, LA, LB)
    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    rA, rB = _active_rank_cap(A), _active_rank_cap(B)
    cap = maximum_storage_rank(C)
    blk = min(block, max(cap, 1))
    side = choose_tlr_sampling_side(LA, LB, cap, blk, rA, rB)
    # Complementary packing makes compressed_ftlr_inner_order(LB) isa
    # TileColMajor unconditionally true, so the fixed-row "family=:row_right"
    # variant (right-sampling with the row fixed, rather than the column) is
    # never selected -- it's superseded by :column, which is strictly
    # available whenever :right is. Only two families remain reachable.
    family = side === :right ? :column : :row_left
    nmember = family === :column ? qm : qn
    bm = nominal_tile_size(C, 1)
    bn = nominal_tile_size(C, 2)
    return (
        backend=typeof(get_backend(C)), T=T, rankT=eltype(C.ranks),
        family=family, qm=qm, qk=qk, qn=qn, nmember=nmember,
        rA=rA, rB=rB, block=blk, maxrank=cap, bm=bm, bn=bn,
        Thi=tlr_orthogonalization_type(T),
    )
end

function _prepare_tlr_gemm_workspace(C, A, B, workspace;
                                     transA::Char, transB::Char, block::Int)
    spec = _tlr_gemm_workspace_spec(C, A, B; transA, transB, block)
    if workspace === nothing
        return CompressedGemmWorkspace(C, spec), spec
    elseif workspace isa Int
        workspace >= 0 ||
            throw(ArgumentError("workspace bytes must be nonnegative"))
        required = _tlr_gemm_workspace_bytes(spec, 1)
        workspace >= required || throw(ArgumentError(
            "workspace has $workspace bytes; at least $required bytes are required"))
        ws = CompressedGemmWorkspace(C, spec; bytes=workspace)
        return ws, spec
    elseif workspace isa CompressedGemmWorkspace
        workspace.operation == spec || throw(ArgumentError(
            "CompressedGemmWorkspace geometry, backend, or element type does not match this operation"))
        return workspace, spec
    end
    throw(ArgumentError(
        "workspace must be nothing, an integer byte count, or CompressedGemmWorkspace"))
end

"""Internal fixed-width ARA driver used by allocation-returning [`gemm`](@ref)."""
function _gemm_tlr!(C::CompressedFTLRMatrix{BackendT,T},
                    A::CompressedFTLRMatrix{BackendT,T},
                    B::CompressedFTLRMatrix{BackendT,T};
               alpha=true, beta=false,
               transA::Char='N', transB::Char='N',
               tol::Real=0.0, rel::Bool=false,
               eps_rel=nothing, r_required::Int=10, block::Int=32,
               compute=nothing, workspace=nothing) where {BackendT,T}
    # argument checks
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    _validate_canonical_tlr_gemm(C, LA, LB)
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    r_required >= 1 || throw(ArgumentError("r_required must be positive"))
    block >= 1 || throw(ArgumentError("block must be positive"))

    # compute policy and sampling tolerance
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

    # workspace and the four zero-copy factor-panel views the runs sample from
    ws_owner, workspace_spec = _prepare_tlr_gemm_workspace(
        C, A, B, workspace; transA, transB, block)
    qmA, qnA = regular_grid_size(LA)
    qmB, qnB = regular_grid_size(LB)
    ops = (
        av=(data=_compressed_ftlr_uniform_view(_compressed_ftlr_inner_storage(LA)),
            order=compressed_ftlr_inner_order(LA), qm=qmA, qn=qnA),
        bu=(data=_compressed_ftlr_uniform_view(_compressed_ftlr_outer_storage(LB)),
            order=compressed_ftlr_outer_order(LB), qm=qmB, qn=qnB),
        bv=(data=_compressed_ftlr_uniform_view(_compressed_ftlr_inner_storage(LB)),
            order=compressed_ftlr_inner_order(LB), qm=qmB, qn=qnB),
        au=(data=_compressed_ftlr_uniform_view(_compressed_ftlr_outer_storage(LA)),
            order=compressed_ftlr_outer_order(LA), qm=qmA, qn=qnA),
    )

    # traversal shape shared by both families
    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    rA, rB = _active_rank_cap(A), _active_rank_cap(B)
    cap_C = maximum_storage_rank(C)
    blk = min(block, max(cap_C, 1))
    backend = get_backend(C)
    ranks_dev = ws_owner.ranks_global
    err_dev = ws_owner.errors_global
    bm_tile = nominal_tile_size(C, 1)
    bn_tile = nominal_tile_size(C, 2)

    if workspace_spec.family === :column
        # NT, right choice: fixed columns share H. `arena` and `ws_owner`'s
        # traversal/diagnostic buffers are sized once for this family/shape
        # and reused (reset) across every column below; each iteration
        # constructs a fresh `RunCoupling{:column}`/`ARAWorkspace` from the
        # arena and drives it to convergence via `_rolling_lane_loop!`,
        # which rolls pending members into released slots as active ones
        # retire instead of running the whole column as one fixed batch.
        arena = ws_owner.arena
        cap = ws_owner.capacity
        for j in 1:qn
            _arena_reset!(arena)
            initial = 1:min(cap, qm)
            run = RunCoupling(
                Val(:column), ops, initial, j; alpha=α, beta=β, C,
                block=blk, maxrank=cap_C, rA, rB, compute=mode, arena)
            ara_ws = ARAWorkspace(
                T, backend, bm_tile, cap_C, cap; block=blk, arena,
                state_storage=ws_owner.ara_state)
            _rolling_lane_loop!(
                C, run, ara_ws, 1:qm, j, ops, arena, ws_owner;
                beta=β, eps_rel=sample_tol, r_required, tol, rel,
                compute=mode,
            )
        end
    else
        # side === :left: fixed output rows, left sampling.
        arena = ws_owner.arena
        cap = ws_owner.capacity
        for i in 1:qm
            _arena_reset!(arena)
            initial = 1:min(cap, qn)
            run = RunCoupling(
                Val(:row), ops, i, initial; alpha=α, beta=β, C,
                block=blk, maxrank=cap_C, rA, rB, compute=mode,
                arena,
            )
            ara_ws = ARAWorkspace(
                T, backend, bn_tile, cap_C, cap; block=blk, arena,
                state_storage=ws_owner.ara_state)
            _rolling_lane_loop!(
                C, run, ara_ws, 1:qn, i, ops, arena, ws_owner;
                beta=β, eps_rel=sample_tol, r_required, tol, rel,
                compute=mode,
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

function _uniform_ara_operand(A::CompressedFTLRMatrix{BackendT,T}) where {BackendT,T}
    qm, qn = grid_size(A)
    cap = maxrank(A)
    uniform = CompressedFTLRMatrix(
        get_backend(A), T, size(A)..., nominal_tile_size(A), fill(cap, qm, qn);
        outer_order=TileRowMajor, inner_order=TileColMajor)
    @inbounds for j in 1:qn, i in 1:qm
        r = _compressed_ftlr_rank(A, i, j)
        r == 0 && continue
        U, V = get_factors(A, i, j)
        Ud = compressed_ftlr_storage_outer(uniform, i, j)
        Vd = compressed_ftlr_storage_inner(uniform, i, j)
        copyto!(view(Ud, :, 1:r), U)
        copyto!(view(Vd, :, 1:r), V)
    end
    return uniform
end

function _pack_ara_output(staging::CompressedFTLRMatrix;
                          rank_multiple::Int=0)
    qm, qn = grid_size(staging)
    rank_grid = [ranks(staging)[
                     tile_linear_index(staging.outer.order, qm, qn, i, j)]
                 for i in 1:qm, j in 1:qn]
    C = CompressedFTLRMatrix(
        get_backend(staging), eltype(staging), size(staging)...,
        nominal_tile_size(staging), rank_grid;
        outer_order=TileRowMajor, inner_order=TileColMajor,
        rank_multiple)
    @inbounds for j in 1:qn, i in 1:qm
        slot = tile_linear_index(C.outer.order, qm, qn, i, j)
        C.resid[slot] = residuals(staging)[
            tile_linear_index(staging.outer.order, qm, qn, i, j)]
        r = rank_grid[i, j]
        r == 0 && continue
        # `staging` retains one fixed physical width even after its diagnostic
        # rank vector is overwritten with the discovered ranks.
        Us = _compressed_ftlr_factor_view(staging.outer, i, j, i, r)
        Vs = _compressed_ftlr_factor_view(staging.inner, i, j, j, r)
        copyto!(compressed_ftlr_outer(C, i, j), Us)
        copyto!(compressed_ftlr_inner(C, i, j), Vs)
    end
    return C
end

"""
    gemm(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
         maxrank, rank_multiple=0, ...)

Return a newly allocated compressed approximation to
`alpha * op(A) * op(B)`. Output ranks are discovered by ARA, so this operation
cannot be an in-place `gemm!` on finalized packed storage. Fixed-width factors
exist only as private numerical staging and are compacted before return.

The compressed-output algorithm currently requires regular-grid operands.
Containers, dense compression, and dense-output `gemm!` continue to support a
ragged final tile row or column. Tensor-core compute modes additionally require
input stored widths, `maxrank`, and the returned `rank_multiple` to use their
backend rank quantum; incompatible calls throw before scheduling.
"""
function gemm(A::CompressedFTLRMatrix{BackendT,T},
              B::CompressedFTLRMatrix{BackendT,T};
              maxrank::Int,
              rank_multiple::Int=0,
              alpha=true,
              transA::Char='N', transB::Char='N',
              tol::Real=0.0, rel::Bool=false,
              eps_rel=nothing, r_required::Int=10, block::Int=32,
              compute=nothing, workspace=nothing) where {BackendT,T}
    maxrank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))
    rank_multiple >= 0 || throw(ArgumentError("rank_multiple must be nonnegative"))
    workspace isa CompressedGemmWorkspace && throw(ArgumentError(
        "allocation-returning gemm accepts workspace=nothing or a byte count; " *
        "a CompressedGemmWorkspace is bound to private output staging"))
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("logical contraction tile sizes must agree"))
    any(!iszero, tail_tile_size(X, d) for X in (LA, LB) for d in 1:2) &&
        throw(ArgumentError(
            "compressed-output gemm currently requires regular-grid tiling"))
    typeof(get_backend(A)) === typeof(get_backend(B)) || throw(ArgumentError(
        "compressed operands must use the same backend"))

    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    validate_tlr_gemm_storage(LA, mode; name="left operand")
    validate_tlr_gemm_storage(LB, mode; name="right operand")
    required_multiple = required_tlr_gemm_rank_multiple(get_backend(A), T, mode)
    if required_multiple > 1
        iszero(maxrank % required_multiple) || throw(ArgumentError(
            "maxrank=$maxrank is incompatible with this GEMM precision; " *
            "use a multiple of $required_multiple"))
        rank_multiple > 0 && iszero(rank_multiple % required_multiple) ||
            throw(ArgumentError(
                "compressed-output GEMM with this precision requires " *
                "rank_multiple=$required_multiple (or a multiple of it)"))
    end

    out_tile = (nominal_tile_size(LA, 1), nominal_tile_size(LB, 2))
    maxrank <= min(out_tile...) || throw(ArgumentError(
        "maxrank=$maxrank exceeds output tile extent $(min(out_tile...))"))
    qm, qn = grid_size(LA)[1], grid_size(LB)[2]
    if maxrank == 0
        return CompressedFTLRMatrix(
            get_backend(A), T, size(LA, 1), size(LB, 2), out_tile,
            Base.zeros(Int, qm, qn); rank_multiple,
            outer_order=TileRowMajor, inner_order=TileColMajor)
    end

    UA = _uniform_ara_operand(A)
    UB = _uniform_ara_operand(B)
    staging = CompressedFTLRMatrix(
        get_backend(A), T, size(LA, 1), size(LB, 2), out_tile,
        fill(maxrank, qm, qn);
        outer_order=TileRowMajor, inner_order=TileColMajor)
    _gemm_tlr!(
        staging, UA, UB; alpha, beta=false, transA, transB, tol, rel, eps_rel,
        r_required, block, compute=mode, workspace)
    return _pack_ara_output(staging; rank_multiple)
end
