# Allocation-returning compressed-output GEMM: sampling-side selection, run
# scattering, and reusable-workspace traversal.

@inline _active_rank_cap(A::CompressedFTLRMatrix) =
    isempty(A.ranks) ? 0 : min(maxrank(A), maximum(Int, A.ranks))

"""
    require_complementary_packing(X, name)

Require logical outer `TileRowMajor` and inner `TileColMajor` packing.
Transposition preserves this complementary logical layout, enabling the same
zero-copy path for all transpose combinations.
"""
@inline function require_complementary_packing(X, name::String)
    compressed_ftlr_outer_order(X) isa TileRowMajor &&
        compressed_ftlr_inner_order(X) isa TileColMajor || throw(ArgumentError(
        "canonical TLR gemm! requires complementary packing (outer row-major, " *
        "inner col-major) for $name"))
    return nothing
end

"""
    choose_tlr_sampling_side(LA, LB, rmaxC, block, rA, rB) -> Symbol

Choose the lower-workspace ARA sampling side. Complementary packing makes both
zero-copy sampling stacks available for every transpose combination, so the
decision compares the retained core workspace of fixed-column and fixed-row
runs.
"""
function choose_tlr_sampling_side(LA::AbstractTLRMatrix,
                                  LB::AbstractTLRMatrix,
                                  rmaxC::Int, block::Int,
                                  rA::Int, rB::Int)
    require_complementary_packing(LA, "A")
    require_complementary_packing(LB, "B")

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
    store_tlr_run!(C, U, V, ranks_run, err_run, outer_slots, inner_slots,
                    ranks_dev, err_dev, slots_dev, slots_host)

Scatter one run's factors and diagnostics into `C`. Outer and inner factors
need separate slot maps; diagnostics use outer-factor order. `slots_dev` and
`slots_host` are caller-owned traversal scratch.
"""
function store_tlr_run!(C::CompressedFTLRMatrix, U, V, ranks_run, err_run,
                         outer_slots::AbstractVector{Int},
                         inner_slots::AbstractVector{Int},
                         ranks_dev, err_dev, slots_dev, slots_host)
    backend = get_backend(C)
    count = length(outer_slots)

    # outer factors and diagnostics
    @inbounds for p in 1:count
        slots_host[p] = Int32(outer_slots[p])
    end
    copyto!(slots_dev, slots_host)
    sd_outer = view(slots_dev, 1:count)
    _store_tlr_run_factor_kernel!(backend)(
        compressed_ftlr_uniform_view(C.outer), U, sd_outer; ndrange=size(U),
    )
    _store_tlr_run_diagnostic_kernel!(backend)(
        ranks_dev, err_dev, ranks_run, err_run, sd_outer; ndrange=count,
    )

    # inner factors
    @inbounds for p in 1:count
        slots_host[p] = Int32(inner_slots[p])
    end
    copyto!(slots_dev, slots_host)
    sd_inner = view(slots_dev, 1:count)
    _store_tlr_run_factor_kernel!(backend)(
        compressed_ftlr_uniform_view(C.inner), V, sd_inner; ndrange=size(V),
    )
    return nothing
end

function _validate_canonical_tlr_gemm(C::CompressedFTLRMatrix,
                                      LA::AbstractTLRMatrix,
                                      LB::AbstractTLRMatrix)
    require_complementary_packing(C, "C")
    require_complementary_packing(LA, "A")
    require_complementary_packing(LB, "B")
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

function tlr_gemm_workspace_spec(C::CompressedFTLRMatrix{BackendT,T},
                                  A::CompressedFTLRMatrix{BackendT,T},
                                  B::CompressedFTLRMatrix{BackendT,T};
                                  transA::Char='N', transB::Char='N',
                                  block::Int=32) where {BackendT,T}
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    _validate_canonical_tlr_gemm(C, LA, LB)

    # sampling cost
    qm, qk = grid_size(LA)
    _, qn = grid_size(LB)
    rA, rB = _active_rank_cap(A), _active_rank_cap(B)
    cap = maximum_storage_rank(C)
    blk = min(block, max(cap, 1))
    side = choose_tlr_sampling_side(LA, LB, cap, blk, rA, rB)

    # reachable run family
    # Complementary packing makes `:row_right` redundant with `:column`.
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
    spec = tlr_gemm_workspace_spec(C, A, B; transA, transB, block)

    if workspace === nothing
        return CompressedGemmWorkspace(C, spec), spec
    elseif workspace isa Int
        workspace >= 0 ||
            throw(ArgumentError("workspace bytes must be nonnegative"))
        required = tlr_gemm_workspace_bytes(spec, 1)
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

    # workspace and zero-copy factor panels
    ws_owner, workspace_spec = _prepare_tlr_gemm_workspace(
        C, A, B, workspace; transA, transB, block)
    qmA, qnA = regular_grid_size(LA)
    qmB, qnB = regular_grid_size(LB)
    ops = (
        av=(data=compressed_ftlr_uniform_view(compressed_ftlr_inner_storage(LA)),
            order=compressed_ftlr_inner_order(LA), qm=qmA, qn=qnA),
        bu=(data=compressed_ftlr_uniform_view(compressed_ftlr_outer_storage(LB)),
            order=compressed_ftlr_outer_order(LB), qm=qmB, qn=qnB),
        bv=(data=compressed_ftlr_uniform_view(compressed_ftlr_inner_storage(LB)),
            order=compressed_ftlr_inner_order(LB), qm=qmB, qn=qnB),
        au=(data=compressed_ftlr_uniform_view(compressed_ftlr_outer_storage(LA)),
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
        # fixed-column rolling lanes
        # Each column reuses its arena and admits pending members as slots retire.
        arena = ws_owner.arena
        cap = ws_owner.capacity
        for j in 1:qn
            arena_reset!(arena)
            initial = 1:min(cap, qm)
            run = RunCoupling(
                Val(:column), ops, initial, j; alpha=α, beta=β, C,
                block=blk, maxrank=cap_C, rA, rB, compute=mode, arena)
            ara_ws = ARAWorkspace(
                T, backend, bm_tile, cap_C, cap; block=blk, arena,
                state_storage=ws_owner.ara_state)
            rolling_lane_loop!(
                C, run, ara_ws, 1:qm, j, ops, arena, ws_owner;
                beta=β, eps_rel=sample_tol, r_required, tol, rel,
                compute=mode,
            )
        end
    else
        # fixed-row rolling lanes
        arena = ws_owner.arena
        cap = ws_owner.capacity
        for i in 1:qm
            arena_reset!(arena)
            initial = 1:min(cap, qn)
            run = RunCoupling(
                Val(:row), ops, i, initial; alpha=α, beta=β, C,
                block=blk, maxrank=cap_C, rA, rB, compute=mode,
                arena,
            )
            ara_ws = ARAWorkspace(
                T, backend, bn_tile, cap_C, cap; block=blk, arena,
                state_storage=ws_owner.ara_state)
            rolling_lane_loop!(
                C, run, ara_ws, 1:qn, i, ops, arena, ws_owner;
                beta=β, eps_rel=sample_tol, r_required, tol, rel,
                compute=mode,
            )
        end
    end

    # host diagnostics
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

    # logical factors in uniform capacities
    @inbounds for j in 1:qn, i in 1:qm
        r = compressed_ftlr_rank(A, i, j)
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
    # discovered ranks and packed destination
    qm, qn = grid_size(staging)
    rank_grid = [ranks(staging)[
                     tile_linear_index(staging.outer.order, qm, qn, i, j)]
                 for i in 1:qm, j in 1:qn]
    C = CompressedFTLRMatrix(
        get_backend(staging), eltype(staging), size(staging)...,
        nominal_tile_size(staging), rank_grid;
        outer_order=TileRowMajor, inner_order=TileColMajor,
        rank_multiple)

    # factors and residuals
    @inbounds for j in 1:qn, i in 1:qm
        slot = tile_linear_index(C.outer.order, qm, qn, i, j)
        C.resid[slot] = residuals(staging)[
            tile_linear_index(staging.outer.order, qm, qn, i, j)]
        r = rank_grid[i, j]
        r == 0 && continue
        # `staging` retains one fixed physical width even after its diagnostic
        # rank vector is overwritten with the discovered ranks.
        Us = compressed_ftlr_factor_view(staging.outer, i, j, i, r)
        Vs = compressed_ftlr_factor_view(staging.inner, i, j, j, r)
        copyto!(compressed_ftlr_outer(C, i, j), Us)
        copyto!(compressed_ftlr_inner(C, i, j), Vs)
    end
    return C
end

"""
    gemm(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
         maxrank, rank_multiple=0, r_required=10, ...)

Return a compressed approximation to `alpha * op(A) * op(B)`. ARA discovers
ranks in private fixed-width staging, which is compacted before return.
Operands must use regular-grid tiling. Tensor-core modes require stored widths,
`maxrank`, and `rank_multiple` to satisfy the backend rank quantum.
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
    # argument and geometry checks
    maxrank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))
    r_required >= 1 || throw(ArgumentError("r_required must be positive"))
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

    # compute and rank alignment
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

    # output geometry
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

    # uniform staging and compression
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
