# imports
include("precision.jl")
include("lowering/strategy.jl")
include("operands.jl")
include("ara/tile_apply.jl")
include("ara/range_find.jl")
include("workspace.jl")
include("lowering/schedule.jl")
include("lowering/stages.jl")
include("direct.jl")
include("regions/interior.jl")
include("regions/corner.jl")
include("regions/right.jl")
include("regions/bottom.jl")
include("tlr_dense.jl")

@inline function _validate_logical_gemm(C, LA::LogicalTLROperand, LB::LogicalTLROperand)
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match: size(op(A),2) == size(op(B),1)"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("op(A)'s column tile size must equal op(B)'s row tile size (contraction tiling)"))
    return nothing
end

"""
    gemm!(C, A, B; workspace,
          alpha=true, beta=false,
          transA='N', transB='N', compute=nothing) -> C

Compute `C := alpha·(op(A)·op(B)) + beta·C` for dense-diagonal TLR matrices `A`,
`B` into the dense column-major matrix `C`. `transA` and `transB` accept
case-insensitive `N/T`. Transposed operands currently require square matrices with
equal square tiling.

`workspace` is either a global byte count or a reusable `DenseGemmWorkspace`.
It is split between the concurrent interior and serialized-boundary streams
using `InteriorFirstWorkspace`.
`compute` selects the accumulation mode; when omitted it defaults to `Float32` for
`Float16` operands and otherwise to the operand type. `alpha` and `beta` are
converted to that compute type.
"""
function gemm!(C::AbstractMatrix, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char=('N'), transB::Char=('N'), compute=nothing,
    workspace_policy=InteriorFirstWorkspace()) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_logical_gemm(C, LA, LB)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    backend = get_backend(A)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    if _istrans(LA) || _istrans(LB)
        dense_diag_square = size(A, 1) == size(A, 2) && size(B, 1) == size(B, 2)
        square_tiles = nominal_tile_size(A, 1) == nominal_tile_size(A, 2) &&
                       nominal_tile_size(B, 1) == nominal_tile_size(B, 2)
        (dense_diag_square && square_tiles && nominal_tile_size(A) == nominal_tile_size(B)) ||
            throw(ArgumentError("transposed dense-diagonal TLR GEMM currently requires square operands with equal square tiling"))
    else
        nominal_tile_size(A) == nominal_tile_size(B) ||
            throw(DimensionMismatch("A and B must share the same nominal tile size"))
    end

    ScalarT = gemm_compute_type(mode)
    α = ScalarT(alpha)
    β = ScalarT(beta)
    one_β = one(ScalarT)
    ws, interior_arena, auxiliary_arena, split =
        _prepare_dense_gemm_workspace(
            A, B, workspace, workspace_policy; transA, transB)
    WI, WA = split.interior, split.auxiliary

    interior = () -> begin                                        # C_int
        tlr_gemm_int_by_int(C, LA, LB, α, β; budget=WI, compute=mode,
                            arena=interior_arena)
        tlr_gemm_rpanel_by_bpanel(C, LA, LB, α; beta=one_β, budget=WI,
                                  compute=mode, arena=interior_arena)
    end
    boundaries = () -> begin
        tlr_gemm_int_by_rpanel(C, LA, LB, α; beta=β, budget=WA,
                               compute=mode, arena=auxiliary_arena)
        tlr_gemm_rpanel_by_corner(C, LA, LB, α; beta=one_β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
        tlr_gemm_bpanel_by_int(C, LA, LB, α; beta=β, budget=WA,
                               compute=mode, arena=auxiliary_arena)
        tlr_gemm_corner_by_bpanel(C, LA, LB, α; beta=one_β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
        tlr_gemm_corner_by_corner(C, LA, LB, α; beta=β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
        tlr_gemm_bpanel_by_rpanel(C, LA, LB, α; beta=one_β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
    end

    if backend isa KernelAbstractions.CPU
        interior();
        boundaries()
    else
        with_stream(interior, backend, ws.streams[1])
        with_stream(boundaries, backend, ws.streams[2])
        for s in ws.streams
            sync_stream(backend, s)
        end
    end
    return C
end

"""
    gemm!(C, A::TLRMatrix, B::TLRMatrix; workspace,
          alpha=true, beta=false) -> C

Fully low-rank TLR × TLR → dense `C := alpha·(A·B) + beta·C`. Every tile is low-rank,
so the product is the four-region block product with each region a low-rank staged
term — no dense-diagonal / dense-corner special cases.

Like the dense-diagonal path, each region's first writer folds β and the second
accumulates (β = 1); the interior's first writer is `O_A O_B`, which folds β through
the same layout-aware mechanism (`_offdiag_offdiag_gemm!`). Any rectangular grid and
boundary tiling is supported. When either operand has rank 0 the product is
identically zero, so `C` is just scaled by β.

Operand storage is inferred from `A` and `B`, output storage from `C`, and `compute`
selects the accumulation mode. Intermediate factors always retain operand storage;
`alpha` and `beta` use compute precision.
"""
function gemm!(C::AbstractMatrix, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char=('N'), transB::Char=('N'), compute=nothing,
    workspace_policy=InteriorFirstWorkspace()) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_logical_gemm(C, LA, LB)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    backend = get_backend(A)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)

    ScalarT = gemm_compute_type(mode)
    α = ScalarT(alpha)
    β = ScalarT(beta)
    one_β = one(ScalarT)
    ws, interior_arena, auxiliary_arena, split =
        _prepare_dense_gemm_workspace(
            A, B, workspace, workspace_policy; transA, transB)
    WI, WA = split.interior, split.auxiliary

    if maxrank(A) == 0 || maxrank(B) == 0
        _scale_output!(C, β)
        return C
    end

    interior = () -> begin                                       # C_int
        _offdiag_offdiag_gemm!(C, LA, LB; alpha=α, beta=β, budget=WI,
                               compute=mode, arena=interior_arena)
        tlr_gemm_rpanel_by_bpanel(C, LA, LB, α; beta=one_β, budget=WI,
                                  compute=mode, arena=interior_arena)
    end
    boundaries = () -> begin
        tlr_gemm_int_by_rpanel(C, LA, LB, α; beta=β, budget=WA,
                               compute=mode, arena=auxiliary_arena)
        tlr_gemm_rpanel_by_corner(C, LA, LB, α; beta=one_β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
        tlr_gemm_bpanel_by_int(C, LA, LB, α; beta=β, budget=WA,
                               compute=mode, arena=auxiliary_arena)
        tlr_gemm_corner_by_bpanel(C, LA, LB, α; beta=one_β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
        tlr_gemm_corner_by_corner(C, LA, LB, α; beta=β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
        tlr_gemm_bpanel_by_rpanel(C, LA, LB, α; beta=one_β, budget=WA,
                                  compute=mode, arena=auxiliary_arena)
    end

    if backend isa KernelAbstractions.CPU
        interior();
        boundaries()
    else
        with_stream(interior, backend, ws.streams[1])
        with_stream(boundaries, backend, ws.streams[2])
        for s in ws.streams
            sync_stream(backend, s)
        end
    end
    return C
end

@inline function _validate_dense_backend(C, tlr, dense)
    backend = get_backend(tlr)
    typeof(get_backend(dense)) === typeof(backend) &&
        typeof(get_backend(C)) === typeof(backend) ||
        throw(ArgumentError("TLR, dense operand, and output must use the same backend"))
    return backend
end

"""
    gemm!(C, A::TLRMatrix, B::AbstractMatrix; ...)

Compute `C := alpha·op(A)·op(B) + beta·C` with a fully low-rank left operand
and a standalone dense right operand. Intermediates retain the operand storage type.
"""
function gemm!(C::AbstractMatrix, A::TLRMatrix{BackendT,T}, B::AbstractMatrix{T};
    workspace, alpha=true, beta=false,
    transA::Char='N', transB::Char='N', compute=nothing) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_dense_operand(B, transB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    backend = _validate_dense_backend(C, A, B)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    ScalarT = gemm_compute_type(mode)
    _, arena, budget = _prepare_single_gemm_workspace(A, workspace)
    return _tlr_dense_gemm!(C, LA, LB, ScalarT(alpha), ScalarT(beta),
                            budget, mode, arena)
end

"""
    gemm!(C, A::AbstractMatrix, B::TLRMatrix; ...)

Compute `C := alpha·op(A)·op(B) + beta·C` with a standalone dense left operand
and a fully low-rank right operand. Intermediates retain the operand storage type.
"""
function gemm!(C::AbstractMatrix, A::AbstractMatrix{T}, B::TLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char='N', transB::Char='N', compute=nothing) where {BackendT,T}
    LA = logical_dense_operand(A, transA)
    LB = logical_operand(B, transB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    backend = _validate_dense_backend(C, B, A)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    ScalarT = gemm_compute_type(mode)
    _, arena, budget = _prepare_single_gemm_workspace(B, workspace)
    return _dense_tlr_gemm!(C, LA, LB, ScalarT(alpha), ScalarT(beta),
                            budget, mode, arena)
end

# ─── Canonical TLR × TLR → TLR (R3) ──────────────────────────────────────────

@inline _active_rank_cap(A::TLRMatrix) =
    isempty(A.ranks) ? 0 : min(maxrank(A), maximum(Int, A.ranks))

@inline function _right_sampling_workspace_elems(qm::Int, qk::Int,
                                                 rA::Int, rB::Int,
                                                 block::Int, maxrankC::Int)
    qk * (block * (rB + qm * rA) +
          maxrankC * qm * (rA + rB) +
          qm * rA * rB)
end

@inline function _left_sampling_workspace_elems(qn::Int, qk::Int,
                                                rA::Int, rB::Int,
                                                block::Int, maxrankC::Int)
    qk * (block * (rA + qn * rB) +
          maxrankC * qn * (rA + rB) +
          qn * rA * rB)
end

"""
    choose_tlr_sampling_side(LA, LB, maxrankC, block, rA, rB) -> Symbol

Choose the repeated ARA apply from the logical layouts. A logical row-major A
permits zero-copy right sampling; a logical column-major B permits zero-copy
left sampling. When both are available, compare the peak retained core
workspace of one fixed-column right run with one fixed-row left run.
"""
function choose_tlr_sampling_side(LA::LogicalTLROperand,
                                  LB::LogicalTLROperand,
                                  maxrankC::Int, block::Int,
                                  rA::Int, rB::Int)
    can_right = tile_order(LA) isa TileRowMajor
    can_left = tile_order(LB) isa TileColMajor
    can_right || can_left || throw(ArgumentError(
        "canonical TLR GEMM does not yet support transA='T', transB='N': " *
        "neither contraction stack is contiguous; run-level packing/reduction " *
        "is deferred to the general-storage API"))
    can_right && !can_left && return :right
    can_left && !can_right && return :left

    qm, qk = tilegrid_size(LA)
    _, qn = tilegrid_size(LB)
    right = _right_sampling_workspace_elems(
        qm, qk, rA, rB, block, maxrankC)
    left = _left_sampling_workspace_elems(
        qn, qk, rA, rB, block, maxrankC)
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
    _store_tlr_run!(C, U, V, ranks_run, err_run, slots, ranks_dev, err_dev, slots_dev)

Scatter one run's factors and diagnostics into `C`'s canonical storage.
`slots_dev` is caller-owned scratch (sized to at least `length(slots)`,
reused across the driver's traversal of `C`'s rows/columns) rather than
allocated here, since every run in that traversal needs an identically-sized
buffer.
"""
function _store_tlr_run!(C::TLRMatrix, U, V, ranks_run, err_run,
                         slots::Vector{Int}, ranks_dev, err_dev, slots_dev)
    backend = get_backend(C)
    count = length(slots)
    copyto!(view(slots_dev, 1:count), Int32.(slots))
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
            "canonical R3 TLR gemm! currently requires regular-grid tiling"))
    return nothing
end

"""
    gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
          alpha=true, beta=false, transA='N', transB='N',
          tol=0, rel=false, eps_rel=nothing, r_required=10, block=32,
          compute=nothing) -> C

Canonical physical-row-major TLR result GEMM:

    C := alpha * op(A) * op(B) + beta * C

`C`, `A`, and `B` must all be physically `TileRowMajor`. `NN` uses right
sampling, `TT` uses left sampling, and `NT` chooses between both zero-copy
terminal stacks from rank-derived intermediate workspace. `TN` is deliberately
unsupported until the general-storage packing/reduction path is implemented.

`tol` controls final Frobenius truncation. `eps_rel` controls adaptive range
capture and defaults to `max(tol, ara_stopping_floor(promoted_type))`.
"""
function gemm!(C::TLRMatrix{BackendT,T},
               A::TLRMatrix{BackendT,T},
               B::TLRMatrix{BackendT,T};
               alpha=true, beta=false,
               transA::Char='N', transB::Char='N',
               tol::Real=0.0, rel::Bool=false,
               eps_rel=nothing, r_required::Int=10, block::Int=32,
               compute=nothing) where {BackendT,T}
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

    ops = logical_operands(LA, LB)
    LC = logical_operand(C)
    qm, qk = tilegrid_size(LA)
    _, qn = tilegrid_size(LB)
    rA, rB = _active_rank_cap(A), _active_rank_cap(B)
    blk = min(block, max(maxrank(C), 1))
    side = choose_tlr_sampling_side(LA, LB, maxrank(C), blk, rA, rB)
    backend = get_backend(C)
    ranks_dev = allocate(backend, eltype(C.ranks), qm * qn)
    err_dev = allocate(backend, Float64, qm * qn)

    if side === :right && tile_order(LB) isa TileColMajor
        # NT, right choice: fixed columns share H. U/V/rr/ee/slots_dev are
        # loop-invariant in shape (every column run has qm members), so they
        # are allocated once and reused; range_find_column_run! and
        # _store_tlr_run! both overwrite every slot they use on every call.
        U = allocate(backend, T, nominal_tile_size(C, 1), maxrank(C), qm)
        V = allocate(backend, T, nominal_tile_size(C, 2), maxrank(C), qm)
        rr = allocate(backend, eltype(C.ranks), qm)
        ee = allocate(backend, Float64, qm)
        slots_dev = allocate(backend, Int32, qm)
        for j in 1:qn
            range_find_column_run!(
                U, V, rr, ee, ops, 1:qm, j;
                alpha=α, beta=β, C=LC, eps_rel=sample_tol, r_required,
                tol, rel, block=blk, rankA=rA, rankB=rB, compute=mode,
            )
            slots = [j + (i - 1) * qn for i in 1:qm]
            _store_tlr_run!(C, U, V, rr, ee, slots, ranks_dev, err_dev, slots_dev)
        end
    else
        # NN right sampling, or NT/TT left sampling: fixed output rows.
        U = allocate(backend, T, nominal_tile_size(C, 1), maxrank(C), qn)
        V = allocate(backend, T, nominal_tile_size(C, 2), maxrank(C), qn)
        rr = allocate(backend, eltype(C.ranks), qn)
        ee = allocate(backend, Float64, qn)
        slots_dev = allocate(backend, Int32, qn)
        for i in 1:qm
            if side === :right
                range_find_row_right_run!(
                    U, V, rr, ee, ops, i, 1:qn;
                    alpha=α, beta=β, C=LC, eps_rel=sample_tol, r_required,
                    tol, rel, block=blk, rankA=rA, rankB=rB, compute=mode,
                )
            else
                range_find_row_left_run!(
                    U, V, rr, ee, ops, i, 1:qn;
                    alpha=α, beta=β, C=LC, eps_rel=sample_tol, r_required,
                    tol, rel, block=blk, rankA=rA, rankB=rB, compute=mode,
                )
            end
            slots = collect(((i - 1) * qn + 1):(i * qn))
            _store_tlr_run!(C, U, V, rr, ee, slots, ranks_dev, err_dev, slots_dev)
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
