# imports
include("precision.jl")
include("lowering/strategy.jl")
include("operands.jl")
include("row_basis_workspace.jl")
include("row_basis.jl")
include("lowering/schedule.jl")
include("coefficient_accumulation.jl")
include("orthogonal_merge.jl")
include("row_basis_gemm.jl")
include("lowering/stages.jl")
include("direct.jl")
include("tlr_output.jl")
include("regions/interior.jl")
include("regions/corner.jl")
include("regions/right.jl")
include("regions/bottom.jl")
include("tlr_dense.jl")

const DEFAULT_GEMM_BUDGET = 10^9

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
    gemm!(C, A, B; alpha=true, beta=false, max_workspace=DEFAULT_GEMM_BUDGET,
          transA='N', transB='N', compute=nothing) -> C

Compute `C := alpha·(op(A)·op(B)) + beta·C` for dense-diagonal TLR matrices `A`,
`B` into the dense column-major matrix `C`. `transA` and `transB` accept
case-insensitive `N/T`. Transposed operands currently require square matrices with
equal square tiling.

The output traversal of `C` is a function of the operand layouts (`A.order`,
`B.order`) — not a free knob. `max_workspace` (bytes) sets how long a contiguous
run of `C` is materialized at once (see `lowering/schedule.jl`).
`compute` selects the accumulation mode; when omitted it defaults to `Float32` for
`Float16` operands and otherwise to the operand type. `alpha` and `beta` are
converted to that compute type.
"""
function gemm!(C::AbstractMatrix, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T};
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
    transA::Char=('N'), transB::Char=('N'), compute=nothing) where {BackendT,T}
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
    W = max_workspace

    interior = () -> begin                                        # C_int
        tlr_gemm_int_by_int(C, LA, LB, α, β; budget=W, compute=mode)             #   A_int B_int  (folds β)
        tlr_gemm_rpanel_by_bpanel(C, LA, LB, α; beta=one_β, budget=W, compute=mode)  # u_A v_Bᵀ  (accumulate)
    end
    right = () -> begin                                          # C_right
        tlr_gemm_int_by_rpanel(C, LA, LB, α; beta=β, budget=W, compute=mode)     #   A_int u_B    (folds β)
        tlr_gemm_rpanel_by_corner(C, LA, LB, α; beta=one_β, budget=W, compute=mode)        #   u_A γ_B      (accumulate)
    end
    bottom = () -> begin                                         # C_bottom
        tlr_gemm_bpanel_by_int(C, LA, LB, α; beta=β, budget=W, compute=mode)     #   v_Aᵀ B_int   (folds β)
        tlr_gemm_corner_by_bpanel(C, LA, LB, α; beta=one_β, budget=W, compute=mode)        #   γ_A v_Bᵀ     (accumulate)
    end
    corner = () -> begin                                         # C_corner
        tlr_gemm_corner_by_corner(C, LA, LB, α; beta=β, compute=mode)            #   γ_A γ_B       (folds β)
        tlr_gemm_bpanel_by_rpanel(C, LA, LB, α; beta=one_β, budget=W, compute=mode)        #   v_Aᵀ u_B     (accumulate)
    end

    if backend isa KernelAbstractions.CPU
        interior();
        right();
        bottom();
        corner()
    else
        streams = create_streams(backend, 4)
        with_stream(interior, backend, streams[1])
        with_stream(right, backend, streams[2])
        with_stream(bottom, backend, streams[3])
        with_stream(corner, backend, streams[4])
        for s in streams
            sync_stream(backend, s)
        end
    end
    return C
end

"""
    gemm!(C, A::TLRMatrix, B::TLRMatrix; alpha=true, beta=false, max_workspace) -> C

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
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
    transA::Char=('N'), transB::Char=('N'), compute=nothing) where {BackendT,T}
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
    W = max_workspace

    if maxrank(A) == 0 || maxrank(B) == 0
        _scale_output!(C, β)
        return C
    end

    interior = () -> begin                                       # C_int
        _offdiag_offdiag_gemm!(C, LA, LB; alpha=α, beta=β, budget=W, compute=mode)  # op(A)ᵢ op(B)ᵢ (folds β)
        tlr_gemm_rpanel_by_bpanel(C, LA, LB, α; beta=one_β, budget=W, compute=mode)  # u_A v_Bᵀ (no-op when aligned)
    end
    right = () -> begin                                          # C_right
        tlr_gemm_int_by_rpanel(C, LA, LB, α; beta=β, budget=W, compute=mode)         # A_int u_B (folds β)
        tlr_gemm_rpanel_by_corner(C, LA, LB, α; beta=one_β, budget=W, compute=mode)           # u_A γ_B
    end
    bottom = () -> begin                                         # C_bottom
        tlr_gemm_bpanel_by_int(C, LA, LB, α; beta=β, budget=W, compute=mode)         # v_Aᵀ B_int (folds β)
        tlr_gemm_corner_by_bpanel(C, LA, LB, α; beta=one_β, budget=W, compute=mode)          # γ_A v_Bᵀ
    end
    corner = () -> begin                                         # C_corner
        tlr_gemm_corner_by_corner(C, LA, LB, α; beta=β, compute=mode)               # γ_A γ_B (folds β)
        tlr_gemm_bpanel_by_rpanel(C, LA, LB, α; beta=one_β, budget=W, compute=mode)          # v_Aᵀ u_B
    end

    if backend isa KernelAbstractions.CPU
        interior();
        right();
        bottom();
        corner()
    else
        # Regions write disjoint quadrants → independent streams, one host sync at end.
        streams = create_streams(backend, 4)
        with_stream(interior, backend, streams[1])
        with_stream(right, backend, streams[2])
        with_stream(bottom, backend, streams[3])
        with_stream(corner, backend, streams[4])
        for s in streams
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
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
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
    return _tlr_dense_gemm!(C, LA, LB, ScalarT(alpha), ScalarT(beta),
                            max_workspace, mode)
end

"""
    gemm!(C, A::AbstractMatrix, B::TLRMatrix; ...)

Compute `C := alpha·op(A)·op(B) + beta·C` with a standalone dense left operand
and a fully low-rank right operand. Intermediates retain the operand storage type.
"""
function gemm!(C::AbstractMatrix, A::AbstractMatrix{T}, B::TLRMatrix{BackendT,T};
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
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
    return _dense_tlr_gemm!(C, LA, LB, ScalarT(alpha), ScalarT(beta),
                            max_workspace, mode)
end

# ─── TLR × TLR → TLR (milestone 4) ────────────────────────────────────────────

@inline function _validate_tlr_output(C::TLRMatrix, LA::LogicalTLROperand, LB::LogicalTLROperand)
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match: size(op(A),2) == size(op(B),1)"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("op(A)'s column tile size must equal op(B)'s row tile size (contraction tiling)"))
    (nominal_tile_size(C, 1) == nominal_tile_size(LA, 1) &&
     nominal_tile_size(C, 2) == nominal_tile_size(LB, 2)) ||
        throw(DimensionMismatch("C's tile size must be (op(A) row tile, op(B) col tile)"))
    (tail_tile_size(LA, 1) == 0 && tail_tile_size(LA, 2) == 0 &&
     tail_tile_size(LB, 2) == 0 && tail_tile_size(C, 1) == 0 && tail_tile_size(C, 2) == 0) ||
        throw(ArgumentError("TLR-output GEMM currently requires aligned (regular-grid) tiling on all axes"))
    return nothing
end

function _run_tlr_output!(C::TLRMatrix, ops, geom, placement::KAsGemmK,
                          fold, alpha, budget, compute;
                          eps_sq::Float64, rel::Bool)
    ws = _alloc_tlr_output_workspace(C, geom, placement, ops, budget, fold)
    return _tlr_gemm_rowfamily!(C, ops, geom, placement, fold, alpha, budget,
                                compute, ws; eps_sq, rel)
end

_run_tlr_output!(::TLRMatrix, ops, geom, ::KAsSerialLoop, fold, alpha, budget, compute;
                 eps_sq::Float64, rel::Bool) =
    throw(ArgumentError("TLR-output GEMM for the column-family layout " *
                        "(A tile-column-major × B tile-row-major) is not yet supported"))

"""
    gemm!(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix; alpha=true, beta=false,
          tol=0.0, rel=false, max_workspace, transA='N', transB='N', compute=nothing) -> C

Fully low-rank `C := alpha·op(A)·op(B)` compressed in place into the TLR container `C`.
Currently restricted to regular-grid tiling (no boundary tiles), `beta == 0`, and the
three row-family layout pairs (every order pair except A tile-column-major with B
tile-row-major).

Each output tile is accumulated into a bounded dense slab and then compressed with the
randomized-sketch `compress!` core; `tol`/`rel` are the per-tile approximation budget
(distinct from the `max_workspace` byte budget). `residuals(C)` reports the achieved
per-tile error — a tile whose true rank exceeds `maxrank(C)` keeps full rank and reports
a residual above `tol`.
"""
function gemm!(C::TLRMatrix{BackendT,T}, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T};
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
    transA::Char='N', transB::Char='N', compute=nothing,
    tol::Real=0.0, rel::Bool=false) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_tlr_output(C, LA, LB)
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    backend = get_backend(C)
    validate_tlr_gemm_precision(backend, T, T, mode)
    ScalarT = gemm_compute_type(mode)

    α = ScalarT(alpha)
    # The row-basis path handles every non-transposed layout on both the CPU and
    # CUDA backends (the shared A row basis and B Z stack are contiguous for the
    # preferred layout, packed otherwise). Transposed operands fall through to the
    # M4 dense fallback, which does not support beta.
    if !_istrans(LA) && !_istrans(LB)
        return _row_basis_gemm!(C, A, B; alpha=α, beta=ScalarT(beta), tol, rel, compute=mode)
    end
    if maxrank(A) == 0 || maxrank(B) == 0
        fill!(C.ranks, zero(eltype(C.ranks)))
        fill!(C.resid, 0.0)
        return C
    end
    iszero(beta) || throw(ArgumentError("TLR-output GEMM with transposed operands (M4 fallback) does not support beta != 0"))
    ops = logical_operands(LA, LB)
    geom = interior_geometry(LA, LB)
    fold = choose_fold(ops)
    placement = placement_for_fold(fold, ops)
    _run_tlr_output!(C, ops, geom, placement, fold, α, max_workspace, mode;
                     eps_sq=Float64(tol)^2, rel)
    return C
end
