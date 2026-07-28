# gemm! entry points for the dense-output TLR GEMM: materializes a dense
# C from TLR/dense-diagonal operands via the budgeted, region-scheduled
# low-rank terms in low_rank_terms.jl, regions/, and dense_products.jl.

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

"""
    gemm!(C, A::BCLRMatrix, B::BCLRMatrix; workspace, alpha=true, beta=false)

CUDA-only exact-rank BCLR dense accumulation. The initial implementation
supports a full regular grid and logical `N/T` operands, using grouped GEMM
for all three stages and selecting FoldRight/FoldLeft from packed layouts.
"""
function gemm!(C::AbstractMatrix, A::BCLRMatrix{BackendT,T}, B::BCLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char='N', transB::Char='N', compute=nothing,
    workspace_policy=InteriorFirstWorkspace()) where {BackendT,T}
    workspace_policy isa InteriorFirstWorkspace ||
        throw(ArgumentError("BCLR dense GEMM currently supports InteriorFirstWorkspace only"))
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    ScalarT = gemm_compute_type(mode)
    return _bclr_gemm!(C, LA, LB; workspace, alpha=ScalarT(alpha), beta=ScalarT(beta), compute=mode)
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
