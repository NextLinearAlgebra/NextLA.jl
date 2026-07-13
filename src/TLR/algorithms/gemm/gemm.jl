# imports
include("core/layout.jl")
include("core/panel.jl")
include("core/schedule.jl")
include("core/stage.jl")
include("terms/interior.jl")
include("terms/corner.jl")
include("terms/right.jl")
include("terms/bottom.jl")

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
          transA='N', transB='N') -> C

Compute `C := alpha·(op(A)·op(B)) + beta·C` for dense-diagonal TLR matrices `A`,
`B` into the dense column-major matrix `C`. `transA` and `transB` accept
case-insensitive `N/T`. Transposed operands currently require square matrices with
equal square tiling.

The output traversal of `C` is a function of the operand layouts (`A.order`,
`B.order`) — not a free knob.  The only tunable is `max_workspace` (bytes), which
sets how long a contiguous run of `C` is materialized at once (see `schedule.jl`).
"""
function gemm!(C::AbstractMatrix, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T};
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
    transA::Char=('N'), transB::Char=('N')) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_logical_gemm(C, LA, LB)
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

    α = T(alpha)
    β = T(beta)
    one_β = one(T)
    W = max_workspace

    interior = () -> begin                                        # C_int
        tlr_gemm_int_by_int(C, LA, LB, α, β; budget=W)             #   A_int B_int  (folds β)
        tlr_gemm_rpanel_by_bpanel(C, LA, LB, α; beta=one_β, budget=W)  # u_A v_Bᵀ  (accumulate)
    end
    right = () -> begin                                          # C_right
        tlr_gemm_int_by_rpanel(C, LA, LB, α; beta=β, budget=W)     #   A_int u_B    (folds β)
        tlr_gemm_rpanel_by_corner(C, LA, LB, α; beta=one_β)        #   u_A γ_B      (accumulate)
    end
    bottom = () -> begin                                         # C_bottom
        tlr_gemm_bpanel_by_int(C, LA, LB, α; beta=β, budget=W)     #   v_Aᵀ B_int   (folds β)
        tlr_gemm_corner_by_bpanel(C, LA, LB, α; beta=one_β)        #   γ_A v_Bᵀ     (accumulate)
    end
    corner = () -> begin                                         # C_corner
        tlr_gemm_corner_by_corner(C, LA, LB, α; beta=β)            #   γ_A γ_B       (folds β)
        tlr_gemm_bpanel_by_rpanel(C, LA, LB, α; beta=one_β)        #   v_Aᵀ u_B     (accumulate)
    end

    backend = get_backend(A)
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
"""
function gemm!(C::AbstractMatrix, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T};
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET,
    transA::Char=('N'), transB::Char=('N')) where {BackendT,T}
    LA = logical_operand(A, transA)
    LB = logical_operand(B, transB)
    _validate_logical_gemm(C, LA, LB)

    α = T(alpha)
    β = T(beta)
    one_β = one(T)
    W = max_workspace

    if maxrank(A) == 0 || maxrank(B) == 0
        _scale_output!(C, β)
        return C
    end

    interior = () -> begin                                       # C_int
        _offdiag_offdiag_gemm!(C, LA, LB; alpha=α, beta=β, budget=W)  # op(A)ᵢ op(B)ᵢ (folds β)
        tlr_gemm_rpanel_by_bpanel(C, LA, LB, α; beta=one_β, budget=W)  # u_A v_Bᵀ (no-op when aligned)
    end
    right = () -> begin                                          # C_right
        tlr_gemm_int_by_rpanel(C, LA, LB, α; beta=β, budget=W)         # A_int u_B (folds β)
        tlr_gemm_rpanel_by_corner(C, LA, LB, α; beta=one_β)           # u_A γ_B
    end
    bottom = () -> begin                                         # C_bottom
        tlr_gemm_bpanel_by_int(C, LA, LB, α; beta=β, budget=W)         # v_Aᵀ B_int (folds β)
        tlr_gemm_corner_by_bpanel(C, LA, LB, α; beta=one_β)          # γ_A v_Bᵀ
    end
    corner = () -> begin                                         # C_corner
        tlr_gemm_corner_by_corner(C, LA, LB, α; beta=β)               # γ_A γ_B (folds β)
        tlr_gemm_bpanel_by_rpanel(C, LA, LB, α; beta=one_β)          # v_Aᵀ u_B
    end

    backend = get_backend(A)
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
