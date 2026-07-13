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

"""
    gemm!(C, A, B; alpha=true, beta=false, max_workspace=DEFAULT_GEMM_BUDGET) -> C

Compute `C := alpha·(A·B) + beta·C` for TLR matrices `A`, `B` into the dense
column-major matrix `C`.

The output traversal of `C` is a function of the operand layouts (`A.order`,
`B.order`) — not a free knob.  The only tunable is `max_workspace` (bytes), which
sets how long a contiguous run of `C` is materialized at once (see `schedule.jl`).
"""
function gemm!(C::AbstractMatrix, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T};
    alpha=true, beta=false, max_workspace::Int=DEFAULT_GEMM_BUDGET) where {BackendT,T}
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("size(A,2) must equal size(B,1)"))
    (size(C, 1), size(C, 2)) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("C must be size(A,1) × size(B,2)"))
    nominal_tile_size(A) == nominal_tile_size(B) ||
        throw(DimensionMismatch("A and B must share the same nominal tile size"))

    α = T(alpha)
    β = T(beta)
    one_β = one(T)
    W = max_workspace

    interior = () -> begin                                        # C_int
        tlr_gemm_int_by_int(C, A, B, α, β; budget=W)             #   A_int B_int  (folds β)
        tlr_gemm_rpanel_by_bpanel(C, A, B, α; beta=one_β, budget=W)  # u_A v_Bᵀ  (accumulate)
    end
    right = () -> begin                                          # C_right
        tlr_gemm_int_by_rpanel(C, A, B, α; beta=β, budget=W)     #   A_int u_B    (folds β)
        tlr_gemm_rpanel_by_corner(C, A, B, α; beta=one_β)        #   u_A γ_B      (accumulate)
    end
    bottom = () -> begin                                         # C_bottom
        tlr_gemm_bpanel_by_int(C, A, B, α; beta=β, budget=W)     #   v_Aᵀ B_int   (folds β)
        tlr_gemm_corner_by_bpanel(C, A, B, α; beta=one_β)        #   γ_A v_Bᵀ     (accumulate)
    end
    corner = () -> begin                                         # C_corner
        tlr_gemm_corner_by_corner(C, A, B, α; beta=β)            #   γ_A γ_B       (folds β)
        tlr_gemm_bpanel_by_rpanel(C, A, B, α; beta=one_β)        #   v_Aᵀ u_B     (accumulate)
    end

    backend = A.backend
    if backend isa KernelAbstractions.CPU
        interior(); right(); bottom(); corner()
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
    transA::Char='N', transB::Char='N') where {BackendT,T}
    tA = _istrans(transA)
    tB = _istrans(transB)
    # Effective (op-applied) dimensions and contraction tile sizes.
    mA, kA = tA ? (size(A, 2), size(A, 1)) : (size(A, 1), size(A, 2))
    kB, nB = tB ? (size(B, 2), size(B, 1)) : (size(B, 1), size(B, 2))
    kA == kB ||
        throw(DimensionMismatch("inner dimensions must match: size(op(A),2) == size(op(B),1)"))
    (size(C, 1), size(C, 2)) == (mA, nB) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    # Only the contraction tiling must align: op(A)'s column tile size == op(B)'s row
    # tile size. Tiles may otherwise be rectangular (bm ≠ bn), which only resizes the
    # intermediate buffers. Row/column tails then match automatically since kA == kB.
    tcA = tA ? nominal_tile_size(A, 1) : nominal_tile_size(A, 2)
    trB = tB ? nominal_tile_size(B, 2) : nominal_tile_size(B, 1)
    tcA == trB ||
        throw(DimensionMismatch("op(A)'s column tile size must equal op(B)'s row tile size (contraction tiling)"))
    # Phase 1: transpose is handled only in the interior term, so it requires
    # boundary-free (aligned) operands (the panel/corner terms are not yet op-aware).
    if tA || tB
        (_is_aligned(A) && _is_aligned(B)) ||
            throw(ArgumentError("transA/transB ≠ 'N' currently requires boundary-free (aligned) TLR matrices"))
    end

    α = T(alpha)
    β = T(beta)
    one_β = one(T)
    W = max_workspace

    if maxrank(A) == 0 || maxrank(B) == 0
        _scale_output!(C, β)
        return C
    end

    interior = () -> begin                                       # C_int
        _offdiag_offdiag_gemm!(C, A, B; alpha=α, beta=β, budget=W, transA=transA, transB=transB)  # op(A)ᵢ op(B)ᵢ (folds β)
        tlr_gemm_rpanel_by_bpanel(C, A, B, α; beta=one_β, budget=W)  # u_A v_Bᵀ (no-op when aligned)
    end
    right = () -> begin                                          # C_right
        tlr_gemm_int_by_rpanel(C, A, B, α; beta=β, budget=W)         # A_int u_B (folds β)
        tlr_gemm_rpanel_by_corner(C, A, B, α; beta=one_β)           # u_A γ_B
    end
    bottom = () -> begin                                         # C_bottom
        tlr_gemm_bpanel_by_int(C, A, B, α; beta=β, budget=W)         # v_Aᵀ B_int (folds β)
        tlr_gemm_corner_by_bpanel(C, A, B, α; beta=one_β)          # γ_A v_Bᵀ
    end
    corner = () -> begin                                         # C_corner
        tlr_gemm_corner_by_corner(C, A, B, α; beta=β)               # γ_A γ_B (folds β)
        tlr_gemm_bpanel_by_rpanel(C, A, B, α; beta=one_β)          # v_Aᵀ u_B
    end

    backend = A.backend
    if backend isa KernelAbstractions.CPU
        interior(); right(); bottom(); corner()
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
