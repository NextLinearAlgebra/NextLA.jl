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

    # The four output regions write DISJOINT quadrants of C, so they run on four
    # independent streams. Within a region the first-writer term folds β·C and the
    # second accumulates onto the SAME tiles — FIFO ordering on the region's stream
    # enforces that dependency, so no per-region barrier is needed; one host sync at
    # the end. All boundary regions are no-ops on the square, m%b==0 path.
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
