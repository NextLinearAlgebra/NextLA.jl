# ─── TLR × TLR → dense GEMM ─────────────────────────────────────────────────────
#
# Computes  C = A·B  for two TLR matrices into a DENSE column-major `C`.
# Splitting each operand as diagonal + off-diagonal,
#
#   A·B = (D_A + O_A)(D_B + O_B)
#       = D_A D_B  +  O_A D_B  +  D_A O_B  +  O_A O_B
#
# the hard term is  O_A O_B, computed by the three-stage scheme (see `layout.jl`).
# The other three terms are single-stage / dense batched products accumulated into
# the same dense `C`; they are handled separately and added incrementally.
#
# Architecture:
#   core/   — the shared substrate every term is built from:
#     layout.jl   — `Stride1Axis`, `KAxisSchedule`, `FreeAxisSchedule` traits
#     panel.jl    — zero-copy `PanelView`s + logical operands
#     schedule.jl — budgeted row/column run iterators + S/T scratch & batch buffers
#     stage.jl    — stage descriptors and direct `gemm_batched!` emission
#   terms/  — one file per contribution to C (reuse `core/`):
#     diagonal.jl — the three easy diagonal terms (D_AD_B, O_AD_B, D_AO_B)
#     corner.jl   — boundary corner term  vᵀx  (needs m%b ≠ 0)

include("core/layout.jl")
include("core/panel.jl")
include("core/schedule.jl")
include("core/stage.jl")
include("terms/diagonal.jl")
include("terms/corner.jl")

# ─── Hard term:  O_A O_B  ───────────────────────────────────────────────────────

"""
    _offdiag_gemm!(C, A, B; alpha, budget) -> C

Accumulate `alpha · O_A O_B` into the dense `C`.  Selects the reduction placement
from the operand layouts, then for each budgeted run lowers Stage 1/2/3 straight
to `gemm_batched!` via `execute_stage!`.
"""
function _offdiag_gemm!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix{<:Any,T};
                        alpha, budget) where {T}
    _, nt = _interior_grid(A)
    (nt == 1 || A.maxrank == 0 || B.maxrank == 0) && return C

    placement = k_axis_schedule(stride1_axis_left(A), stride1_axis_right(B))
    ops = logical_operands(A, B)
    ws = allocate_workspace(placement, A, B, C, budget)
    alpha_t = T(alpha)

    for run in runs(placement, A, B, budget)
        prepare_run!(placement, run, ws)
        execute_stage!(stage1(placement, run, ops, ws))
        execute_stage!(stage2(placement, run, ops, ws))
        execute_stage!(stage3(placement, run, ops, ws, C, alpha_t))
    end
    return C
end

# ─── Entry point ────────────────────────────────────────────────────────────────

# Default off-diagonal-product workspace budget (bytes) for the run scheduler.
const _DEFAULT_GEMM_BUDGET = 128 * 1024 * 1024

"""
    gemm!(C, A, B; alpha=true, beta=false, max_workspace=_DEFAULT_GEMM_BUDGET) -> C

Compute `C := alpha·(A·B) + beta·C` for TLR matrices `A`, `B` into the dense
column-major matrix `C`.

The output traversal of `C` is a function of the operand layouts (`A.order`,
`B.order`) — not a free knob.  The only tunable is `max_workspace` (bytes), which
sets how long a contiguous run of `C` is materialized at once (see `schedule.jl`).
"""
function gemm!(C::AbstractMatrix, A::TLRMatrix, B::TLRMatrix;
               alpha=true, beta=false, max_workspace::Int=_DEFAULT_GEMM_BUDGET)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("size(A,2) must equal size(B,1)"))
    (size(C, 1), size(C, 2)) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("C must be size(A,1) × size(B,2)"))
    blocksize(A) == blocksize(B) ||
        throw(DimensionMismatch("A and B must share the same block size"))
    A.tile_m == A.tile_n == B.tile_m == B.tile_n ||
        throw(DimensionMismatch("gemm! currently requires square tiles"))

    α = eltype(C)(alpha)
    β = eltype(C)(beta)
    _easy_terms!(C, A, B, α, β)                              # β·C + D_AD_B + O_AD_B + D_AO_B
    _offdiag_gemm!(C, A, B; alpha=α, budget=max_workspace)   # O_A O_B  (accumulate, β=1)
    # Boundary hard terms (only fire when m % b ≠ 0); β already applied above.
    _corner_term!(C, A, B, α; beta=one(eltype(C)))          # vᵀx  (corner block)
    return C
end

"""
    _easy_terms!(C, A, B, α, β) -> C

Run the three easy terms with `β·C` folded into the first GEMM that touches each
tile, so no separate `_scale_output!` pass is needed. Diagonal tiles are written
only by `D_A D_B`; off-diagonal tiles by both `O_A D_B` and `D_A O_B`, so the
first present carries `β` and the second accumulates with `β = 1`. The hard term
`O_A O_B` then runs last with `β = 1`.

The diagonal group (`D_A D_B`) and the off-diagonal group (`O_A D_B` + `D_A O_B`)
write disjoint tile sets, so on a GPU backend they run concurrently on two
streams; both are synced before returning (the hard term touches both sets).
"""
function _easy_terms!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, α, β) where {T}
    if A.maxrank == 0 && B.maxrank == 0
        # Product is block-diagonal: nothing writes the off-diagonal of C, so it
        # must be scaled explicitly; D_A D_B then accumulates onto the diagonal.
        _scale_output!(C, β)
        _diag_diag!(C, A, B, α)
        return C
    end

    diagonal_group = () -> _diag_diag!(C, A, B, α; beta=β)      # diagonal first-writer
    offdiagonal_group = function ()
        if A.maxrank > 0
            _offdiag_diag!(C, A, B, α; beta=β)                 # off-diagonal first-writer
            _diag_offdiag!(C, A, B, α; beta=one(T))            # accumulate (no-op if B.maxrank==0)
        else
            _diag_offdiag!(C, A, B, α; beta=β)                 # B.maxrank>0: off-diagonal first-writer
        end
    end

    backend = A.backend
    if backend isa KernelAbstractions.CPU
        diagonal_group()
        offdiagonal_group()
    else
        streams = create_streams(backend, 2)
        with_stream(diagonal_group, backend, streams[1])
        with_stream(offdiagonal_group, backend, streams[2])
        for s in streams
            sync_stream(backend, s)
        end
    end
    return C
end
