"""Pure per-tile rank facts for one CompressedFTLR operand pair.

No cost or workspace concept lives here — only "what are the ranks and output
tile shapes." `_compressed_ftlr_fold_cost` (below) turns these facts into
FoldRight/FoldLeft cost estimates; nothing here depends on which fold is used.
"""

@inline _compressed_ftlr_right_valid(A, B) =
    compressed_ftlr_outer_order(A) isa TileRowMajor && compressed_ftlr_outer_order(B) isa TileRowMajor
@inline _compressed_ftlr_left_valid(A, B) =
    compressed_ftlr_outer_order(B) isa TileRowMajor && compressed_ftlr_inner_order(B) isa TileColMajor

@inline function _compressed_ftlr_prefix(values::Vector{Int})
    prefix = Base.zeros(Int, length(values) + 1)
    @inbounds for i in eachindex(values)
        prefix[i + 1] = prefix[i] + values[i]
    end
    return prefix
end

@inline _compressed_ftlr_range_total(prefix::Vector{Int}, rows::UnitRange{Int}) =
    prefix[last(rows) + 1] - prefix[first(rows)]

# column-range views: work is scheduled over tile rows `irange` × tile columns
# `jrange`; a narrower `jrange` than the full `1:qn` is what lets the workspace
# budget be met below one output row, so every `j`-aggregating quantity below
# is queried on the range rather than read whole.

"""`σ_k` restricted to `jrange`: `Σ_{j ∈ jrange} rB_kj`."""
@inline _compressed_ftlr_row_rank(meta, k::Int, jrange) =
    meta.b_row_k_prefix[k, last(jrange) + 1] - meta.b_row_k_prefix[k, first(jrange)]

"""Column offset of tile `j` inside a `jrange`-local S block."""
@inline _compressed_ftlr_row_rank_offset(meta, k::Int, j::Int, jrange) =
    meta.b_row_k_prefix[k, j] - meta.b_row_k_prefix[k, first(jrange)]

"""Total output width spanned by `jrange`, in matrix columns."""
@inline _compressed_ftlr_width(meta, jrange) =
    meta.output_col_prefix[last(jrange) + 1] - meta.output_col_prefix[first(jrange)]

"""Matrix column span of `jrange` in the dense output."""
@inline function _compressed_ftlr_output_cols(B, jrange)
    lo = first(_tile_axis_range(B, first(jrange), 2))
    hi = last(_tile_axis_range(B, last(jrange), 2))
    return lo:hi
end

"""
    _compressed_ftlr_rank_metadata(A, B)

Host-only rank reductions and O(1) range-query tables: `a_k_prefix[i,·]` is A's
running rank prefix over `k`; `b_row_ranks[k] = σ_k = Σ_j rB_kj`; `b_col_ranks[j]
= γ_j = Σ_k rB_kj`; `pair_ranks[i] = Σ_k rA_ik·σ_k` is the number of S-arena
elements row `i` contributes (fold-independent, since Stage 1 is shared).
`output_row_heights`/`output_col_widths` are the physical tile extents (`m_i`,
`n_j` in the paper's notation), independent of rank.
"""
function _compressed_ftlr_rank_metadata(A, B)
    qm, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    a_k_prefix = Base.zeros(Int, qm, qk + 1)
    b_row_ranks = Base.zeros(Int, qk)
    b_col_ranks = Base.zeros(Int, qn)
    b_col_k_prefix = Base.zeros(Int, qn, qk + 1)
    b_row_k_prefix = Base.zeros(Int, qk, qn + 1)
    # Count of NONZERO B tiles per contraction row, as a prefix over j. Stage 2
    # skips `(k,j)` GEMMs where `rB_kj == 0`, leaving FoldRight's arena with
    # reserved-but-unwritten space, so `three_stage.jl` must know whether such a
    # hole exists. A prefix (rather than a plain flag) keeps that query O(1)
    # for an arbitrary column range, which the column-block scheduler needs.
    b_row_nonzero_prefix = Base.zeros(Int, qk, qn + 1)
    @inbounds for k in 1:qk, j in 1:qn
        r = _compressed_ftlr_storage_rank(B, k, j)
        b_row_ranks[k] += r
        b_col_ranks[j] += r
        b_row_nonzero_prefix[k, j + 1] = b_row_nonzero_prefix[k, j] + (r != 0)
    end
    @inbounds for j in 1:qn, k in 1:qk
        b_col_k_prefix[j, k + 1] = b_col_k_prefix[j, k] + _compressed_ftlr_storage_rank(B, k, j)
    end
    @inbounds for k in 1:qk, j in 1:qn
        b_row_k_prefix[k, j + 1] = b_row_k_prefix[k, j] + _compressed_ftlr_storage_rank(B, k, j)
    end
    pair_ranks = Base.zeros(Int, qm)
    @inbounds for i in 1:qm, k in 1:qk
        r = _compressed_ftlr_storage_rank(A, i, k)
        a_k_prefix[i, k + 1] = a_k_prefix[i, k] + r
        pair_ranks[i] += r * b_row_ranks[k]
    end
    b_col_prefix = _compressed_ftlr_prefix(b_col_ranks)
    output_row_heights = [length(_tile_axis_range(A, i, 1)) for i in 1:qm]
    output_col_widths = [length(_tile_axis_range(B, j, 2)) for j in 1:qn]
    output_col_prefix = _compressed_ftlr_prefix(output_col_widths)
    return (; a_k_prefix, qk, b_col_ranks, b_col_k_prefix, b_row_k_prefix,
              b_col_prefix, b_row_nonzero_prefix, pair_ranks,
              output_row_heights, output_col_widths, output_col_prefix)
end

"""
    _compressed_ftlr_fold_cost(meta, A, B)

Closed-form per-row byte and FLOP cost for both fold bracketings, given rank
metadata (above). Mirrors the paper's boxed formulas:

    F_R(i) = 2·n_·Σ_k rA_ik·rB_k· + 2·m_i·n_·ρ_i        (Stage 2 + Stage 3, FoldRight)
    F_L(i) = 2·m_i·Σ_k rA_ik·rB_k· + 2·m_i·n_·γ_·        (Stage 2 + Stage 3, FoldLeft)

Stage 2's cost is identical either way (`pair_ranks[i]`/`Σ_k rA_ik·rB_kj` terms),
so only the Stage-3 term differs. Every quantity here is computed *unconditionally*
per row rather than short-circuited to zero when `pair_ranks[i] == 0`: a row with
no A-rank at all does reduce to zero automatically (every term it appears in is a
product with a factor that is itself zero), but a row can have `pair_ranks[i] == 0`
while still needing real Stage-3 storage/FLOPs — either because `ρ_i > 0` (A has
rank at some `k` where B's row-block `k` happens to be entirely rank-zero, so
Stage 1 contributes nothing there but Stage 3 still processes that row's own
`ρ_i`-wide operand), or because FoldLeft's Stage 3 GEMM spans a run's whole
contiguous physical row range (`run_height`), so a zero-rank row embedded between
nonzero neighbors still occupies its `row_height · b_total_rank` slice of the T
arena even though it contributes nothing to the S arena. A per-row ternary
shortcut on `pair_ranks[i]` would silently drop exactly this contribution and
under-count the true workspace/FLOP need — the unconditional formula, summed via
ordinary prefix sums over any row range, always matches what the executor
(`three_stage.jl`) actually allocates and computes.
"""
function _compressed_ftlr_fold_cost(meta, A, B,
                                    jrange::UnitRange{Int}=1:length(meta.output_col_widths))
    qm = length(meta.pair_ranks)
    qk = meta.qk
    Tbytes = sizeof(eltype(A))
    width = _compressed_ftlr_width(meta, jrange)
    total_rank = meta.b_col_prefix[last(jrange) + 1] -
                 meta.b_col_prefix[first(jrange)]

    # `pair_ranks` is stored over the whole grid; restricted to `jrange` it is
    # Σ_k rA_ik·σ_k(jrange). Recomputing costs O(qm·qk), the same order as the
    # rest of this function, and is the only per-column-block work needed.
    pair = if jrange == 1:length(meta.output_col_widths)
        meta.pair_ranks
    else
        p = Base.zeros(Int, qm)
        @inbounds for i in 1:qm, k in 1:qk
            p[i] += _compressed_ftlr_storage_rank(A, i, k) *
                    _compressed_ftlr_row_rank(meta, k, jrange)
        end
        p
    end

    right = _compressed_ftlr_right_valid(A, B) ?
        [(pair[i] + width * meta.a_k_prefix[i, end]) * Tbytes for i in 1:qm] : nothing
    left = _compressed_ftlr_left_valid(A, B) ?
        [(pair[i] + meta.output_row_heights[i] * total_rank) * Tbytes for i in 1:qm] : nothing
    (right === nothing && left === nothing) && throw(ArgumentError(
        "CompressedFTLR needs row-packed B outer factors and either row-packed A outer factors or column-packed B inner factors"))
    row_bytes = [min(right === nothing ? typemax(Int) : right[i],
                     left === nothing ? typemax(Int) : left[i]) for i in 1:qm]

    # Σ_j w_j Σ_k rA_ik rB_kj = Σ_k rA_ik (Σ_j w_j rB_kj) = Σ_k rA_ik ω_k -- hoisting
    # ω_k turns an O(qm·qn·qk) comprehension into O(qk·qn + qm·qk). The FoldLeft
    # Stage-3 term Σ_j w_j γ_j is independent of i, so it is hoisted the same way.
    # Both sums run over `jrange` only.
    omega = Base.zeros(Int, qk)
    @inbounds for k in 1:qk, j in jrange
        omega[k] += meta.output_col_widths[j] * _compressed_ftlr_storage_rank(B, k, j)
    end
    weighted_col_rank = 0
    @inbounds for j in jrange
        weighted_col_rank += meta.output_col_widths[j] * meta.b_col_ranks[j]
    end

    right_flops = right === nothing ? nothing :
        [sum((_compressed_ftlr_storage_rank(A, i, k) * omega[k] for k in 1:qk); init=0) +
         meta.output_row_heights[i] * width * meta.a_k_prefix[i, end]
         for i in 1:qm]
    left_flops = left === nothing ? nothing :
        [meta.output_row_heights[i] * (pair[i] + weighted_col_rank) for i in 1:qm]

    maximum_bytes = min(right === nothing ? typemax(Int) : sum(right),
                        left === nothing ? typemax(Int) : sum(left))
    return (
        columns=jrange, nrows=qm,
        right_byte_prefix=right === nothing ? nothing : _compressed_ftlr_prefix(right),
        left_byte_prefix=left === nothing ? nothing : _compressed_ftlr_prefix(left),
        right_flop_prefix=right_flops === nothing ? nothing : _compressed_ftlr_prefix(right_flops),
        left_flop_prefix=left_flops === nothing ? nothing : _compressed_ftlr_prefix(left_flops),
        minimum=isempty(row_bytes) ? 0 : maximum(row_bytes), maximum=maximum_bytes,
    )
end

"""One scheduled work unit: output tile rows `rows` × tile columns `cols`,
executed with a single `fold`. `cols` spanning the whole grid is the
whole-width case; a narrower span subdivides below one output row, which is
what lets the workspace budget be met at a granularity finer than a full row."""
function _compressed_ftlr_rank_plan(A, B)
    meta = _compressed_ftlr_rank_metadata(A, B)
    profile = _compressed_ftlr_fold_cost(meta, A, B)
    return (; meta..., profile)
end

"""Smallest workspace in which this GEMM can run at all.

This is the single-column-block floor, not the full-width per-row floor: when a
budget cannot hold a whole output row the scheduler subdivides the output into
column blocks, so the true requirement is lower. Sizing a workspace at exactly
this value is valid but yields the most subdivided (and therefore slowest)
schedule; `gemm_maximum_workspace_bytes` is the value that runs fastest."""
function gemm_minimum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    return _compressed_ftlr_column_floor(_compressed_ftlr_rank_plan(LA, LB), LA, LB)
end

function gemm_maximum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    return _compressed_ftlr_rank_plan(LA, LB).profile.maximum
end

"""
    gemm_workspace_bytes(A, B; runs=1, transA='N', transB='N') -> Int

Smallest workspace that schedules this GEMM into at most `runs` work units.

Run count is the quantity worth controlling: each run submits three grouped
GEMM calls, so it sets the fixed overhead, and it is what the workspace budget
actually buys. `runs = 1` is the fastest schedule and coincides with
`gemm_maximum_workspace_bytes`; the result is clamped at
`gemm_minimum_workspace_bytes`, so asking for more runs than the grid can be
split into simply returns the floor.

Useful identity for sizing: run count is very nearly reciprocal in the budget,
`runs ≈ maximum_bytes / bytes`, so "a quarter of the maximum" and "about four
runs" mean the same thing.

Prefer this over reasoning in rows per run: work units are rectangles, not whole
output rows, once the budget falls below a full-width row.
"""
function gemm_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                              runs::Int=1, transA::Char='N', transB::Char='N')
    runs >= 1 || throw(ArgumentError("runs must be positive; got $runs"))
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    plan = _compressed_ftlr_rank_plan(LA, LB)
    lo = _compressed_ftlr_column_floor(plan, LA, LB)
    hi = plan.profile.maximum
    scheduled(bytes) = length(_compressed_ftlr_column_schedule(plan, LA, LB, plan.profile, bytes))
    # Run count is monotone non-increasing in the budget, so bisect for the
    # smallest budget still meeting the target instead of scanning.
    scheduled(lo) <= runs && return lo
    while lo < hi
        mid = lo + (hi - lo) ÷ 2
        if scheduled(mid) <= runs
            hi = mid
        else
            lo = mid + 1
        end
    end
    return hi
end

"""Contiguous output-row runs greedily packed under an exact ragged budget."""
function _compressed_ftlr_row_runs(profile, budget::Int)
    budget >= profile.minimum || throw(ArgumentError(
        "workspace has $budget bytes; at least $(profile.minimum) bytes are required"))
    runs = DenseAccumulationRun[]
    i = 1
    while i <= profile.nrows
        j = i - 1
        while j < profile.nrows && _compressed_ftlr_select_fold(profile, i:(j + 1), budget) !== nothing
            j += 1
        end
        # A zero-work row is allowed even for a zero byte budget.
        j >= i || (j = i)
        fold = _compressed_ftlr_select_fold(profile, i:j, budget)
        fold === nothing && throw(ArgumentError("workspace cannot schedule CompressedFTLR row $i"))
        push!(runs, DenseAccumulationRun(i:j, profile.columns, fold))
        i = j + 1
    end
    return runs
end

function _compressed_ftlr_select_fold(profile, rows::UnitRange{Int}, budget::Int)
    right_bytes = profile.right_byte_prefix === nothing ? typemax(Int) :
        _compressed_ftlr_range_total(profile.right_byte_prefix, rows)
    left_bytes = profile.left_byte_prefix === nothing ? typemax(Int) :
        _compressed_ftlr_range_total(profile.left_byte_prefix, rows)
    right_ok = right_bytes <= budget
    left_ok = left_bytes <= budget
    !right_ok && !left_ok && return nothing
    right_ok && !left_ok && return :right
    left_ok && !right_ok && return :left
    right_flops = _compressed_ftlr_range_total(profile.right_flop_prefix, rows)
    left_flops = _compressed_ftlr_range_total(profile.left_flop_prefix, rows)
    return right_flops <= left_flops ? :right : :left
end

# ------------------------------------------------- column-block partitioning
# A row run's arenas scale with the FULL output width, so the smallest schedulable
# unit is one whole output tile row and the workspace floor is
# `max_i(row_bytes[i])`. Below that the GEMM simply cannot run, however the rows
# are grouped. Splitting the output into contiguous COLUMN blocks makes the unit
# a rectangle instead: both S and T then scale with the block's width, lowering
# the floor by roughly the same factor.
#
# The rectangles tile C, so the output is still written exactly once -- unlike
# splitting the contraction, which would turn C into a repeated read-modify-write.
# The only duplicated traffic is re-reading A's outer factor once per block, which
# is negligible against C.
#
# Blocks are only formed when a full-width schedule does not fit, so the common
# case takes the fast path below and behaves exactly as before.

"""
Widest column block starting at `j0` whose per-row workspace floor fits `budget`.

The floor is monotone in the block width (widening only adds non-negative bytes),
so this bisects instead of scanning -- O(log qn) profile builds per block rather
than O(qn). Returns `nothing` when even the single column `j0` cannot fit, which
is a genuinely unschedulable workspace.
"""
function _compressed_ftlr_widest_column_block(meta, A, B, j0::Int, qn::Int, budget::Int)
    _compressed_ftlr_fold_cost(meta, A, B, j0:j0).minimum <= budget || return nothing
    lo, hi = j0, qn                      # lo always feasible, hi+1 always infeasible
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if _compressed_ftlr_fold_cost(meta, A, B, j0:mid).minimum <= budget
            lo = mid
        else
            hi = mid - 1
        end
    end
    return j0:lo
end

"""
Schedule the whole output under `budget`, subdividing into column blocks only if
a full-width schedule does not fit.

Each block is scheduled independently by the row-run scheduler and the blocks
execute sequentially, resetting the arena between runs -- so peak workspace stays
the maximum over runs rather than their sum.
"""
function _compressed_ftlr_column_schedule(meta, A, B, profile,
                                          budget::Int)
    # Fast path: full width fits, so behave exactly as the whole-width scheduler.
    profile.minimum <= budget && return _compressed_ftlr_row_runs(profile, budget)

    qn = length(meta.output_col_widths)
    runs = DenseAccumulationRun[]
    j = 1
    while j <= qn
        block = _compressed_ftlr_widest_column_block(meta, A, B, j, qn, budget)
        block === nothing && throw(ArgumentError(
            "workspace has $budget bytes; not enough for a single output tile column"))
        append!(runs, _compressed_ftlr_row_runs(
            _compressed_ftlr_fold_cost(meta, A, B, block), budget))
        j = last(block) + 1
    end
    return runs
end

"""Smallest workspace in which any schedule exists: the per-row floor of the
narrowest (single-column) blocks, which is what column subdivision buys over the
full-width floor `profile.minimum`."""
function _compressed_ftlr_column_floor(meta, A, B)
    qn = length(meta.output_col_widths)
    floor_bytes = 0
    @inbounds for j in 1:qn
        floor_bytes = max(floor_bytes, _compressed_ftlr_fold_cost(meta, A, B, j:j).minimum)
    end
    return floor_bytes
end

function _prepare_compressed_ftlr_workspace(A, B, plan, workspace)
    profile = plan.profile
    T = eltype(A)
    ws = if workspace isa Int
        DenseGemmWorkspace(A, workspace)
    elseif workspace isa DenseGemmWorkspace
        eltype(workspace) === T || throw(ArgumentError(
            "workspace element type $(eltype(workspace)) does not match operand type $T"))
        typeof(get_backend(workspace.storage)) === typeof(get_backend(A)) ||
            throw(ArgumentError("workspace and operands must use the same backend"))
        workspace
    else
        throw(ArgumentError("workspace must be an integer byte count or DenseGemmWorkspace"))
    end
    # `profile.minimum` is the FULL-WIDTH per-row floor. Falling below it is not
    # fatal any more -- the scheduler subdivides into column blocks -- so only
    # pay for the (more expensive) true floor when the cheap bound fails.
    bytes = sizeof(ws)
    if bytes < profile.minimum
        required = _compressed_ftlr_column_floor(plan, A, B)
        bytes >= required || throw(ArgumentError(
            "workspace has $bytes bytes; at least $required bytes are required"))
    end
    budget = min(bytes, profile.maximum)
    arena = GemmArena(view(ws.storage, :), 1)
    return ws, arena, budget, profile
end
