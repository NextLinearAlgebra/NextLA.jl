"""Host-only rank metadata plus the fold-cost profile derived from it, for one
CompressedFTLR GEMM. See `rank_metadata.jl` for the pure rank facts and
`fold_cost.jl` for how they turn into FoldRight/FoldLeft costs."""
struct CompressedFTLRRankPlan
    a_k_prefix::Matrix{Int}      # (logical i, prefix through logical k)
    b_row_ranks::Vector{Int}     # σ_k = Σ_j rB_kj
    b_col_ranks::Vector{Int}     # γ_j = Σ_k rB_kj
    b_col_k_prefix::Matrix{Int}  # (logical j, prefix through logical k)
    b_row_k_prefix::Matrix{Int} # (logical k, prefix through logical j): Σ_{j'<j} rB_kj'
    b_col_prefix::Vector{Int}    # prefix of γ_j across logical j
    b_row_nonzero_prefix::Matrix{Int} # (logical k, prefix through j): #{j' < j : rB_kj' != 0}
    pair_ranks::Vector{Int}      # p_i = Σ_k rA_ik σ_k
    b_total_rank::Int
    output_row_heights::Vector{Int}
    output_col_widths::Vector{Int}
    output_col_prefix::Vector{Int}
    profile::RaggedWorkspaceProfile
end

"""One scheduled work unit: output tile rows `rows` × tile columns `cols`,
executed with a single `fold`. `cols` spanning the whole grid is the
whole-width case; a narrower span subdivides below one output row, which is
what lets the workspace budget be met at a granularity finer than a full row."""
struct RaggedRowRun
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    fold::Symbol
end

function _compressed_ftlr_rank_plan(A, B)
    meta = _compressed_ftlr_rank_metadata(A, B)
    profile = _compressed_ftlr_fold_cost(meta, A, B)
    return CompressedFTLRRankPlan(meta.a_k_prefix, meta.b_row_ranks, meta.b_col_ranks,
                        meta.b_col_k_prefix, meta.b_row_k_prefix, meta.b_col_prefix,
                        meta.b_row_nonzero_prefix, meta.pair_ranks, meta.b_total_rank,
                        meta.output_row_heights, meta.output_col_widths, meta.output_col_prefix,
                        profile)
end

"""Smallest workspace in which this GEMM can run at all.

This is the single-column-block floor, not the full-width per-row floor: when a
budget cannot hold a whole output row the scheduler subdivides the output into
column blocks, so the true requirement is lower. Sizing a workspace at exactly
this value is valid but yields the most subdivided (and therefore slowest)
schedule; `gemm_maximum_workspace_bytes` is the value that runs fastest."""
function gemm_minimum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    LA, LB = logical_operand(A, transA), logical_operand(B, transB)
    return _compressed_ftlr_column_floor(_compressed_ftlr_rank_plan(LA, LB), LA, LB)
end

function gemm_maximum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    return _compressed_ftlr_rank_plan(logical_operand(A, transA), logical_operand(B, transB)).profile.maximum
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
    LA, LB = logical_operand(A, transA), logical_operand(B, transB)
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
function _compressed_ftlr_row_runs(profile::RaggedWorkspaceProfile, budget::Int)
    budget >= profile.minimum || throw(ArgumentError(
        "workspace has $budget bytes; at least $(profile.minimum) bytes are required"))
    runs = RaggedRowRun[]
    i = 1
    while i <= length(profile.row_bytes)
        j = i - 1
        while j < length(profile.row_bytes) && _compressed_ftlr_select_fold(profile, i:(j + 1), budget) !== nothing
            j += 1
        end
        # A zero-work row is allowed even for a zero byte budget.
        j >= i || (j = i)
        fold = _compressed_ftlr_select_fold(profile, i:j, budget)
        fold === nothing && throw(ArgumentError("workspace cannot schedule CompressedFTLR row $i"))
        push!(runs, RaggedRowRun(i:j, profile.columns, fold))
        i = j + 1
    end
    return runs
end

function _compressed_ftlr_select_fold(profile::RaggedWorkspaceProfile, rows::UnitRange{Int}, budget::Int)
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
function _compressed_ftlr_column_schedule(meta, A, B, profile::RaggedWorkspaceProfile,
                                          budget::Int)
    # Fast path: full width fits, so behave exactly as the whole-width scheduler.
    profile.minimum <= budget && return _compressed_ftlr_schedule(profile, budget)

    qn = length(meta.output_col_widths)
    runs = RaggedRowRun[]
    j = 1
    while j <= qn
        block = _compressed_ftlr_widest_column_block(meta, A, B, j, qn, budget)
        block === nothing && throw(ArgumentError(
            "workspace has $budget bytes; not enough for a single output tile column"))
        append!(runs, _compressed_ftlr_schedule(
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

function _prepare_compressed_ftlr_workspace(A, B, plan::CompressedFTLRRankPlan, workspace)
    profile = plan.profile
    T = eltype(A)
    ws = if workspace isa Integer
        DenseGemmWorkspace(_compressed_ftlr_parent(A), Int(workspace))
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
    arena = DenseGemmArena(view(ws.storage, :), 1)
    return ws, arena, budget, profile
end
