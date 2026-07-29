"""Helpers used by the CompressedFTLR symbolic dense-output benchmark."""
module DenseGemmCommon

using NextLA.TLRmodule:
    CompressedFTLRMatrix,
    grid_size,
    logical_operand,
    _compressed_ftlr_axis_range,
    _compressed_ftlr_execution_rank,
    _compressed_ftlr_rank,
    _compressed_ftlr_rank_plan,
    _compressed_ftlr_row_runs

include(joinpath(@__DIR__, "operand_generation.jl"))
using .ExperimentMatrixGeneration: generate_ftlr_operands

export generate_ftlr_operands

function _row_run_workspace_bytes(
    A::CompressedFTLRMatrix,
    B::CompressedFTLRMatrix,
    rows::Int,
)
    rows >= 1 || throw(ArgumentError("rows per run must be positive"))
    profile = _compressed_ftlr_rank_plan(
        logical_operand(A), logical_operand(B)).profile
    width = min(rows, length(profile.row_bytes))
    best = 0
    for first in 1:(length(profile.row_bytes) - width + 1)
        last = first + width - 1
        right = profile.right_byte_prefix === nothing ? typemax(Int) :
            profile.right_byte_prefix[last + 1] - profile.right_byte_prefix[first]
        left = profile.left_byte_prefix === nothing ? typemax(Int) :
            profile.left_byte_prefix[last + 1] - profile.left_byte_prefix[first]
        best = max(best, min(right, left))
    end
    return clamp(best, profile.minimum, profile.maximum)
end

function _tlr_tlr_executed_flops(
    A::CompressedFTLRMatrix,
    B::CompressedFTLRMatrix,
    workspace_bytes,
)
    LA, LB = logical_operand(A), logical_operand(B)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    budget = min(Int(workspace_bytes), plan.profile.maximum)
    flops = 0.0
    _, qk = grid_size(LA)
    _, qn = grid_size(LB)
    N = size(LB, 2)
    for run in _compressed_ftlr_row_runs(plan.profile, budget), i in run.rows
        plan.pair_ranks[i] == 0 && continue
        mi = length(_compressed_ftlr_axis_range(LA, i, 1))
        common = 0.0
        fold_specific = 0.0
        for l in 1:qk, j in 1:qn
            ra = _compressed_ftlr_execution_rank(LA, i, l)
            rb = _compressed_ftlr_execution_rank(LB, l, j)
            (ra == 0 || rb == 0) && continue
            tk = length(_compressed_ftlr_axis_range(LA, l, 2))
            nj = length(_compressed_ftlr_axis_range(LB, j, 2))
            common += tk * ra * rb
            fold_specific += run.fold === :right ? nj * ra * rb : mi * ra * rb
        end
        terminal = if run.fold === :right
            mi * N * plan.a_k_prefix[i, end]
        else
            sum(mi * plan.output_col_widths[j] * plan.b_col_ranks[j]
                for j in 1:qn)
        end
        flops += 2.0 * (common + fold_specific + terminal)
    end
    return flops
end

function _tlr_tlr_exact_flops(
    A::CompressedFTLRMatrix,
    B::CompressedFTLRMatrix,
    workspace_bytes,
)
    LA, LB = logical_operand(A), logical_operand(B)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    budget = min(Int(workspace_bytes), plan.profile.maximum)
    flops = 0.0
    _, qk = grid_size(LA)
    _, qn = grid_size(LB)
    N = size(LB, 2)
    for run in _compressed_ftlr_row_runs(plan.profile, budget), i in run.rows
        mi = length(_compressed_ftlr_axis_range(LA, i, 1))
        common = 0.0
        fold_specific = 0.0
        exact_a_total = 0
        exact_b_cols = zeros(Int, qn)
        for l in 1:qk
            exact_a_total += _compressed_ftlr_rank(LA, i, l)
            for j in 1:qn
                ra = _compressed_ftlr_rank(LA, i, l)
                rb = _compressed_ftlr_rank(LB, l, j)
                exact_b_cols[j] += rb
                (ra == 0 || rb == 0) && continue
                tk = length(_compressed_ftlr_axis_range(LA, l, 2))
                nj = length(_compressed_ftlr_axis_range(LB, j, 2))
                common += tk * ra * rb
                fold_specific += run.fold === :right ? nj * ra * rb : mi * ra * rb
            end
        end
        terminal = if run.fold === :right
            mi * N * exact_a_total
        else
            sum(mi * plan.output_col_widths[j] * exact_b_cols[j] for j in 1:qn)
        end
        flops += 2.0 * (common + fold_specific + terminal)
    end
    return flops
end

end
