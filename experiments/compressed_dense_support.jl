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
    _compressed_ftlr_column_schedule

include(joinpath(@__DIR__, "operand_generation.jl"))
using .ExperimentMatrixGeneration: generate_ftlr_operands

export generate_ftlr_operands

function _run_has_execution_work(A, B, run, qk)
    @inbounds for i in run.rows, k in 1:qk, j in run.cols
        if _compressed_ftlr_execution_rank(A, i, k) > 0 &&
           _compressed_ftlr_execution_rank(B, k, j) > 0
            return true
        end
    end
    return false
end

function _tlr_tlr_executed_flops(A::CompressedFTLRMatrix,
                                  B::CompressedFTLRMatrix,
                                  workspace_bytes)
    LA, LB = logical_operand(A), logical_operand(B)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    budget = min(Int(workspace_bytes), plan.profile.maximum)
    flops = 0.0
    _, qk = grid_size(LA)
    runs = _compressed_ftlr_column_schedule(
        plan, LA, LB, plan.profile, budget)
    for run in runs
        _run_has_execution_work(LA, LB, run, qk) || continue
        width = sum(plan.output_col_widths[j] for j in run.cols)
        for i in run.rows
            mi = length(_compressed_ftlr_axis_range(LA, i, 1))
            common = 0.0
            fold_specific = 0.0
            for l in 1:qk, j in run.cols
                ra = _compressed_ftlr_execution_rank(LA, i, l)
                rb = _compressed_ftlr_execution_rank(LB, l, j)
                (ra == 0 || rb == 0) && continue
                tk = length(_compressed_ftlr_axis_range(LA, l, 2))
                nj = length(_compressed_ftlr_axis_range(LB, j, 2))
                common += tk * ra * rb
                fold_specific += run.fold === :right ?
                    nj * ra * rb : mi * ra * rb
            end
            terminal = if run.fold === :right
                mi * width * plan.a_k_prefix[i, end]
            else
                sum(mi * plan.output_col_widths[j] * plan.b_col_ranks[j]
                    for j in run.cols)
            end
            flops += 2.0 * (common + fold_specific + terminal)
        end
    end
    return flops
end

function _tlr_tlr_exact_flops(A::CompressedFTLRMatrix,
                               B::CompressedFTLRMatrix,
                               workspace_bytes)
    LA, LB = logical_operand(A), logical_operand(B)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    budget = min(Int(workspace_bytes), plan.profile.maximum)
    flops = 0.0
    _, qk = grid_size(LA)
    runs = _compressed_ftlr_column_schedule(
        plan, LA, LB, plan.profile, budget)
    for run in runs
        _run_has_execution_work(LA, LB, run, qk) || continue
        width = sum(plan.output_col_widths[j] for j in run.cols)
        for i in run.rows
            mi = length(_compressed_ftlr_axis_range(LA, i, 1))
            common = 0.0
            fold_specific = 0.0
            exact_a_total = 0
            exact_b_cols = zeros(Int, length(plan.output_col_widths))
            for l in 1:qk
                exact_a_total += _compressed_ftlr_rank(LA, i, l)
                for j in run.cols
                    ra = _compressed_ftlr_rank(LA, i, l)
                    rb = _compressed_ftlr_rank(LB, l, j)
                    exact_b_cols[j] += rb
                    (ra == 0 || rb == 0) && continue
                    tk = length(_compressed_ftlr_axis_range(LA, l, 2))
                    nj = length(_compressed_ftlr_axis_range(LB, j, 2))
                    common += tk * ra * rb
                    fold_specific += run.fold === :right ?
                        nj * ra * rb : mi * ra * rb
                end
            end
            terminal = if run.fold === :right
                mi * width * exact_a_total
            else
                sum(mi * plan.output_col_widths[j] * exact_b_cols[j]
                    for j in run.cols)
            end
            flops += 2.0 * (common + fold_specific + terminal)
        end
    end
    return flops
end

end
