# =============================================================================
# TEMPORARY / BENCHMARK-ONLY -- delete this whole file to revert to greedy.
# =============================================================================
# An optimal replacement for `_compressed_ftlr_row_runs` (schedule.jl), kept in
# a separate file so the two policies can be benchmarked head-to-head and the
# loser removed cleanly.
#
# To remove: delete this file, drop its `include` from gemm/dense_result.jl,
# and change the single `_compressed_ftlr_schedule` call in analysis.jl back to
# `_compressed_ftlr_row_runs`. Nothing else references anything defined here.
#
# WHY: the shipped greedy extends a run until the workspace stops it, then
# commits. That is provably optimal for minimising the NUMBER of runs, but not
# their COST -- it never reconsiders a boundary, so rows whose cheaper fold
# differs get merged and forced onto one global choice. Under a generous
# budget it swallows the entire grid into a single run and picks one fold for
# every row. Measured with experiments/fold_schedule_tradeoff.jl (Q=32, q8):
#
#     distribution   greedy -> DP     runs (greedy -> DP)
#     uniform          -1.59%           1 -> 16
#     decay            -3.03%           1 ->  3
#     corner           -29.36%          1 ->  2      (adversarial construction)
#
# CAVEAT, and the reason this is not on by default: the cost model is pure
# arithmetic and cannot see per-grouped-call overhead. At `call_cost = 0` the
# DP splits freely to chase small FLOP wins -- on :uniform it takes 16 runs to
# save 1.59%, i.e. 16x the grouped-call submissions and arena resets, which can
# easily be a net LOSS in wall clock. `call_cost` prices one extra run in the
# same MAC units as the FLOP model; raising it monotonically consolidates runs
# (measured: at 1e8 the :uniform schedule collapses from 16 runs to 2). Sweep
# it against real timings before trusting the FLOP ranking.

"""
    _compressed_ftlr_row_runs_dp(profile, budget; call_cost=0.0)

Cost-optimal partition of the output rows into contiguous runs, under the same
byte budget and the same per-range fold criterion (`_compressed_ftlr_select_fold`)
the greedy scheduler uses -- so the two differ only in where they place run
boundaries, never in how a given run is costed or executed.

`DP[j] = min_i { DP[i-1] + cost(i:j) }` over feasible ranges, with `cost` the
cheaper feasible fold's FLOPs plus `call_cost`. Optimal substructure is the
standard exchange argument: an optimal partition's prefix must itself be
optimal, else substituting a cheaper prefix would beat it. Since every last-run
start is tried against the true optimal prefix cost, the result is the global
minimum over this schedule class (contiguous row ranges, one fold each) -- which
is exactly the class the executor can run without extra copies. It is NOT a
claim of optimality over unrestricted schedules; `tile_optimal_flops` in
experiments/fold_schedule_tradeoff.jl bounds that larger space.

O(q²) range queries, each O(1) off the profile's prefix sums. Cannot return a
worse schedule than greedy: greedy's own partition is one of the candidates.
"""
function _compressed_ftlr_row_runs_dp(profile::RaggedWorkspaceProfile, budget::Int;
                                      call_cost::Float64=0.0)
    budget >= profile.minimum || throw(ArgumentError(
        "workspace has $budget bytes; at least $(profile.minimum) bytes are required"))
    q = length(profile.row_bytes)
    q == 0 && return RaggedRowRun[]
    # `Base.` qualified: KernelAbstractions exports backend-dispatched `zeros`/`fill`.
    dp = Base.fill(Inf, q + 1)     # dp[j+1] = optimal cost of scheduling rows 1..j
    back = Base.zeros(Int, q + 1)
    dp[1] = 0.0
    @inbounds for j in 1:q
        # Widening a range only adds non-negative bytes, so once i:j does not
        # fit, no smaller i can either -- stop instead of scanning to 1.
        for i in j:-1:1
            fold = _compressed_ftlr_select_fold(profile, i:j, budget)
            fold === nothing && break
            isinf(dp[i]) && continue
            prefix = fold === :right ? profile.right_flop_prefix : profile.left_flop_prefix
            value = dp[i] + _compressed_ftlr_range_total(prefix, i:j) + call_cost
            if value < dp[j + 1]
                dp[j + 1] = value
                back[j + 1] = i
            end
        end
    end
    # `budget >= profile.minimum` guarantees every single row fits under at
    # least one fold, so a covering partition always exists.
    isinf(dp[q + 1]) && throw(ArgumentError(
        "workspace cannot schedule the CompressedFTLR row grid"))
    runs = RaggedRowRun[]
    j = q
    while j >= 1
        i = back[j + 1]
        pushfirst!(runs, RaggedRowRun(i:j, profile.columns,
                                      _compressed_ftlr_select_fold(profile, i:j, budget)))
        j = i - 1
    end
    return runs
end

"""Which row-run scheduler the dense-output CompressedFTLR path uses:
`:greedy` (shipped, default) or `:dp` (`_compressed_ftlr_row_runs_dp`).
Set for benchmarking, e.g. `NextLA.TLRmodule.COMPRESSED_FTLR_SCHEDULE_POLICY[] = :dp`."""
const COMPRESSED_FTLR_SCHEDULE_POLICY = Ref{Symbol}(:greedy)

"""Per-run FLOP-equivalent penalty used by the `:dp` policy -- see the caveat
at the top of this file. Ignored by `:greedy`."""
const COMPRESSED_FTLR_DP_CALL_COST = Ref{Float64}(0.0)

"""Single dispatch point for row-run scheduling; the only production caller is
`analyze_compressed_gemm`."""
function _compressed_ftlr_schedule(profile::RaggedWorkspaceProfile, budget::Int)
    policy = COMPRESSED_FTLR_SCHEDULE_POLICY[]
    policy === :greedy && return _compressed_ftlr_row_runs(profile, budget)
    policy === :dp && return _compressed_ftlr_row_runs_dp(
        profile, budget; call_cost=COMPRESSED_FTLR_DP_CALL_COST[])
    throw(ArgumentError("unknown CompressedFTLR schedule policy $policy (use :greedy or :dp)"))
end
