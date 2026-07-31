# fold_schedule_tradeoff.jl -- CPU-only, no GPU required.
#
#   julia --project=experiments experiments/fold_schedule_tradeoff.jl
#   UNFUSED_PENALTY=4.2 TRADEOFF_Q=64 julia --project=experiments experiments/fold_schedule_tradeoff.jl
#
# Answers two SEPARATE questions about CompressedFTLR's stage-3 scheduling,
# kept separate deliberately because they depend on different things:
#
#   Q1: is chasing fusion worth it at all?
#       Needs UNFUSED_PENALTY = measured (unfused_time / fused_time) for equal
#       FLOPs -- get this from `PROBE_PHASE=fusion h100_audit_probe.jl` on the
#       target GPU/dtype. Fused and unfused stage-3 GEMMs do the SAME total
#       arithmetic split into different task grains, so their FLOP counts are
#       provably identical: a pure-FLOP model cannot see this axis at all.
#
#   Q2: beyond always picking the single best-fused fold globally, does
#       per-region adaptive peeling (mixing FoldRight/FoldLeft tile-by-tile)
#       buy anything more?
#       This does NOT depend on UNFUSED_PENALTY -- both the "always fused"
#       baseline and the peeling DP assume full fusion, so this gap is purely
#       about how heterogeneous the rank distribution is.
#
# Reuses the ranked-grid generators from h100_audit_probe.jl's `rankgrid`,
# plus one added "block" distribution (alternating dense/sparse rows) meant to
# stress-test whether adaptive peeling has anything to gain.
#
# Simplification: uniform tile size (bm=bn=BM, N=q*BM exactly) -- this script
# is a scheduling-policy comparison, not a tail-tile correctness test.

using NextLA, Printf, Random

const _T    = NextLA.TLRmodule
const Q     = parse(Int, get(ENV, "TRADEOFF_Q",    "32"))
const BM    = parse(Int, get(ENV, "TRADEOFF_BM",   "128"))
const RMAX  = parse(Int, get(ENV, "TRADEOFF_RMAX", "128"))
const PENALTY = parse(Float64, get(ENV, "UNFUSED_PENALTY", "3.0"))

if !haskey(ENV, "UNFUSED_PENALTY")
    println("NOTE: UNFUSED_PENALTY not set -- using placeholder $PENALTY.")
    println("      Measure the real value with `PROBE_PHASE=fusion` in")
    println("      h100_audit_probe.jl on the target GPU/dtype and re-run with")
    println("      UNFUSED_PENALTY=<value>. Q1 below is not trustworthy until then.")
end

# ---------------------------------------------------------------- rank grids
rankgrid(q, rmax; seed=7, dist=:uniform) = begin
    rng = MersenneTwister(seed)
    if dist === :uniform
        [rand(rng, 1:rmax) for _ in 1:q, _ in 1:q]
    elseif dist === :decay          # heavy-tailed: realistic TLR off-diagonal decay
        [clamp(round(Int, rmax * exp(-2.5 * abs(i - j) / q)) + 1, 1, rmax) for i in 1:q, j in 1:q]
    elseif dist === :block          # alternating dense/sparse rows: stresses per-region adaptivity
        [isodd(i) ? rand(rng, rmax÷2:rmax) : rand(rng, 1:max(1, rmax÷8)) for i in 1:q, j in 1:q]
    else
        error("unknown dist $dist")
    end
end

execgrid(raw) = _T._compressed_ftlr_execution_rank.(raw, :q8)

# ------------------------------------------------------- whole-matrix totals
# Matches the codebase's `right_flops[i]`/`left_flops[i]` (schedule.jl), in the
# same MAC units (no factor of 2 -- only ratios matter here, so it cancels).
function whole_matrix_flops(Rexec, R2exec, bm, bn)
    q = size(Rexec, 1); N = q * bn
    sigma = [sum(@view R2exec[k, :]) for k in 1:q]        # sigma_k = sum_j rB_kj
    omega = [bn * sigma[k] for k in 1:q]                  # omega_k = sum_j bn_j*rB_kj
    gamma = [sum(@view R2exec[:, j]) for j in 1:q]        # gamma_j = sum_k rB_kj
    weighted_col_rank = sum(bn .* gamma)
    right_row = zeros(Float64, q)   # FoldRight, always naturally fused
    left_row  = zeros(Float64, q)   # FoldLeft, unfused under today's row-run scheduler
    for i in 1:q
        rho_i = sum(@view Rexec[i, :])
        stage2_r = sum(Rexec[i, k] * omega[k] for k in 1:q)
        right_row[i] = stage2_r + bm * rho_i * N
        pair_i = sum(Rexec[i, k] * sigma[k] for k in 1:q)
        left_row[i] = bm * pair_i + bm * weighted_col_rank
    end
    return right_row, left_row
end

# ------------------------------------------------------------- peeling DP
# Remaining-region invariant: after any sequence of peels, what's left is the
# contiguous submatrix rows i0..q, cols j0..q (a row peel writes row i0 fused
# across cols j0..q; a column peel writes col j0 fused across rows i0..q).
# O(q^2) states; each transition is O(q^2) here (brute-force, not optimized to
# O(1) via 2D prefix sums -- correctness over cleverness for a one-off script,
# and Q is small enough that this is still sub-second).
function row_peel_cost(Rexec, R2exec, i, j0, bm, bn)
    q = size(Rexec, 1)
    rho_i = sum(@view Rexec[i, :])
    stage2 = sum(Rexec[i, k] * R2exec[k, j] * bn for k in 1:q, j in j0:q)
    width_rem = (q - j0 + 1) * bn
    return stage2 + bm * rho_i * width_rem
end

function col_peel_cost(Rexec, R2exec, j, i0, bm, bn)
    q = size(Rexec, 1)
    gamma_j = sum(@view R2exec[:, j])
    stage2 = sum(Rexec[i, k] * R2exec[k, j] * bm for i in i0:q, k in 1:q)
    height_rem = (q - i0 + 1) * bm
    return stage2 + height_rem * gamma_j * bn
end

function peeling_dp(Rexec, R2exec, bm, bn)
    q = size(Rexec, 1)
    cost = zeros(Float64, q + 1, q + 1)   # cost[q+1, :] = cost[:, q+1] = 0 (empty region)
    for i0 in q:-1:1, j0 in q:-1:1
        rp = row_peel_cost(Rexec, R2exec, i0, j0, bm, bn) + cost[i0 + 1, j0]
        cp = col_peel_cost(Rexec, R2exec, j0, i0, bm, bn) + cost[i0, j0 + 1]
        cost[i0, j0] = min(rp, cp)
    end
    return cost[1, 1]
end

# --------------------------------------------------------------------- main
println("Q=$Q BM=$BM RMAX=$RMAX  UNFUSED_PENALTY=$PENALTY")
@printf("\n%-8s %14s %14s %14s %14s %14s %10s %10s\n",
        "dist", "F_right", "F_left", "today", "1+3(fused)", "peeling", "Q1 gap%", "Q2 gap%")

for dist in (:uniform, :decay, :block)
    rawA = rankgrid(Q, RMAX; seed=7,  dist)
    rawB = rankgrid(Q, RMAX; seed=11, dist)   # independent seed: avoids a forced tie
    Rexec, R2exec = execgrid(rawA), execgrid(rawB)

    right_row, left_row = whole_matrix_flops(Rexec, R2exec, BM, BM)
    F_right, F_left = sum(right_row), sum(left_row)

    # today: `_compressed_ftlr_select_fold`'s actual criterion -- compare raw
    # FLOPs to decide, then pay whatever fusion state that fold ACTUALLY gets
    # (FoldRight always fused, FoldLeft always unfused under row-run today).
    today = sum(right_row[i] <= left_row[i] ? right_row[i] : left_row[i] * PENALTY for i in 1:Q)

    # 1+3: ONE global decision for the whole matrix -- run FoldRight-fused (row
    # run) or FoldLeft-fused (column run, not built yet), whichever has lower
    # TOTAL FLOPs. This is NOT a per-row minimum: a single row can't be
    # "FoldLeft-fused" at all -- FoldLeft's fusion benefit only exists when
    # aggregated over a genuine row range (that's what its Tstack_j height
    # comes from), so left_row[i] at single-row granularity is identical
    # whether fused or not. Per-region mixing is exactly what peeling adds.
    best_fused = min(F_right, F_left)

    peeling = peeling_dp(Rexec, R2exec, BM, BM)

    # Sanity checks on the derivation itself, not just the final numbers.
    # "Always row" (never take a column peel) must reproduce F_right exactly;
    # "always column" must reproduce F_left exactly. This is a much stronger
    # check than peeling <= best_fused alone -- it verifies the peel-cost
    # formulas agree with the whole-matrix formulas, not just their ordering.
    always_row = sum(row_peel_cost(Rexec, R2exec, i, 1, BM, BM) for i in 1:Q)
    always_col = sum(col_peel_cost(Rexec, R2exec, j, 1, BM, BM) for j in 1:Q)
    if !isapprox(always_row, F_right; rtol=1e-9)
        println("  !! WARNING: always-row peel ($always_row) != F_right ($F_right) for $dist")
    end
    if !isapprox(always_col, F_left; rtol=1e-9)
        println("  !! WARNING: always-col peel ($always_col) != F_left ($F_left) for $dist")
    end
    if peeling > best_fused + 1e-6
        println("  !! WARNING: peeling > best_fused for $dist -- DP or formulas are wrong")
    end
    if best_fused > today + 1e-6
        println("  !! WARNING: best_fused > today for $dist -- PENALTY < 1 or formulas are wrong")
    end

    gap1 = 100 * (today - best_fused) / today            # value of fixing today + building col-run
    gap2 = 100 * (best_fused - peeling) / best_fused      # additional value of full adaptive peeling

    @printf("%-8s %14.3e %14.3e %14.3e %14.3e %14.3e %9.2f%% %9.2f%%\n",
            dist, F_right, F_left, today, best_fused, peeling, gap1, gap2)
end

println("""

Q1 gap%  = (today - best_fused) / today
           value of (a) teaching the scheduler about fusion instead of comparing
           raw FLOPs blindly, and (b) building FoldLeft-fused-by-column support.
           Scales directly with UNFUSED_PENALTY -- re-run with the H100-measured
           value before trusting this column.

Q2 gap%  = (best_fused - peeling) / best_fused
           additional value of full per-region adaptive peeling BEYOND always
           picking the single best-fused fold for the whole matrix. Independent
           of UNFUSED_PENALTY (both terms assume full fusion) -- this column is
           trustworthy right now. Small on :uniform/:decay would mean peeling's
           scheduler complexity (second profile, mixed-run execution) isn't
           worth it beyond options 1+3; large on realistic distributions would
           justify it. :block is a deliberately adversarial stress case, not a
           claim about real TLR matrices.
""")
