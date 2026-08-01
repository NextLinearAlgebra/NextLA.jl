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

using NextLA, KernelAbstractions, Printf, Random

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
    else
        error("unknown dist $dist (use :corner_pair for the adversarial case)")
    end
end

# Adversarial pair for A/B (must be used TOGETHER, not independently like the
# distributions above): stage-2's total FLOPs are provably IDENTICAL between
# FoldRight and FoldLeft whenever bm==bn (verified: bn*sum(A.*B) == bm*sum(A.*B)
# for any grids), so peeling's entire possible benefit lives purely in how
# rho_i (A's row-rank totals) and gamma_j (B's column-rank totals) are
# organized. This pair makes A's TOP rows cheap/uniform-across-k and BOTTOM
# rows expensive; B's LEFT columns expensive/uniform-across-k and RIGHT
# columns cheap -- opposite halves, so no single global fold choice can pick
# up both cheap regions, but peeling (row-peel the cheap top, then col-peel
# the cheap right, leaving only the unavoidable expensive corner) can. A
# hand-computed q=6 toy of exactly this construction gave peeling at 51% of
# either pure global choice -- this is the distribution that number came from.
function corner_pair(q, rmax; seed_a=7, seed_b=11)
    rng_a, rng_b = MersenneTwister(seed_a), MersenneTwister(seed_b)
    cheap, expensive = 1:max(1, rmax ÷ 16), (3 * rmax) ÷ 4:rmax
    top = q ÷ 2
    rawA = [(i <= top ? rand(rng_a, cheap) : rand(rng_a, expensive)) for i in 1:q, _ in 1:q]
    rawB = [(j <= top ? rand(rng_b, expensive) : rand(rng_b, cheap)) for _ in 1:q, j in 1:q]
    return rawA, rawB
end

execgrid(raw) = _T._compressed_ftlr_execution_rank.(raw, :q8)

# --------------------------------------------------- per-tile optimal bound
# Only meaningful once UNFUSED_PENALTY is confirmed ~1 (measured on H100
# FP16, production scale: 1.00x, 0.99x, 1.00x across rho=256/1024/2048 --
# fusion buys nothing measurable). If unfused genuinely costs nothing, every
# output tile (i,j) is independently computable via EITHER arithmetic route
# (FoldRight-style: bm*rho_i*bn_j, or FoldLeft-style: bm*gamma_j*bn_j) with no
# fusion or contiguity requirement at all -- so the true optimum is a trivial
# per-tile min, not a row/region-level decision. Stage-2's cost is IDENTICAL
# per tile (not just in aggregate) whenever bm==bn: both routes reduce to
# bm*sum_k(A[i,k]*B[k,j]) for that tile, so it drops out of every comparison
# and only the stage-3 term (rho_i vs gamma_j) decides.
function tile_optimal_flops(Rexec, R2exec, bm, bn)
    q = size(Rexec, 1)
    rho   = [sum(@view Rexec[i, :]) for i in 1:q]
    gamma = [sum(@view R2exec[:, j]) for j in 1:q]
    stage2_total = sum(bm * Rexec[i, k] * R2exec[k, j] for i in 1:q, k in 1:q, j in 1:q)
    stage3_total = sum(bm * bn * min(rho[i], gamma[j]) for i in 1:q, j in 1:q)
    return stage2_total + stage3_total
end

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

# ----------------------------------------------- REAL scheduler (not a model)
# The `today` baseline above assumes independent PER-ROW fold decisions. The
# actual scheduler (`_compressed_ftlr_row_runs`, schedule.jl) greedily EXTENDS
# a run across as many rows as the workspace budget allows, then picks ONE
# fold for the whole accumulated range -- coarser than per-row, and coarser
# still with the generous workspace typically used (gemm_maximum_workspace_bytes).
# This builds REAL CompressedFTLRMatrix objects and drives the production
# `_compressed_ftlr_rank_plan` + `_compressed_ftlr_row_runs` directly, so the
# reported cost is what the shipped scheduler actually does -- not a model of it.
function real_scheduler_flops(rawA, rawB, bm; policy=:q8, budget_kind=:maximum)
    cpu = KernelAbstractions.CPU()
    q = size(rawA, 1)
    A = NextLA.CompressedFTLRMatrix(cpu, Float32, q * bm, q * bm, (bm, bm), rawA;
                                    execution_rank_policy=policy)
    B = NextLA.CompressedFTLRMatrix(cpu, Float32, q * bm, q * bm, (bm, bm), rawB;
                                    execution_rank_policy=policy)
    LA, LB = _T.logical_operand(A, 'N'), _T.logical_operand(B, 'N')
    plan = _T._compressed_ftlr_rank_plan(LA, LB)
    p = plan.profile
    budget = budget_kind === :maximum ? p.maximum : p.minimum
    runs = _T._compressed_ftlr_row_runs(p, budget)
    total = sum(runs) do run
        rows = run.rows
        run.fold === :right ? sum(@view p.right_flops[rows]) : sum(@view p.left_flops[rows])
    end
    sizes = [length(run.rows) for run in runs]
    folds = [run.fold for run in runs]
    return total, length(runs), sizes, folds
end

# --------------------------------------------------------------------- main
println("Q=$Q BM=$BM RMAX=$RMAX  UNFUSED_PENALTY=$PENALTY")
@printf("\n%-8s %14s %14s %14s %14s %14s %14s %10s %10s\n",
        "dist", "F_right", "F_left", "today", "1+3(fused)", "peeling", "tile-opt", "Q1 gap%", "Q2 gap%")

for dist in (:uniform, :decay, :corner)
    rawA, rawB = if dist === :corner
        corner_pair(Q, RMAX; seed_a=7, seed_b=11)
    else
        rankgrid(Q, RMAX; seed=7, dist), rankgrid(Q, RMAX; seed=11, dist)   # independent seed: avoids a forced tie
    end
    Rexec, R2exec = execgrid(rawA), execgrid(rawB)

    right_row, left_row = whole_matrix_flops(Rexec, R2exec, BM, BM)
    F_right, F_left = sum(right_row), sum(left_row)

    # today: `_compressed_ftlr_select_fold`'s actual criterion -- compare raw
    # FLOPs to decide, then pay whatever fusion state that fold ACTUALLY gets
    # (FoldRight always fused, FoldLeft always unfused under row-run today).
    # NOTE: when PENALTY < 1 (unfused genuinely as fast or faster than fused,
    # as measured), `today` can legitimately beat `best_fused` -- it is making
    # an INDEPENDENT per-row choice with no fusion cost to worry about, which
    # is strictly more freedom than "pick one fold for the whole matrix" once
    # fusion is free. That is not a bug; it is the reason `tile_optimal` below
    # exists, since `today` still isn't the tightest achievable bound either.
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

    # The tightest bound: IF unfused is genuinely free (measured ~1x), no
    # scheduling constraint is needed at all -- every tile can independently
    # pick its cheaper arithmetic route. See tile_optimal_flops above.
    tile_opt = tile_optimal_flops(Rexec, R2exec, BM, BM)

    # REAL scheduler cost, driving the actual shipped code (not a per-row model).
    real_max, nruns_max, sizes_max, _ = real_scheduler_flops(rawA, rawB, BM; budget_kind=:maximum)
    real_min, nruns_min, sizes_min, _ = real_scheduler_flops(rawA, rawB, BM; budget_kind=:minimum)
    gap_real_max = 100 * (real_max - tile_opt) / real_max
    gap_real_min = 100 * (real_min - tile_opt) / real_min
    @printf("  [real scheduler] generous budget: %d run(s), sizes %s, cost=%.3e, gap-to-tile-opt=%.2f%%\n",
            nruns_max, string(extrema(sizes_max)), real_max, gap_real_max)
    @printf("  [real scheduler] tight budget:    %d run(s), sizes %s, cost=%.3e, gap-to-tile-opt=%.2f%%\n",
            nruns_min, string(extrema(sizes_min)), real_min, gap_real_min)

    # Sanity checks on the derivation itself, not just the final numbers.
    # "Always row" (never take a column peel) must reproduce F_right exactly;
    # "always column" must reproduce F_left exactly. This is a much stronger
    # check than peeling <= best_fused alone -- it verifies the peel-cost
    # formulas agree with the whole-matrix formulas, not just their ordering.
    always_row = sum(row_peel_cost(Rexec, R2exec, i, 1, BM, BM) for i in 1:Q)
    always_col = sum(col_peel_cost(Rexec, R2exec, j, 1, BM, BM) for j in 1:Q)
    if tile_opt > peeling + 1e-6
        println("  !! WARNING: tile_opt > peeling for $dist -- tile_optimal_flops is wrong")
    end
    if !isapprox(always_row, F_right; rtol=1e-9)
        println("  !! WARNING: always-row peel ($always_row) != F_right ($F_right) for $dist")
    end
    if !isapprox(always_col, F_left; rtol=1e-9)
        println("  !! WARNING: always-col peel ($always_col) != F_left ($F_left) for $dist")
    end
    if peeling > best_fused + 1e-6
        println("  !! WARNING: peeling > best_fused for $dist -- DP or formulas are wrong")
    end
    if today < tile_opt - 1e-6
        println("  !! WARNING: today < tile_opt for $dist -- tile_opt is not a valid lower bound")
    end
    if real_max < tile_opt - 1e-6 || real_min < tile_opt - 1e-6
        println("  !! WARNING: real scheduler beat tile_opt for $dist -- tile_opt is not a valid lower bound")
    end
    if real_min > real_max + 1e-6
        println("  !! WARNING: tighter budget ($real_min) beat generous budget ($real_max) for $dist -- unexpected, check greedy run construction")
    end
    # NOTE: best_fused > today is EXPECTED whenever PENALTY < 1 -- see the
    # comment above `today`'s definition. It is not checked as an error here.

    gap1 = 100 * (today - best_fused) / today            # value of fixing today + building col-run
    gap2 = 100 * (best_fused - peeling) / best_fused      # additional value of full adaptive peeling
    gap3 = 100 * (today - tile_opt) / today               # remaining gap between today and the true optimum

    @printf("%-8s %14.3e %14.3e %14.3e %14.3e %14.3e %14.3e %9.2f%% %9.2f%%  (gap3=%.2f%%)\n",
            dist, F_right, F_left, today, best_fused, peeling, tile_opt, gap1, gap2, gap3)
end

println("""

Q1 gap%  = (today - best_fused) / today
           value of (a) teaching the scheduler about fusion instead of comparing
           raw FLOPs blindly, and (b) building FoldLeft-fused-by-column support.
           Scales directly with UNFUSED_PENALTY. NEGATIVE means `today` (which
           already decides per row, independently) beats a single global
           choice -- expected once PENALTY is measured near 1, since per-row
           independence becomes free. Do not build col-run support for this
           reason alone if PENALTY ~1 on your target hardware/dtype.

Q2 gap%  = (best_fused - peeling) / best_fused
           additional value of full per-region adaptive peeling BEYOND always
           picking the single best-fused fold for the whole matrix. Independent
           of UNFUSED_PENALTY (both terms assume full fusion) -- trustworthy
           regardless of the fusion measurement. Small on :uniform/:decay means
           peeling's complexity (second profile, mixed-run execution) isn't
           worth it beyond a global choice; :corner is a deliberately
           adversarial upper bound, not a claim about real TLR matrices.

gap3 (printed inline) = (today - tile_opt) / today
           ONLY meaningful once PENALTY is confirmed ~1 (measured H100 FP16,
           production scale: 1.00x/0.99x/1.00x across rho=256/1024/2048).
           If unfused is genuinely free, per-tile scheduling with NO fusion or
           contiguity constraint dominates every other option, including
           peeling -- tile_opt <= peeling always. A small gap3 means `today`'s
           existing simple per-row-run FLOP comparison is *already* close to
           the true optimum and nothing here is worth building; a large gap3
           would motivate per-tile (not per-row, not per-region) scheduling
           instead of either 1+3 or peeling.
""")
