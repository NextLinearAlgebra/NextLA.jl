# plot_run_progress_svg.jl -- poster figure for the DP + column-block scheduler.
#
#   julia --project=experiments experiments/plot_run_progress_svg.jl
#
# Writes experiments/figures/run_progress_<name>.svg, one per scenario.
#
# Unlike plot_schedule_svg.jl (which compares greedy vs the row-run DP), this
# shows ONE schedule -- the production DP scheduler, INCLUDING column blocking
# -- at two consecutive points in execution order: after run i, and after run
# i+1. Every rectangle drawn is a real `RaggedRowRun` returned by
# `_compressed_ftlr_column_schedule` with `COMPRESSED_FTLR_SCHEDULE_POLICY[] =
# :dp`, at a budget chosen (via `_compressed_ftlr_column_floor` /
# `profile.minimum`) to force column blocking -- so some runs are FULL output
# rows and some are PARTIAL rows (narrower than the full grid width), which is
# exactly the new capability this figure exists to show. Nothing here is
# hand-drawn or modelled.

using NextLA, KernelAbstractions, Printf
const _T = NextLA.TLRmodule

const C_RIGHT   = "#0072B2"   # FoldRight
const C_LEFT    = "#E69F00"   # FoldLeft
const C_PENDING = "#E8E8E8"   # not yet scheduled
const C_BAR     = "#999999"
const C_TEXT    = "#222222"
const C_BORDER  = "#1A1A1A"   # standard run border
const C_NEW     = "#D55E00"   # newest-run highlight (Okabe-Ito vermillion)

fold_color(f) = f === :right ? C_RIGHT : C_LEFT
fold_name(f)  = f === :right ? "FoldRight" : "FoldLeft"

function rect(x, y, w, h, fill; op=1.0, stroke="none", sw=0.0)
    fillattr = fill == "none" ? "fill=\"none\"" :
               @sprintf("fill=\"%s\" fill-opacity=\"%.2f\"", fill, op)
    return @sprintf(
        "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" %s stroke=\"%s\" stroke-width=\"%.2f\"/>",
        x, y, w, h, fillattr, stroke, sw)
end

txt(x, y, s; size=13, anchor="start", weight="normal", fill=C_TEXT, style="normal") = @sprintf(
    "<text x=\"%.2f\" y=\"%.2f\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"%d\" font-weight=\"%s\" font-style=\"%s\" fill=\"%s\" text-anchor=\"%s\">%s</text>",
    x, y, size, weight, style, fill, anchor, s)

line(x1, y1, x2, y2; stroke="#FFFFFF", sw=0.5, op=1.0) = @sprintf(
    "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"%s\" stroke-width=\"%.2f\" stroke-opacity=\"%.2f\"/>",
    x1, y1, x2, y2, stroke, sw, op)

"""Pixel rectangle for tile range (rows, cols) inside a panel starting at (x0,y0)."""
tile_rect(x0, y0, cell, rows, cols) = (
    x0 + (first(cols) - 1) * cell, y0 + (first(rows) - 1) * cell,
    length(cols) * cell, length(rows) * cell,
)

"""One panel: marginals + tile grid + the first `nshown` runs colored in,
with `newest` (if given) drawn with the highlight border."""
function panel!(out, x0, y0, cell, qm, qn, runs, nshown, newest, gamma, rho, title, subtitle)
    gh = 30.0
    gmax = maximum(gamma)
    for j in 1:qn
        h = gh * gamma[j] / gmax
        push!(out, rect(x0 + (j - 1) * cell, y0 - 8 - h, cell * 0.8, h, C_BAR; op=0.75))
    end
    push!(out, txt(x0, y0 - 8 - gh - 7, "γ_j"; size=10, fill="#666666"))

    rmax = maximum(rho)
    wr = 34.0
    for i in 1:qm
        w = wr * rho[i] / rmax
        push!(out, rect(x0 - 10 - w, y0 + (i - 1) * cell, w, cell * 0.8, C_BAR; op=0.75))
    end
    push!(out, txt(x0 - 10 - wr, y0 - 6, "ρ_i"; size=10, anchor="start", fill="#666666"))

    # Pending background, drawn once for the whole grid.
    push!(out, rect(x0, y0, qn * cell, qm * cell, C_PENDING; op=1.0))
    for i in 0:qm
        push!(out, line(x0, y0 + i * cell, x0 + qn * cell, y0 + i * cell; stroke="#CCCCCC", sw=0.5))
    end
    for j in 0:qn
        push!(out, line(x0 + j * cell, y0, x0 + j * cell, y0 + qm * cell; stroke="#CCCCCC", sw=0.5))
    end

    for (idx, run) in enumerate(runs[1:nshown])
        x, y, w, h = tile_rect(x0, y0, cell, run.rows, run.cols)
        push!(out, rect(x, y, w, h, fold_color(run.fold); op=0.88))
        for i in 0:length(run.rows)
            push!(out, line(x, y + i * cell, x + w, y + i * cell; stroke="#FFFFFF", sw=0.6, op=0.5))
        end
        for j in 0:length(run.cols)
            push!(out, line(x + j * cell, y, x + j * cell, y + h; stroke="#FFFFFF", sw=0.6, op=0.5))
        end
        highlight = idx == newest
        push!(out, rect(x, y, w, h, "none";
                        stroke=highlight ? C_NEW : C_BORDER, sw=highlight ? 3.2 : 2.0))
        if w >= 20 && h >= 16
            push!(out, txt(x + w / 2, y + h / 2 + 4, string(idx); size=11, anchor="middle",
                           weight="bold", fill=highlight ? C_NEW : "#FFFFFF"))
        end
    end
    push!(out, rect(x0, y0, qn * cell, qm * cell, "none"; stroke="#333333", sw=1.0))
    push!(out, txt(x0, y0 + qm * cell + 20, title; size=14, weight="bold"))
    push!(out, txt(x0, y0 + qm * cell + 37, subtitle; size=12, fill="#555555"))
    return nothing
end

function figure(name, rawA, rawB, bm; budget_frac=0.4, call_cost=0.0)
    q = size(rawA, 1)
    cpu = KernelAbstractions.CPU()
    A = NextLA.CompressedFTLRMatrix(cpu, Float32, q * bm, q * bm, (bm, bm), rawA)
    B = NextLA.CompressedFTLRMatrix(cpu, Float32, q * bm, q * bm, (bm, bm), rawB)
    LA, LB = _T.logical_operand(A, 'N'), _T.logical_operand(B, 'N')
    plan = _T._compressed_ftlr_rank_plan(LA, LB)
    profile = plan.profile

    rowfloor = profile.minimum
    colfloor = _T._compressed_ftlr_column_floor(plan, LA, LB)
    budget = round(Int, colfloor + budget_frac * (rowfloor - colfloor))

    # Stage 2/3 cost is EXACTLY additive per row when the fold doesn't change
    # (see schedule_dp.jl's call-cost caveat), so merging same-fold rows is
    # FLOP-neutral -- at call_cost=0 the DP has no reason to prefer it, and a
    # figure generated that way would show nothing but single-row runs. A
    # nonzero call cost is what actually makes it merge, and is also the
    # honest way to run it: a real deployment calibrates this from measured
    # launch overhead, it doesn't leave it at zero.
    _T.COMPRESSED_FTLR_SCHEDULE_POLICY[] = :dp
    _T.COMPRESSED_FTLR_DP_CALL_COST[] = call_cost
    runs = _T._compressed_ftlr_column_schedule(plan, LA, LB, profile, budget)
    _T.COMPRESSED_FTLR_SCHEDULE_POLICY[] = :greedy   # restore defaults
    _T.COMPRESSED_FTLR_DP_CALL_COST[] = 0.0

    @printf("%s: q=%d  rowfloor=%d  colfloor=%d  budget=%d (%.0f%% of the way)\n",
            name, q, rowfloor, colfloor, budget, 100 * budget_frac)
    @printf("  %d runs, column blocks: %s\n", length(runs),
            string([length(b) for b in unique(r.cols for r in runs)]))
    for (k, r) in enumerate(runs)
        @printf("   run %2d: rows=%-8s cols=%-8s fold=%s\n", k, r.rows, r.cols, r.fold)
    end

    # Frame at the first column-block boundary: shows one full block finishing
    # and the next (differently-sized) block beginning -- the clearest single
    # transition for "full vs partial row" if one exists in this schedule.
    transition = findfirst(k -> runs[k].cols != runs[k - 1].cols, 2:length(runs))
    i = transition === nothing ? max(1, length(runs) ÷ 2) : transition
    @printf("  showing transition: after run %d -> after run %d\n\n", i, i + 1)

    gamma = [sum(_T._compressed_ftlr_execution_rank(LB, k, j) for k in 1:q) for j in 1:q]
    rho   = [sum(_T._compressed_ftlr_execution_rank(LA, i_, k) for k in 1:q) for i_ in 1:q]

    cell = 220.0 / q
    x_rho, w_rho = 56.0, 44.0
    xA = x_rho + w_rho + 10
    side = q * cell
    gapAB = 90.0
    xB = xA + side + gapAB
    y0 = 108.0
    W = xB + side + 40
    H = y0 + side + 100

    out = String[]
    push!(out, @sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">", W, H, W, H))
    push!(out, rect(0, 0, W, H, "#FFFFFF"))
    push!(out, txt(x_rho, 26, "DP schedule, in execution order (q=$q)"; size=16, weight="bold"))
    push!(out, txt(x_rho, 43, "column-blocked runs: widths vary with the workspace budget · numbers are run order"; size=11.5, fill="#555555"))

    panel!(out, xA, y0, cell, q, q, runs, i, nothing, gamma, rho,
          "after run $i", "$i of $(length(runs)) runs complete")
    panel!(out, xB, y0, cell, q, q, runs, i + 1, i + 1, gamma, rho,
          "after run $(i+1)", "run $(i+1) just added (highlighted)")

    ax = xA + side + 14
    push!(out, txt(ax + (gapAB - 28) / 2, y0 + side / 2 + 6, "→"; size=22, anchor="middle", fill="#888888"))

    ly = y0 + side + 62
    lx = x_rho
    push!(out, rect(lx, ly - 10, 14, 14, C_RIGHT; op=0.88)); push!(out, txt(lx + 20, ly, "FoldRight"; size=11))
    lx += 100
    push!(out, rect(lx, ly - 10, 14, 14, C_LEFT; op=0.88)); push!(out, txt(lx + 20, ly, "FoldLeft"; size=11))
    lx += 95
    push!(out, rect(lx, ly - 10, 14, 14, C_PENDING)); push!(out, txt(lx + 20, ly, "not yet scheduled"; size=11))
    lx += 130
    push!(out, rect(lx, ly - 10, 14, 14, "none"; stroke=C_NEW, sw=2.5)); push!(out, txt(lx + 20, ly, "newest run"; size=11))

    push!(out, "</svg>")
    dir = joinpath(@__DIR__, "figures"); isdir(dir) || mkpath(dir)
    path = joinpath(dir, "run_progress_$(name).svg")
    open(path, "w") do io; println(io, join(out, "\n")); end
    println("wrote ", path)
    return path
end

import Random
# `_compressed_ftlr_widest_column_block` always picks the width where the
# single MOST EXPENSIVE remaining row barely fits the budget -- so a smooth
# rank gradient never leaves headroom for OTHER rows at that width to merge.
# Explicit low/high row bands do: cheap rows merge into one run even at a
# width calibrated to the expensive ones, giving both effects at once --
# multi-row runs AND column blocks narrower than the full grid.
function banded_grid(q, bm; seed, bands)
    rng = Random.MersenneTwister(seed)
    ranks = zeros(Int, q, q)
    for (rows, lo, hi) in bands, i in rows
        ranks[i, :] .= rand(rng, lo:hi, q)
    end
    return ranks
end

q, bm = 10, 32   # bm must be large enough that :q8 bucketing (round up to a
                 # multiple of 8) doesn't collapse every band to the same
                 # execution rank -- at bm=8 it did, silently, since cld(r,8)*8
                 # == 8 for any 1 <= r <= 7.
rawA = banded_grid(q, bm; seed=1, bands=[(1:4, 2, 6), (5:7, 12, 16), (8:10, 24, 28)])
rawB = banded_grid(q, bm; seed=2, bands=[(1:10, 10, 14)])
# budget_frac=0.92 gives a [9,1] column-block split -- the closest this
# mechanism gets to "full vs. partial": a genuinely full-width block is only
# possible when nothing else competes for it (a single block spanning all qn
# columns), so there is no schedule with one literally-full-width run
# alongside a narrower one. A 9-of-10 block next to a 1-of-10 sliver is the
# honest visual stand-in for that contrast.
figure("final", rawA, rawB, bm; budget_frac=0.92, call_cost=15000.0)
