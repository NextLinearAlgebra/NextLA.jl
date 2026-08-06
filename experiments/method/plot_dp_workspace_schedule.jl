# Draw one dynamic-programming schedule at two consecutive points in execution
# order (after work unit i and after work unit i+1).
#
#   julia --project=experiments experiments/method/plot_dp_workspace_schedule.jl
#
# Writes experiments/figures/method/dp_workspace_schedule.svg.
#
# The illustration uses full-width row runs so every bracket shares one output
# edge. Every rectangle is a real `RaggedRowRun` returned by the DP scheduler.

using NextLA, KernelAbstractions, Printf
const _T = NextLA.TLRmodule

const C_RIGHT  = "#0072B2"
const C_LEFT   = "#E69F00"
const C_BAR    = "#999999"
const C_TEXT   = "#222222"
const C_PEND   = "#EDEDED"
const C_NEW    = "#D55E00"

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
    "<text x=\"%.2f\" y=\"%.2f\" font-family=\"DejaVu Sans,Helvetica,Arial,sans-serif\" font-size=\"%d\" font-weight=\"%s\" font-style=\"%s\" fill=\"%s\" text-anchor=\"%s\">%s</text>",
    x, y, size, weight, style, fill, anchor, s)

"""SVG text with an italic Greek symbol and a true lowered subscript."""
mathlabel(x, y, symbol, subscript; size=12, anchor="start", fill="#666666") = @sprintf(
    "<text x=\"%.2f\" y=\"%.2f\" font-family=\"DejaVu Sans,Helvetica,Arial,sans-serif\" font-size=\"%d\" fill=\"%s\" text-anchor=\"%s\"><tspan font-style=\"italic\">%s</tspan><tspan baseline-shift=\"sub\" font-size=\"%d\">%s</tspan></text>",
    x, y, size, fill, anchor, symbol, round(Int, 0.72 * size), subscript)

line(x1, y1, x2, y2; stroke=C_TEXT, sw=0.5, op=1.0) = @sprintf(
    "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"%s\" stroke-width=\"%.2f\" stroke-opacity=\"%.2f\"/>",
    x1, y1, x2, y2, stroke, sw, op)

"""One panel, schedule_uniform.svg style: full-width bands, grey below the
last completed run, thick perimeter per run, and a bracket + fold label to the
right of every run tall enough to hold one. `newest` (if given) is drawn with
the highlight border and label color instead of the standard ones."""
function panel!(out, x0, y0, side, q, runs, nshown, newest, rho, gamma,
                title, subtitle)
    cell = side / q

    # A row-rank marginal, aligned one bar per output tile row.
    rw = 46.0
    rmax = max(maximum(rho), 1)
    for i in 1:q
        w = rw * rho[i] / rmax
        push!(out, rect(x0 - 10 - w, y0 + (i - 1) * cell,
                        w, cell * 0.82, C_BAR; op=0.75))
    end
    push!(out, mathlabel(x0 - 10 - rw, y0 - 7, "ρ", "i"; size=12))

    # B column-rank marginal, aligned one bar per output tile column.
    gh = 34.0
    gmax = max(maximum(gamma), 1)
    for j in 1:q
        h = gh * gamma[j] / gmax
        push!(out, rect(x0 + (j - 1) * cell, y0 - 8 - h, cell * 0.82, h, C_BAR; op=0.75))
    end
    push!(out, mathlabel(x0, y0 - 8 - gh - 6, "γ", "j"; size=12))

    # Grey background for rows not yet reached, tile grid over everything.
    push!(out, rect(x0, y0, side, side, C_PEND))
    for i in 0:q
        push!(out, line(x0, y0 + i * cell, x0 + side, y0 + i * cell; stroke="#CCCCCC", sw=0.5))
    end
    for j in 0:q
        push!(out, line(x0 + j * cell, y0, x0 + j * cell, y0 + side; stroke="#CCCCCC", sw=0.5))
    end

    for (idx, run) in enumerate(runs[1:nshown])
        yy = y0 + (first(run.rows) - 1) * cell
        hh = length(run.rows) * cell
        push!(out, rect(x0, yy, side, hh, fold_color(run.fold); op=0.85))
        for i in 0:length(run.rows)
            push!(out, line(x0, yy + i * cell, x0 + side, yy + i * cell; stroke="#FFFFFF", sw=0.5, op=0.35))
        end
        for j in 0:q
            push!(out, line(x0 + j * cell, yy, x0 + j * cell, yy + hh; stroke="#FFFFFF", sw=0.5, op=0.35))
        end
        highlight = idx == newest
        # Thick perimeter around the run -- this is what lets two adjacent
        # same-fold runs read as distinct even though their fill is identical.
        push!(out, rect(x0, yy, side, hh, "none";
                        stroke=highlight ? C_NEW : "#333333", sw=highlight ? 3.4 : 2.6))

        bx = x0 + side + 6
        bracket_color = highlight ? C_NEW : "#333333"
        push!(out, line(bx, yy + 1.5, bx, yy + hh - 1.5; stroke=bracket_color, sw=1.5))
        push!(out, line(bx, yy + 1.5, bx + 4, yy + 1.5; stroke=bracket_color, sw=1.5))
        push!(out, line(bx, yy + hh - 1.5, bx + 4, yy + hh - 1.5; stroke=bracket_color, sw=1.5))
        if hh >= 15
            label = fold_name(run.fold) * (highlight ? "  (new)" : "")
            push!(out, txt(bx + 8, yy + hh / 2 + 4, label; size=11,
                           fill=highlight ? C_NEW : fold_color(run.fold), weight="bold"))
        end
    end
    push!(out, rect(x0, y0, side, side, "none"; stroke="#333333", sw=1.0))
    push!(out, txt(x0, y0 + side + 22, title; size=14, weight="bold"))
    isempty(subtitle) ||
        push!(out, txt(x0, y0 + side + 39, subtitle; size=12, fill="#555555"))
    return nothing
end

function figure(name, rawA, rawB, bm; budget_frac=0.5, call_cost=0.0)
    q = size(rawA, 1)
    cpu = KernelAbstractions.CPU()
    A = NextLA.CompressedFTLRMatrix(cpu, Float32, q * bm, q * bm, (bm, bm), rawA;
                                   execution_rank_policy=:q8)
    B = NextLA.CompressedFTLRMatrix(cpu, Float32, q * bm, q * bm, (bm, bm), rawB;
                                   execution_rank_policy=:q8)
    LA, LB = _T.logical_operand(A, 'N'), _T.logical_operand(B, 'N')
    plan = _T._compressed_ftlr_rank_plan(LA, LB)
    profile = plan.profile

    budget = round(Int, profile.minimum + budget_frac * (profile.maximum - profile.minimum))
    runs = _T._compressed_ftlr_row_runs_dp(profile, budget; call_cost=call_cost)

    @printf("%s: q=%d  min=%d  max=%d  budget=%d (%.0f%% of the way)\n",
            name, q, profile.minimum, profile.maximum, budget, 100 * budget_frac)
    @printf("  %d runs\n", length(runs))
    for (k, r) in enumerate(runs)
        @printf("   run %2d: rows=%-8s fold=%s\n", k, r.rows, r.fold)
    end
    length(runs) >= 2 || error("the progress figure needs at least two DP runs")
    # Show a transition near the middle, after enough fine-grained runs have
    # accumulated to make the partition variety visible. Prefer a run at least
    # three rows tall so the newly-added band remains legible at poster scale.
    minimum_prefix = min(6, length(runs) - 1)
    target = min(9, length(runs))
    candidates = [k for k in (minimum_prefix + 1):length(runs)
                  if length(runs[k].rows) >= 3]
    new_run = isempty(candidates) ? clamp(target, minimum_prefix + 1, length(runs)) :
              first(sort(candidates; by=k -> (abs(k - target), -length(runs[k].rows))))
    i = new_run - 1
    @printf("  showing transition: before run %d -> after run %d\n\n", new_run, new_run)

    gamma = [sum(_T._compressed_ftlr_execution_rank(LB, k, j) for k in 1:q) for j in 1:q]
    rho = [sum(_T._compressed_ftlr_execution_rank(LA, i_, k) for k in 1:q) for i_ in 1:q]

    side = 250.0
    xA = 128.0
    gapAB = 210.0   # room for full-length brackets
    xB = xA + side + gapAB
    y0 = 132.0
    W = xB + side + 130
    H = y0 + side + 80

    out = String[]
    push!(out, @sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">", W, H, W, H))
    push!(out, rect(0, 0, W, H, "#FFFFFF"))
    push!(out, txt(62, 30, "Dynamic programming schedules runs to minimize total FLOPs under a workspace budget";
                   size=17, weight="bold"))
    #push!(out, txt(62, 47,
    #               "Run boundaries are set by solving the whole matrix at once.";
    #               size=12, fill="#555555"))
    push!(out, txt(62, 63,
                   "Each run's fold (Right/Left) is then chosen to minimize FLOPs, using the row/column rank totals shown alongside and the tile size";
                   size=12, fill="#555555"))

    panel!(out, xA, y0, side, q, runs, i, nothing, rho, gamma,
           "Work unit $i of $(length(runs))", "")
    panel!(out, xB, y0, side, q, runs, new_run, new_run, rho, gamma,
           "Work unit $new_run of $(length(runs))", "")

    push!(out, txt(xA + side + gapAB / 2, y0 + side / 2 + 6, "→"; size=22, anchor="middle", fill="#888888"))

    dir = normpath(joinpath(@__DIR__, "..", "figures", "method"))
    isdir(dir) || mkpath(dir)
    push!(out, "</svg>")
    path = joinpath(dir, "dp_workspace_schedule.svg")
    open(path, "w") do io; println(io, join(out, "\n")); end
    println("wrote ", path)
    return path
end

import Random
uniform_grid(q, rmax; seed) = begin
    rng = Random.MersenneTwister(seed)
    [rand(rng, 1:rmax) for _ in 1:q, _ in 1:q]
end

# Start from reproducible seeded ranks and use q8 execution-rank bucketing.
# This is a deliberately legible rank profile: groups
# of low-rho rows are separated by high-rho rows, while B's gamma marginal rises
# from low to high. The ranks are fed back through the real matrix constructor,
# cost model, and DP scheduler; the run outlines are not hand drawn.
q, bm, rmax = 32, 128, 128
rawA = uniform_grid(q, rmax; seed=7)
rawB = uniform_grid(q, rmax; seed=11)
row_order = [25, 27, 32, 6, 21, 22, 8, 10, 24, 30, 11, 13, 15, 17, 18, 5,
             20, 26, 29, 9, 16, 1, 2, 3, 4, 7, 12, 14, 19, 23, 28, 31]
column_order = sortperm(vec(sum(rawB; dims=1)))
rawA = rawA[row_order, :]
rawB = rawB[:, column_order]

# Increase the contrast without changing the qualitative seeded pattern. The
# low-rank groups are the rows whose local arithmetic favors FoldRight; their
# reduced rho lets the fixed budget pack visibly taller runs. A smooth column
# scale exposes gamma's contribution to the FoldLeft side of the cost model.
low_rank_source_rows = Set([6, 8, 9, 10, 11, 15, 16, 17, 18,
                            20, 22, 25, 26, 27, 29, 30, 32])
for i in 1:q
    row_order[i] in low_rank_source_rows || continue
    rawA[i, :] .= max.(1, round.(Int, 0.50 .* rawA[i, :]))
end
for j in 1:q
    scale = 0.35 + 0.65 * (j - 1) / (q - 1)
    rawB[:, j] .= max.(1, round.(Int, scale .* rawB[:, j]))
end
figure("final", rawA, rawB, bm; budget_frac=0.10, call_cost=1.0)
