# plot_schedule_svg.jl -- poster figure for the row-run DP scheduler.
#
#   julia --project=experiments experiments/plot_schedule_svg.jl
#   TRADEOFF_Q=32 julia --project=experiments experiments/plot_schedule_svg.jl
#
# Writes experiments/figures/schedule_<dist>.svg, one per distribution.
#
# Every number drawn -- the rank marginals, the run boundaries, the fold each
# run picks, the costs -- comes from the SAME production cost model the shipped
# scheduler uses (`_compressed_ftlr_rank_plan` + `_compressed_ftlr_row_runs`)
# and from `rowrun_dp` in fold_schedule_tradeoff.jl. Nothing here is drawn by
# hand, so the figure cannot drift away from what the code actually does.
#
# The story the figure tells: greedy extends a run until the workspace stops it,
# which under a generous budget swallows the whole matrix and forces ONE fold on
# every row. The DP cuts the row axis exactly where the cheaper fold flips, which
# the rho/gamma marginals make visible.
#
# No plotting dependency -- raw SVG, which is also the most poster-friendly
# output (vector, and editable in Inkscape/Illustrator if you want to restyle).

ENV["TRADEOFF_LIB"] = "1"
include(joinpath(@__DIR__, "fold_schedule_tradeoff.jl"))

using Printf

# Okabe-Ito, matching plot_poster.py's palette.
const C_RIGHT = "#0072B2"   # FoldRight
const C_LEFT  = "#E69F00"   # FoldLeft
const C_BAR   = "#999999"
const C_TEXT  = "#222222"
const C_GRID  = "#FFFFFF"

fold_color(f) = f === :right ? C_RIGHT : C_LEFT
fold_name(f)  = f === :right ? "FoldRight" : "FoldLeft"

# `fill="none"` is emitted as a bare attribute with no fill-opacity: some
# renderers (ImageMagick's built-in MSVG among them) treat "none" plus an
# explicit fill-opacity as opaque black, which would paint the outline rect
# over the whole panel.
function rect(x, y, w, h, fill; op=1.0, stroke="none", sw=0)
    fillattr = fill == "none" ? "fill=\"none\"" :
               @sprintf("fill=\"%s\" fill-opacity=\"%.2f\"", fill, op)
    return @sprintf(
        "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" %s stroke=\"%s\" stroke-width=\"%.2f\"/>",
        x, y, w, h, fillattr, stroke, sw)
end

txt(x, y, s; size=13, anchor="start", weight="normal", fill=C_TEXT, style="normal") = @sprintf(
    "<text x=\"%.2f\" y=\"%.2f\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"%d\" font-weight=\"%s\" font-style=\"%s\" fill=\"%s\" text-anchor=\"%s\">%s</text>",
    x, y, size, weight, style, fill, anchor, s)

line(x1, y1, x2, y2; stroke=C_GRID, sw=0.5, op=1.0) = @sprintf(
    "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"%s\" stroke-width=\"%.2f\" stroke-opacity=\"%.2f\"/>",
    x1, y1, x2, y2, stroke, sw, op)

"""One schedule panel: the C tile grid banded by run, with a run bracket per
band. `runs` is a vector of (rowrange, fold)."""
function panel!(out, x0, y0, side, q, runs, gamma, title, subtitle)
    cell = side / q
    # gamma marginal above the panel (B's column rank -- what FoldLeft pays)
    gmax = maximum(gamma)
    gh = 34.0
    for j in 1:q
        h = gh * gamma[j] / gmax
        push!(out, rect(x0 + (j - 1) * cell, y0 - 8 - h, cell * 0.82, h, C_BAR; op=0.75))
    end
    push!(out, txt(x0, y0 - 10 - gh - 7, "γ_j  (B column rank)"; size=11, fill="#666666"))

    # run bands
    for (rows, fold) in runs
        yy = y0 + (first(rows) - 1) * cell
        hh = length(rows) * cell
        push!(out, rect(x0, yy, side, hh, fold_color(fold); op=0.85))
    end
    # tile grid on top of the bands
    for i in 0:q
        push!(out, line(x0, y0 + i * cell, x0 + side, y0 + i * cell; op=0.35))
        push!(out, line(x0 + i * cell, y0, x0 + i * cell, y0 + side; op=0.35))
    end
    push!(out, rect(x0, y0, side, side, "none"; stroke="#333333", sw=1.0))

    # run brackets + fold labels on the right edge
    for (rows, fold) in runs
        yy = y0 + (first(rows) - 1) * cell
        hh = length(rows) * cell
        bx = x0 + side + 6
        push!(out, line(bx, yy + 1.5, bx, yy + hh - 1.5; stroke="#333333", sw=1.5, op=1.0))
        push!(out, line(bx, yy + 1.5, bx + 4, yy + 1.5; stroke="#333333", sw=1.5))
        push!(out, line(bx, yy + hh - 1.5, bx + 4, yy + hh - 1.5; stroke="#333333", sw=1.5))
        if hh >= 15   # only label bands tall enough to read
            push!(out, txt(bx + 8, yy + hh / 2 + 4, fold_name(fold); size=11,
                           fill=fold_color(fold), weight="bold"))
        end
    end
    push!(out, txt(x0, y0 + side + 22, title; size=14, weight="bold"))
    push!(out, txt(x0, y0 + side + 39, subtitle; size=12, fill="#555555"))
    return nothing
end

function figure(dist; q=Q, bm=BM, rmax=RMAX)
    rawA, rawB = dist === :corner ? corner_pair(q, rmax; seed_a=7, seed_b=11) :
                 (rankgrid(q, rmax; seed=7, dist), rankgrid(q, rmax; seed=11, dist))
    Rexec, R2exec = execgrid(rawA), execgrid(rawB)
    rho   = [sum(@view Rexec[i, :]) for i in 1:q]
    gamma = [sum(@view R2exec[:, j]) for j in 1:q]
    prof  = production_profile(rawA, rawB, bm)
    budget = prof.maximum

    gruns_raw = _T._compressed_ftlr_row_runs(prof, budget)
    gruns = [(r.rows, r.fold) for r in gruns_raw]
    gcost = sum(r -> r.fold === :right ? sum(@view prof.right_flops[r.rows]) :
                                         sum(@view prof.left_flops[r.rows]), gruns_raw)
    dcost, druns = rowrun_dp(prof, budget)
    delta = 100 * (dcost - gcost) / gcost

    side = 250.0
    cell = side / q
    x_rho, w_rho = 62.0, 52.0
    xA = x_rho + w_rho + 14
    xB = xA + side + 120
    y0 = 122.0   # leaves room for the title block above the gamma marginal
    W  = xB + side + 118
    H  = y0 + side + 78

    out = String[]
    push!(out, @sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">", W, H, W, H))
    push!(out, rect(0, 0, W, H, "#FFFFFF"))
    push!(out, txt(x_rho, 30, "Row-run scheduling: greedy vs optimal DP"; size=17, weight="bold"))
    push!(out, txt(x_rho, 49, "output tile grid C (q=$q), banded by the fold each run selects"; size=12, fill="#555555"))

    # rho marginal (A's row rank -- what FoldRight pays), shared by both panels
    rmaxv = maximum(rho)
    for i in 1:q
        w = w_rho * rho[i] / rmaxv
        push!(out, rect(x_rho + w_rho - w, y0 + (i - 1) * cell, w, cell * 0.82, C_BAR; op=0.75))
    end
    push!(out, txt(x_rho + w_rho, y0 - 20, "ρ_i"; size=11, anchor="end", fill="#666666"))
    push!(out, txt(x_rho + w_rho, y0 - 8, "(A row rank)"; size=9, anchor="end", fill="#888888"))

    panel!(out, xA, y0, side, q, gruns, gamma, "greedy (shipped)",
           @sprintf("%d run%s · cost %.3e", length(gruns), length(gruns) == 1 ? "" : "s", gcost))
    panel!(out, xB, y0, side, q, druns, gamma, "row-run DP (optimal)",
           @sprintf("%d run%s · cost %.3e · %+.1f%%", length(druns), length(druns) == 1 ? "" : "s", dcost, delta))

    push!(out, "</svg>")

    dir = joinpath(@__DIR__, "figures")
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, "schedule_$(dist).svg")
    open(path, "w") do io
        println(io, join(out, "\n"))
    end
    @printf("wrote %s   greedy=%.4e (%d runs)  dp=%.4e (%d runs)  %+.2f%%\n",
            path, gcost, length(gruns), dcost, length(druns), delta)
    return path
end

for dist in (:corner, :decay, :uniform)
    figure(dist)
end
