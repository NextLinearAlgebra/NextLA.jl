# Optional: julia test/catalog/setup.jl --plots
# julia --project=test/catalog test/catalog/plot.jl /path/to/run/results.csv
ENV["GKSwstype"] = "100" # headless cluster rendering
using Plots
include("common.jl")
using .CatalogSweep

length(ARGS) == 1 || error("Usage: plot.jl /path/to/results.csv")
rows = CatalogSweep.read_csv(ARGS[1])
isempty(rows) && error("No results to plot")
outdir = joinpath(dirname(abspath(ARGS[1])), "plots")
mkpath(outdir)
number(row, key) = something(tryparse(Float64, get(row, key, "")), NaN)
good(row, key) = row["status"] == "OK" && isfinite(number(row, key))
methods = [m for m in vcat(CatalogSweep.GPU_METHODS, ["sTiles_F64"]) if any(r -> r["method"] == m, rows)]
matrices = sort(unique(getindex.(rows, "matrix")); by=name -> number(first(filter(r -> r["matrix"] == name, rows)), "n"))
lookup = Dict((r["matrix"], r["method"]) => r for r in rows)
floor_error = 1e-18

z = fill(NaN, length(matrices), length(methods))
for (i, name) in enumerate(matrices), (j, method) in enumerate(methods)
    row = get(lookup, (name, method), nothing)
    row !== nothing && good(row, "factor_residual") && (z[i,j] = log10(max(number(row, "factor_residual"), floor_error)))
end
p = heatmap(1:length(methods), 1:length(matrices), z;
    xticks=(1:length(methods), methods), yticks=(1:length(matrices), matrices),
    xrotation=45, yflip=true, colorbar_title="log10 estimated residual", color=:viridis, clims=(-18,0),
    size=(1100, max(700, 25length(matrices))), title="Factor residual; blank = unavailable, not zero")
for (i, name) in enumerate(matrices), (j, method) in enumerate(methods)
    row = get(lookup, (name, method), nothing)
    if row !== nothing && row["status"] != "OK"
        label = row["status"] == "TIMEOUT" ? "T/O" : row["status"] == "SKIP_MEMORY" ? "MEM" : "ERR"
        annotate!(p, j, i, text(label, 6, :red))
    end
end
savefig(p, joinpath(outdir, "factor-residual-heatmap.png"))

function baseline_time(row, baseline)
    reference = get(lookup, (row["matrix"], baseline), nothing)
    reference === nothing && return NaN
    field = baseline == "sTiles_F64" ? "factor_ms" : "min_ms"
    good(reference, field) ? number(reference, field) : NaN
end

for baseline in ("cuSOLVER_F32", "sTiles_F64")
    p = plot(xscale=:log10, yscale=:log10, xlabel="Estimated factor residual (clipped at 1e-18)",
        ylabel="Baseline time / method time", title="Accuracy vs speedup against $baseline", legend=:outerright)
    for method in filter(m -> startswith(m, "M"), methods)
        selected = filter(r -> r["method"] == method && good(r, "factor_residual") && good(r, "min_ms") &&
            number(r,"min_ms") > 0 && baseline_time(r,baseline) > 0, rows)
        isempty(selected) && continue
        scatter!(p, [max(number(r,"factor_residual"),floor_error) for r in selected],
            [baseline_time(r,baseline)/number(r,"min_ms") for r in selected]; label=method)
    end
    hline!(p, [1.0]; label="equal runtime", linestyle=:dash, color=:black)
    savefig(p, joinpath(outdir, "accuracy-speedup-$baseline.png"))
end

p = plot(xscale=:log10, yscale=:log10, xlabel="Matrix dimension", ylabel="cuSOLVER F32 / method time",
    title="Runtime trend (accuracy shown separately)", legend=:outerright)
for method in filter(m -> startswith(m,"M"), methods)
    selected = filter(r -> r["method"] == method && good(r,"min_ms") && number(r,"min_ms") > 0 && baseline_time(r,"cuSOLVER_F32") > 0, rows)
    isempty(selected) && continue
    scatter!(p, [number(r,"n") for r in selected],
        [baseline_time(r,"cuSOLVER_F32")/number(r,"min_ms") for r in selected]; label=method)
end
hline!(p, [1.0]; label="equal runtime", linestyle=:dash, color=:black)
savefig(p, joinpath(outdir, "speedup-vs-dimension.png"))

p = plot(xscale=:log10, yscale=:log10, xlabel="Estimated input quantization error", ylabel="Estimated factor residual",
    title="Storage error vs factorization error", legend=:outerright)
for method in filter(m -> startswith(m,"M"), methods)
    selected = filter(r -> r["method"] == method && good(r,"quantization_error") && good(r,"factor_residual"), rows)
    isempty(selected) && continue
    scatter!(p, [max(number(r,"quantization_error"),floor_error) for r in selected],
        [max(number(r,"factor_residual"),floor_error) for r in selected]; label=method)
end
plot!(p, [floor_error,1.0], [floor_error,1.0]; label="equal error", linestyle=:dash, color=:black)
savefig(p, joinpath(outdir, "quantization-vs-factor-error.png"))
println("Plots saved to ", outdir)
