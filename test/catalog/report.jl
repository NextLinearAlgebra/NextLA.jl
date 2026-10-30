# CPU-only report generator. Rendering uses Plots, installed by setup.jl --plots.
# julia --project=test/catalog test/catalog/report.jl /path/to/run/results.csv [output-directory]
module CatalogReport
include("common.jl")
using .CatalogSweep
using Dates, Printf, SHA, Statistics, TOML

const BASELINES = ["cuSOLVER_F32", "cuSOLVER_F64", "sTiles_F64"]
const METRICS = Dict("factor_residual" => "Estimated factor residual",
    "forward_error" => "Known-solution forward error", "solve_residual" => "Equation residual",
    "backward_error" => "Frobenius-scaled backward error")
const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#332288", "#882255", "#555555", "#44AA99"]
const STATUS_COLORS = Dict("OK" => "#0072B2", "TIMEOUT" => "#E69F00", "ERROR" => "#D55E00",
    "NONFINITE" => "#CC79A7", "SKIP_MEMORY" => "#7766AA", "MISSING" => "#889099",
    "NOT_RECORDED" => "#DDE2E7")
const FLOOR = 1e-18
escape(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&#39;")
num(row, key) = something(tryparse(Float64, get(row, key, "")), NaN)
fmt(x) = isfinite(x) ? @sprintf("%.3g", x) : "—"
ok(row, key) = row !== nothing && get(row, "status", "") == "OK" && isfinite(num(row, key)) && num(row, key) >= 0
time_ms(row, timing) = ok(row, timing * "_ms") && num(row, timing * "_ms") > 0 ? num(row, timing * "_ms") : NaN
qualifies(row, metric, tolerance) = ok(row, metric) && num(row, metric) <= tolerance
geomean(xs) = isempty(xs) ? NaN : exp(mean(log.(xs)))

function prepare(rows, settings=Dict(); timing="median", metric="forward_error", tolerance=1e-3)
    isempty(rows) && error("No result rows yet")
    timing in ("min", "median") || error("REPORT_TIMING must be min or median")
    haskey(METRICS, metric) || error("Unknown REPORT_METRIC: $metric")
    isfinite(tolerance) && tolerance > 0 || error("REPORT_TOLERANCE must be finite and positive")
    all(r -> all(haskey(r, k) for k in ("matrix", "method", "status")), rows) || error("Missing required result columns")
    lookup = Dict((r["matrix"], r["method"]) => r for r in rows)
    length(lookup) == length(rows) || error("Duplicate matrix/method rows; report one experiment at a time")
    # A single run must not silently combine different builds or machines.
    for field in ("source_hash", "git_commit", "stiles_cores", "julia_version", "gpu", "cpu_model")
        values = unique([r[field] for r in rows if !isempty(get(r, field, ""))])
        length(values) <= 1 || error("Mixed $field values; report separate experiments")
    end
    observed_methods = unique(getindex.(rows, "method"))
    expected_methods = if haskey(settings, "gpu_methods") && haskey(settings, "methods")
        vcat(settings["methods"] == "stiles" ? String[] : String.(settings["gpu_methods"]),
            settings["methods"] == "gpu" ? String[] : ["sTiles_F64"])
    else
        observed_methods
    end
    method_set = unique(vcat(expected_methods, observed_methods))
    order = vcat(CatalogSweep.GPU_METHODS, ["sTiles_F64"])
    methods = vcat(filter(in(method_set), order), sort(setdiff(method_set, order)))
    matrices = unique(vcat(String.(get(settings, "selection", String[])), getindex.(rows, "matrix")))
    metadata = Dict(name => first(filter(r -> r["matrix"] == name, rows)) for name in unique(getindex.(rows, "matrix")))
    dimension(name) = get(metadata, name, nothing) === nothing ? Inf : num(metadata[name], "n")
    sort!(matrices; by=name -> (isfinite(dimension(name)) ? dimension(name) : Inf, name))
    (rows=rows, lookup=lookup, matrices=matrices, methods=methods, metadata=metadata,
        timing=timing, metric=metric, tolerance=tolerance, settings=settings)
end
rowat(d, name, method) = get(d.lookup, (name, method), nothing)
status(row) = row === nothing ? "NOT_RECORDED" : get(row, "status", "UNKNOWN")
function speedup(d, name, method, baseline)
    method_time, reference_time = time_ms(rowat(d, name, method), d.timing), time_ms(rowat(d, name, baseline), d.timing)
    ratio = isfinite(method_time) && isfinite(reference_time) ? reference_time / method_time : NaN
    isfinite(ratio) && ratio > 0 ? ratio : NaN
end

function summaries(d)
    summary, winners, problems = Dict{String,Any}[], Dict{String,Any}[], Dict{String,Any}[]
    for method in d.methods
        rr = [rowat(d, name, method) for name in d.matrices]
        errors = [num(r, d.metric) for r in rr if ok(r, d.metric)]
        result = Dict{String,Any}("method" => method, "selected_matrices" => length(rr),
            "recorded" => count(!isnothing, rr), "ok" => count(r -> status(r) == "OK", rr),
            "not_recorded" => count(isnothing, rr), "metric_available" => length(errors),
            "accuracy_metric" => d.metric, "tolerance" => d.tolerance,
            "accuracy_pass" => count(r -> qualifies(r, d.metric, d.tolerance), rr),
            "median_error" => isempty(errors) ? NaN : median(errors),
            "worst_error" => isempty(errors) ? NaN : maximum(errors), "timing_statistic" => d.timing)
        for baseline in BASELINES
            pairs = [speedup(d, name, method, baseline) for name in d.matrices]
            valid = filter(isfinite, pairs)
            passing = [s for (name, s) in zip(d.matrices, pairs) if isfinite(s) && qualifies(rowat(d, name, method), d.metric, d.tolerance)]
            result[baseline * "_pairs"] = length(valid)
            result[baseline * "_geomean_speedup"] = geomean(valid)
            result[baseline * "_accuracy_pass_pairs"] = length(passing)
            result[baseline * "_accuracy_pass_geomean_speedup"] = geomean(passing)
        end
        push!(summary, result)
    end
    for name in d.matrices
        valid = [r for m in d.methods for r in [rowat(d, name, m)] if qualifies(r, d.metric, d.tolerance) && isfinite(time_ms(r, d.timing))]
        sort!(valid; by=r -> time_ms(r, d.timing))
        winner = isempty(valid) ? nothing : first(valid)
        meta = get(d.metadata, name, Dict{String,String}())
        result = Dict{String,Any}("matrix" => name, "n" => get(meta, "n", ""), "group" => get(meta, "group", ""),
            "density" => get(meta, "density", ""), "accuracy_metric" => d.metric, "tolerance" => d.tolerance,
            "eligible_methods" => length(valid), "fastest_observed_qualifying_method" => winner === nothing ? "NONE" : winner["method"],
            "time_ms" => winner === nothing ? NaN : time_ms(winner, d.timing),
            "error" => winner === nothing ? NaN : num(winner, d.metric))
        for baseline in BASELINES
            result[baseline * "_speedup"] = winner === nothing ? NaN : speedup(d, name, winner["method"], baseline)
        end
        push!(winners, result)
        for method in d.methods
            row = rowat(d, name, method)
            if status(row) != "OK" || !ok(row, d.metric) || !qualifies(row, d.metric, d.tolerance)
                push!(problems, Dict{String,Any}("matrix" => name, "method" => method, "status" => status(row),
                    "accuracy_metric" => d.metric, "error" => row === nothing ? "" : get(row, d.metric, ""),
                    "detail" => status(row) != "OK" ? (row === nothing ? "No checkpoint in this snapshot" : get(row, "detail", "")) :
                        !ok(row, d.metric) ? "Metric unavailable or nonfinite" : "Above exploratory accuracy cutoff"))
            end
        end
    end
    summary, winners, problems
end

# Explicit stack intervals avoid relying on backend-specific bar-position support.
function stack_segments(values)
    all(x -> isfinite(x) && x >= 0,values) || error("Stack values must be finite and nonnegative")
    segments = NamedTuple[]
    for i in axes(values,1)
        lower = 0.0
        for j in axes(values,2)
            upper = lower + values[i,j]
            upper > lower && push!(segments,(row=i,column=j,lower=lower,upper=upper))
            lower = upper
        end
    end
    segments
end

point(row,x,y)=(method=row["method"],x=x,y=y,matrix=row["matrix"])

function figures(d; renderer=nothing)
    metric_heatmap = renderer === nothing ? CatalogReport.metric_heatmap : renderer.heatmap
    scatterplot = renderer === nothing ? CatalogReport.scatterplot : renderer.scatter
    statusplot = renderer === nothing ? CatalogReport.statusplot : renderer.status
    stilesplot = renderer === nothing ? CatalogReport.stilesplot : renderer.stiles
    output = NamedTuple[]
    add(id,title,caption,svg) = push!(output,(id=id,title=title,caption=caption,svg=svg))
    add("01-completion", "Completion and failures", "Read this before comparing speedups: unavailable and failed cases are part of the experiment.",statusplot(d))
    for (id,metric) in [("02-forward-error","forward_error"),("03-factor-residual","factor_residual"),
            ("04-equation-residual","solve_residual"),("05-backward-error","backward_error")]
        add(id,METRICS[metric], metric == "factor_residual" ?
            "Gaussian-probe estimate of ||(LL′−A)Ω|| / ||AΩ||. sTiles does not expose a reconstructed factor here, so its cells are unavailable." :
            "Maximum across the known-solution right-hand sides. GPU solves use decoded factors and FP64 triangular arithmetic, without refinement; sTiles uses its native solve.",
            metric_heatmap(d,METRICS[metric],"Darker = larger error; log10 scale clipped to [−18, 0]. Raw values remain in CSV.",
                (n,m) -> ok(rowat(d,n,m),metric) ? num(rowat(d,n,m),metric) : NaN; limits=(-18.0,0.0)))
    end
    add("06-runtime", "Runtime by matrix and method", "Timing excludes input transfer, resetting and accuracy checks. Mixed methods include container construction; pure methods and cuSOLVER cover factorization. sTiles covers numeric factorization, excluding analysis.",
        metric_heatmap(d,"$(uppercasefirst(d.timing)) runtime by matrix and method", "Milliseconds; timings are compared only for status OK.",
            (n,m) -> time_ms(rowat(d,n,m),d.timing); legend="log10 ms"))
    focus = filter(m -> startswith(m,"M") || startswith(m,"Pure"),d.methods)
    for (i,baseline) in enumerate(BASELINES)
        ratios = [speedup(d,n,m,baseline) for n in d.matrices,m in d.methods]
        finite = filter(x -> isfinite(x) && x > 0,vec(ratios))
        bound = isempty(finite) ? 1.0 : max(1.0,maximum(abs.(log2.(finite))))
        add("07$(Char('a'+i-1))-speedup-$baseline", "Speedup relative to $baseline",
            "Baseline time / method time on the same matrix, using $(d.timing) times. Blue means faster, orange slower, white equal. These are timing comparisons; check accuracy separately. No ratio is formed when either measurement is invalid.",
            metric_heatmap(d,"Speedup relative to $baseline", "Positive log2 values = faster; zero = equal time. Sparse CPU / dense GPU scopes differ.",
                (n,m) -> speedup(d,n,m,baseline); logarithm=log2,limits=(-bound,bound),legend="log2 speedup",diverging=true))
        points = [point(r,num(r,d.metric),speedup(d,r["matrix"],r["method"],baseline)) for r in d.rows
            if r["method"] in focus && ok(r,d.metric) && isfinite(speedup(d,r["matrix"],r["method"],baseline))]
        add("08$(Char('a'+i-1))-tradeoff-$baseline", "Accuracy–speed tradeoff: $baseline",
            "Upper left is preferable: smaller $(METRICS[d.metric]) and higher speedup. Dashed line denotes equal runtime. Points are matrices, not independent repetitions.",
            scatterplot(d,"Accuracy versus speedup: $baseline","$(d.timing) timings; successful paired measurements only.",points;
                xlabel=METRICS[d.metric],ylabel="Baseline time / method time",horizontal=1.0))
    end
    for (id,field,label,scale) in [("09-size-trend","n","Matrix dimension n",1.0),("10-density-trend","density","Input nonzero density (%)",100.0)]
        points = [point(r,scale*num(r,field),speedup(d,r["matrix"],r["method"],"cuSOLVER_F32")) for r in d.rows
            if r["method"] in focus && ok(r,field) && num(r,field)>0 && isfinite(speedup(d,r["matrix"],r["method"],"cuSOLVER_F32"))]
        add(id,"Speedup versus $label", "Descriptive trends across this selected catalog. Input sparsity is measured before dense GPU conversion; this is not a causal comparison.",
            scatterplot(d,"Speedup versus $label","Reference: cuSOLVER F32; $(d.timing) time; mixed and pure recursive schedules.",points;
                xlabel=label,ylabel="cuSOLVER F32 / method time",horizontal=1.0))
    end
    points = [point(r,num(r,"quantization_error"),num(r,"factor_residual")) for r in d.rows
        if startswith(r["method"],"M") && ok(r,"quantization_error") && ok(r,"factor_residual")]
    add("11-storage-error", "Input quantization versus factorization error", "Both axes are Gaussian-probe estimates. The dashed line is equal error, not a theoretical bound. This helps distinguish stored-input error from additional factorization error.",
        scatterplot(d,"Quantization versus factorization error","Mixed schedules; both errors measured against the original input.",points;
            xlabel="Estimated input quantization error",ylabel="Estimated factor residual",diagonal=true))
    for (id,label,xvalue,xlog) in [("12-dynamic-range","Input absolute-value dynamic range",r -> num(r,"max_abs")/num(r,"min_abs_nonzero"),true),
            ("13-diagonal-dominance","Fraction of strictly diagonally dominant rows",r -> num(r,"dominant_fraction"),false)]
        points = [point(r,xvalue(r),num(r,d.metric)) for r in d.rows if startswith(r["method"],"M") && ok(r,d.metric) && isfinite(xvalue(r))]
        add(id,"Accuracy versus $label", "Exploratory association across matrices. Dynamic range and diagonal dominance are diagnostics, not condition-number estimates or explanations of causation.",
            scatterplot(d,"Accuracy versus $label","Mixed schedules; each point is one matrix.",points;
                xlabel=label,ylabel=METRICS[d.metric],xlog=xlog,xrange=xlog ? nothing : (0.0,1.0)))
    end
    thresholds = sort(unique(vcat([1e-8,1e-6,1e-4,1e-3,1e-2],d.tolerance)))
    points = [(method=m,x=t,y=100count(n -> qualifies(rowat(d,n,m),d.metric,t),d.matrices)/length(d.matrices),
        tip="$m: threshold=$t; denominator=$(length(d.matrices)) selected matrices") for m in d.methods for t in thresholds]
    add("14-accuracy-coverage", "Accuracy coverage at several tolerances", "Denominator is every selected matrix, including failures and not-yet-recorded cases. Cutoffs are exploratory and adjustable, not application-specific accuracy guarantees.",
        scatterplot(d,"Accuracy coverage across tolerances",METRICS[d.metric]*"; failures and unavailable metrics never count as passes.",points;
            xlabel="Accuracy cutoff",ylabel="Selected matrices passing (%)",ylog=false,lines=true,yrange=(0.0,100.0)))
    add("15-stiles-phases","sTiles analysis cost", "CPU sparse analysis and numeric factorization are shown separately to make the comparison scope explicit.",stilesplot(d))
    output
end

function table(io,rows,columns)
    print(io,"<div class=\"table-wrap\"><table><thead><tr>")
    for (key,label) in columns; print(io,"<th>$(escape(label))</th>"); end
    print(io,"</tr></thead><tbody>")
    for row in rows
        print(io,"<tr>")
        for (key,_) in columns
            value = get(row,key,"")
            print(io,"<td>$(escape(value isa AbstractFloat ? fmt(value) : value))</td>")
        end
        print(io,"</tr>")
    end
    print(io,"</tbody></table></div>")
end
function html_report(d,figs,summary,winners,problems,snapshot_hash,settings_text)
    io = IOBuffer()
    print(io,"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Cholesky catalog · accuracy and performance</title><style>
    body{margin:0;background:#f2f5f8;color:#1b2b40;font:16px/1.55 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:32px 24px}
    h1{font-size:38px;line-height:1.15;margin:8px 0 14px}h2{margin-top:38px}h3{margin:0 0 8px}.eyebrow{letter-spacing:.14em;text-transform:uppercase;color:#34618b;font-size:12px;font-weight:700}
    .cards{display:flex;gap:14px;flex-wrap:wrap}.card{background:white;border:1px solid #dce4ed;border-radius:10px;padding:16px 24px;flex:1}.card b{display:block;font-size:30px}.note{background:#e6eff8;border-left:4px solid #286397;padding:14px 18px}
    figure{background:white;border:1px solid #dce4ed;border-radius:12px;margin:24px 0;padding:20px;break-inside:avoid}svg{display:block;width:100%;height:auto}figcaption{color:#526176;font-size:14px}
    a{color:#175f97}nav{columns:2}nav a{display:block;margin:5px 0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #dce4ed}th{background:#e5edf5;white-space:nowrap}td{max-width:480px;overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:12px}details{margin:20px 0}summary{cursor:pointer;font-weight:600}.small{font-size:13px;color:#526176}
    @media print{body{background:white}main{padding:0}nav{display:none}figure{border:0;padding:0;break-before:page}details{display:block}a{color:inherit;text-decoration:none}.table-wrap{overflow:visible}}
    </style><main><div class="eyebrow">Real-matrix survey · one experiment snapshot</div><h1>Cholesky accuracy<br>and performance</h1>""")
    print(io,"<p>Generated $(escape(Dates.format(now(),"yyyy-mm-dd HH:MM"))) · CPU-only analysis of saved checkpoints.</p>")
    total=length(d.matrices)*length(d.methods)
    goodcount=count(r -> get(r,"status","")=="OK",d.rows)
    print(io,"<div class=\"cards\"><div class=\"card\"><b>$(length(d.matrices))</b>selected matrices</div><div class=\"card\"><b>$(length(d.methods))</b>methods</div><div class=\"card\"><b>$(length(d.rows)) / $total</b>recorded cells</div><div class=\"card\"><b>$goodcount</b>status OK</div></div>")
    print(io,"<p class=\"note\"><strong>Reading this report.</strong> OK means finite measurements, not acceptable accuracy. The summary uses <strong>$(escape(METRICS[d.metric])) ≤ $(fmt(d.tolerance))</strong> as an exploratory cutoff and <strong>$(d.timing)</strong> runtime across repetitions. Missing cells and failures are visible. Geometric-mean speedups use paired successful timings only; pair counts and accuracy coverage must be read alongside them.</p>")
    print(io,"<p class=\"small\">Method key: Mk uses k FP16 block levels followed by FP32 leaves. Pure_F32/F64 are uniform-precision recursive schedules; cuSOLVER is the GPU library baseline; sTiles_F64 is the CPU sparse baseline.</p>")
    if length(d.rows)<total
        print(io,"<p class=\"note\"><strong>Partial snapshot:</strong> $(total-length(d.rows)) matrix/method cells are not recorded. A partial report does not establish that the sweep has finished.</p>")
    end
    if !haskey(d.settings,"selection")
        print(io,"<p class=\"note\">No experiment selection was available in settings.toml. Matrix and method coverage is inferred from observed rows; entirely unrecorded matrices cannot be counted.</p>")
    end
    print(io,"<h2>Method summary</h2><p>Accuracy pass counts include all selected matrices in the denominator. ‘Paired GM’ includes every available paired timing, regardless of the accuracy cutoff.</p>")
    columns=[("method","Method"),("ok","OK"),("accuracy_pass","Accuracy passes"),("selected_matrices","Selected"),
        ("worst_error","Worst finite OK error"),("cuSOLVER_F32_pairs","F32 pairs"),("cuSOLVER_F32_geomean_speedup","F32 paired GM speedup"),
        ("sTiles_F64_pairs","sTiles pairs"),("sTiles_F64_geomean_speedup","sTiles paired GM speedup")]
    table(io,summary,columns)
    print(io,"<h2>Fastest observed method meeting the cutoff</h2><p>Selected independently on each matrix among recorded eligible methods. This is descriptive selection on this dataset, not a validated adaptive policy.</p>")
    table(io,winners,[("matrix","Matrix"),("n","Dimension"),("eligible_methods","Eligible methods"),
        ("fastest_observed_qualifying_method","Fastest qualifying observed"),("time_ms","Time (ms)"),("error","Error"),
        ("cuSOLVER_F32_speedup","vs cuSOLVER F32"),("sTiles_F64_speedup","vs sTiles")])
    print(io,"<h2>Figures</h2><p>Each figure is saved as a vector SVG and a PNG for slides. Matrix names are labeled on heatmaps; raw values are retained in the snapshot CSV. This HTML embeds all figures and works offline.</p><nav>")
    for f in figs; print(io,"<a href=\"#$(f.id)\">$(escape(f.title))</a>"); end
    print(io,"</nav>")
    for f in figs
        print(io,"<figure id=\"$(f.id)\"><h3>$(escape(f.title))</h3>$(f.svg)<figcaption>$(escape(f.caption))</figcaption></figure>")
    end
    print(io,"<details><summary>Failures, unavailable metrics and observations above the cutoff ($(length(problems)))</summary>")
    table(io,problems,[("matrix","Matrix"),("method","Method"),("status","Status"),("error","Error"),("detail","Detail")])
    print(io,"</details><h2>Measurement scope and provenance</h2><ul><li>GPU algorithms convert the sparse input to dense storage. sTiles is a CPU sparse baseline; differences reflect algorithms and hardware as well as precision.</li><li>Mixed timing includes construction plus factorization. Pure recursive/cuSOLVER timing covers factorization. Input transfer, reset, decoding and accuracy are excluded.</li><li>sTiles speedup uses numeric factorization only. Its analysis time is shown separately using the minimum combined repetition. Timing scatter is not an uncertainty interval.</li><li>Factor and input-quantization residuals are randomized Gaussian-probe estimates, not exact Frobenius residuals. Forward/equation/backward errors use known-solution right-hand sides.</li><li>GPU solve diagnostics use decoded factors with FP64 triangular arithmetic and no refinement. sTiles uses native solves. Backward error uses the Frobenius norm of A, not a spectral norm.</li><li>Zero errors are plotted at 1e-18. Heatmaps clip log10 error colors to [−18, 0]; raw values are retained in CSVs.</li><li>Timeouts include startup, loading, compilation, repeats and accuracy checks; a timed-out method was not necessarily the cause of a slow worker.</li><li>Matrix feature plots are exploratory. No condition number is measured, and this selected catalog is not a random population sample.</li></ul>")
    metadata = [(field,join(unique([get(r,field,"") for r in d.rows if !isempty(get(r,field,""))]),"; ")) for field in
        ("gpu","cpu_model","stiles_cores","blas_threads","julia_version","git_commit","source_hash","repeats","probes","nrhs")]
    table(io,[Dict("field"=>k,"value"=>v) for (k,v) in metadata],[("field","Field"),("value","Recorded value")])
    print(io,"<p class=\"small\">Results snapshot SHA-256: $(escape(snapshot_hash)). CSV files alongside this report retain raw values, paired sample counts, qualifying-only speedups and failure details. For PDF, use your browser’s Print → Save as PDF.</p>")
    isempty(settings_text) || print(io,"<details><summary>Recorded experiment settings</summary><pre>$(escape(settings_text))</pre></details>")
    print(io,"</main></html>")
    String(take!(io))
end

function generate(input; outdir=joinpath(dirname(abspath(input)),"report"), timing=get(ENV,"REPORT_TIMING","median"),
        metric=get(ENV,"REPORT_METRIC","forward_error"), tolerance=parse(Float64,get(ENV,"REPORT_TOLERANCE","1e-3")))
    mkpath(joinpath(outdir,"figures"))
    snapshot = read(input)
    snapshot_path = joinpath(outdir,"results-snapshot.csv")
    write(snapshot_path,snapshot)
    rows = CatalogSweep.read_csv(snapshot_path)
    settings_path = joinpath(dirname(abspath(input)),"settings.toml")
    settings_text = isfile(settings_path) ? read(settings_path,String) : ""
    settings = isempty(settings_text) ? Dict() : TOML.parse(settings_text)
    d = prepare(rows,settings; timing=timing,metric=metric,tolerance=tolerance)
    summary,winners,problems = summaries(d)
    function savecsv(name,records,empty_columns)
        columns = isempty(records) ? empty_columns : sort(unique(vcat(collect.(keys.(records))...)))
        CatalogSweep.write_csv(joinpath(outdir,name),records; columns=columns)
    end
    savecsv("method-summary.csv",summary,["method"])
    savecsv("matrix-summary.csv",winners,["matrix"])
    savecsv("problems.csv",problems,["matrix","method","status","detail"])
    isempty(settings_text) || write(joinpath(outdir,"settings-snapshot.toml"),settings_text)
    isdefined(@__MODULE__,:Plots) || error("Rendering is not initialized; run report.jl as a script or load report_render.jl at top level first")
    rawfigs = figures(d)
    figs = NamedTuple[]
    for f in rawfigs
        svg_path = joinpath(outdir,"figures",f.id*".svg")
        Plots.savefig(f.svg,svg_path)
        Plots.savefig(f.svg,joinpath(outdir,"figures",f.id*".png"))
        svg = read(svg_path,String)
        svg = replace(svg,r"<\?xml[^>]*\?>"=>"",r"<!DOCTYPE[^>]*>"=>"")
        # GR reuses SVG IDs; namespace each image before embedding in one document.
        svg = replace(svg,"id=\""=>"id=\"$(f.id)-","url(#"=>"url(#$(f.id)-","href=\"#"=>"href=\"#$(f.id)-")
        push!(figs,merge(f,(svg=svg,)))
    end
    hash = bytes2hex(sha256(snapshot))
    write(joinpath(outdir,"report.html"),html_report(d,figs,summary,winners,problems,hash,settings_text))
    open(joinpath(outdir,"report-settings.toml"),"w") do io
        TOML.print(io,Dict("input_sha256"=>hash,"input"=>abspath(input),"timing"=>timing,"accuracy_metric"=>metric,
            "accuracy_tolerance"=>tolerance,"selected_matrices"=>length(d.matrices),"recorded_rows"=>length(rows)))
    end
    println("Saved $(length(figs)) vector figures and report: ",abspath(joinpath(outdir,"report.html")))
    outdir
end
function main(args=ARGS)
    1 <= length(args) <= 2 || error("Usage: julia test/catalog/report.jl /path/to/results.csv [output-directory]")
    length(args)==2 ? generate(args[1];outdir=args[2]) : generate(args[1])
end

# Julia 1.12 also tracks global bindings in world age. Import the renderer at
# module top level, before main/generate runs, instead of evaluating using Plots
# inside an already-running function. Pure calculation tests do not need Plots.
if abspath(PROGRAM_FILE)==abspath(@__FILE__) || "--render" in ARGS
    include("report_render.jl")
end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && CatalogReport.main()
