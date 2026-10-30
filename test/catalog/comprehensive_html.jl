# Included at module top level by comprehensive.jl. No plotting dependencies.
function htable(io,headers,records)
    print(io,"<div class='scroll'><table><thead><tr>")
    for header in headers;print(io,"<th>",esc(header),"</th>");end
    print(io,"</tr></thead><tbody>")
    for record in records
        print(io,"<tr>")
        for cell in record;print(io,"<td>",esc(fnum(cell)),"</td>");end
        print(io,"</tr>")
    end
    print(io,"</tbody></table></div>")
end
function paragraph(io,body; takeaway=nothing)
    takeaway===nothing || print(io,"<p class='takeaway'><strong>Takeaway: ",takeaway,"</strong></p>")
    print(io,"<p>",body,"</p>")
end
function json_literal(x)
    x===nothing && return "null"
    x isa Bool && return x ? "true" : "false"
    x isa Number && return isfinite(x) ? string(x) : "null"
    x isa AbstractDict && return "{"*join((json_literal(string(k))*":"*json_literal(v) for (k,v) in x),",")*"}"
    x isa AbstractVector && return "["*join(json_literal.(x),",")*"]"
    "\""*replace(string(x),'\\'=>"\\\\",'"'=>"\\\"",'\n'=>"\\n",'\r'=>"\\r",'\t'=>"\\t",'<'=>"\\u003c")*"\""
end
function interactive_figure(io,outdir,id)
    specs=CatalogSweep.read_csv(joinpath(outdir,"chart-data","charts.csv"))
    spec=only(filter(s->s["id"]==id,specs))
    points=CatalogSweep.read_csv(joinpath(outdir,"chart-data",id*".csv"))
    traces=Any[]
    observed=unique(getindex.(points,"series"))
    series_order=vcat(filter(in(observed),METHODS),setdiff(observed,METHODS))
    for series in series_order
        selected=sort(filter(p->p["series"]==series,points);by=p->nval(p,"x"))
        trace=Dict{String,Any}("name"=>series,"type"=>"scatter","mode"=>spec["kind"]=="line" ? "lines+markers" : "markers",
            "x"=>nval.(selected,"x"),"y"=>nval.(selected,"y"),"text"=>getindex.(selected,"label"),
            "marker"=>Dict{String,Any}("size"=>8),"hovertemplate"=>"%{text}<br>x=%{x:.4g}<br>y=%{y:.4g}<extra>%{fullData.name}</extra>")
        methodindex=findfirst(==(series),METHODS)
        if methodindex!==nothing
            color=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#332288","#882255","#555555","#44AA99"][methodindex]
            trace["marker"]["color"]=color;trace["line"]=Dict("color"=>color)
        end
        if startswith(series,"Backward")
            strict=endswith(series,"1e-6")
            trace["line"]=Dict("dash"=>strict ? "dot" : "dash")
            trace["marker"]["symbol"]=strict ? "diamond-open" : "diamond"
        end
        push!(traces,trace)
    end
    xaxis=Dict{String,Any}("title"=>Dict("text"=>spec["xlabel"]),"type"=>spec["xlog"]=="true" ? "log" : "linear","gridcolor"=>"#e2e7ee")
    yaxis=Dict{String,Any}("title"=>Dict("text"=>spec["ylabel"]),"type"=>spec["ylog"]=="true" ? "log" : "linear","gridcolor"=>"#e2e7ee")
    for (axis,key,logscale) in [(xaxis,"x",spec["xlog"]=="true"),(yaxis,"y",spec["ylog"]=="true")]
        if logscale
            vals=filter(>(0),nval.(points,key));lo,hi=extrema(log10.(vals))
            axis["dtick"]=max(1,ceil(Int,(hi-lo)/8));axis["tickformat"]=".0e"
        end
    end
    if id=="29-current-coverage"
        xaxis["tickvals"]=collect(1:length(METHODS));xaxis["ticktext"]=METHODS;xaxis["tickangle"]=-25
        yaxis["range"]=[0,27];yaxis["dtick"]=2
    elseif occursin(r"^3[123]-core",id)
        xaxis["tickvals"]=[1,2,4,8,14]
    elseif startswith(id,"20-") || startswith(id,"21-")
        xaxis["tickvals"]=[4096,8192,16384,32768,65536];xaxis["ticktext"]=["4,096","8,192","16,384","32,768","65,536"]
    elseif startswith(id,"27-")
        xaxis["tickvals"]=[.01,.03,.1,.3,1,3,10,30];xaxis["ticktext"]=["0.01","0.03","0.1","0.3","1","3","10","30"]
    elseif startswith(id,"28-")
        xaxis["tickvals"]=[5000,10000,20000,40000,60000];xaxis["ticktext"]=["5,000","10,000","20,000","40,000","60,000"]
    end
    if occursin("shift",id)
        xaxis["tickvals"]=collect(1:6);xaxis["ticktext"]=OLD_NAMES;xaxis["tickangle"]=-20
    end
    layout=Dict{String,Any}("title"=>Dict("text"=>spec["title"]),"xaxis"=>xaxis,
        "yaxis"=>yaxis,
        "height"=>680,"font"=>Dict("family"=>"Arial, sans-serif","size"=>13,"color"=>"#263648"),
        "margin"=>Dict("l"=>95,"r"=>45,"t"=>70,"b"=>170),"legend"=>Dict("orientation"=>"h","y"=>-.22),
        "colorway"=>["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#332288","#882255","#555555","#44AA99"])
    if occursin("crossing",id) || id in ("30-current-tradeoff","35-current-backward-tradeoff")
        layout["shapes"]=[Dict("type"=>"line","xref"=>"paper","x0"=>0,"x1"=>1,"y0"=>1,"y1"=>1,"line"=>Dict("color"=>"gray","dash"=>"dash"))]
        if id in ("30-current-tradeoff","35-current-backward-tradeoff")
            push!(layout["shapes"],Dict("type"=>"line","yref"=>"paper","y0"=>0,"y1"=>1,"x0"=>1e-3,"x1"=>1e-3,"line"=>Dict("color"=>"#b96316","dash"=>"dot")))
        end
    end
    print(io,"<div id='chart-",id,"' style='min-height:680px'></div><script>window.reportChartJobs.push(Plotly.newPlot('chart-",id,"',",json_literal(traces),",",json_literal(layout),",{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'",id,"',scale:2}}).then(g=>{window.reportCharts.push({id:'",id,"',g:g});}));</script>")
end
function figure(io,outdir,id,caption,takeaway,analysis)
    path=joinpath(outdir,"figures",id*".png")
    print(io,"<figure id='fig-",id,"'>")
    forced=get(ENV,"COMPREHENSIVE_INTERACTIVE","0")=="1" && isfile(joinpath(outdir,"chart-data",id*".csv"))
    if isfile(path) && !forced
        print(io,"<img alt='",esc(caption),"' src='data:image/png;base64,",base64encode(read(path)),"'/>")
    else
        interactive_figure(io,outdir,id)
    end
    print(io,"<figcaption>",caption,"</figcaption>")
    paragraph(io,analysis;takeaway=takeaway)
    print(io,"</figure>")
end
function analysis_rows(rows;kind="real",policy="original",cohort=nothing)
    filter(r->r["kind"]==kind && (kind=="synthetic" || r["policy"]==policy) && (cohort===nothing || r["cohort"]==cohort),rows)
end
function comparison_table(rows,inv)
    d=lookup(rows);out=Vector{Any}[]
    for meta in inv
        meta["kind"]=="real" || continue
        name=meta["matrix"];rr=filter(r->r["matrix"]==name && r["policy"]=="original",rows)
        best=bestmixed(rr);fast=bestmixed(rr;qualified=false)
        cpu=d[(name,"original","sTiles_F64")];gpu=d[(name,"original","cuSOLVER_F32")]
        ratio(ref,r)=r!==nothing && validtime(ref) ? nval(ref,"time_ms")/nval(r,"time_ms") : NaN
        push!(out,[name,meta["cohort"],nval(meta,"n"),nval(meta,"density_percent"),nval(meta,"sparsity_percent"),
            fast===nothing ? "none" : fast["method"],best===nothing ? "none" : best["method"],
            best===nothing ? NaN : nval(best,"forward_error"),ratio(cpu,best),ratio(gpu,best)])
    end
    out
end
const REPORT_SECTIONS=[
    ("summary","Main findings and forward/backward summary plots"),
    ("crossover","Performance: where mixed precision wins and loses"),
    ("scaling","CPU core scaling: actual measurements"),
    ("synthetic","Synthetic dense matrices: the size trend"),
    ("depth","Why deeper mixed precision is not always better"),
    ("errors","Understanding the five error metrics"),
    ("failures","Failures, timeouts and accuracy coverage"),
    ("quantization","Where quantization works and fails"),
    ("shifting","Diagonal shifting: a separate paired experiment"),
    ("inventory","Every matrix and its properties"),
    ("dossiers","Matrix-by-matrix results and interpretation"),
    ("sources","Experimental methods and provenance"),
    ("next","Remaining questions and next measurements"),
    ("conclusion","Overall conclusion and data audit")]
# Keep generation close to its analysis code, but give readers a single explicitly
# ordered route from conclusions to technical evidence. The TOC uses this same list.
function order_report(html)
    matches=collect(eachmatch(r"<h2 id='([^']+)'>[^<]+</h2>",html))
    length(matches)==length(REPORT_SECTIONS) || error("Report section count mismatch")
    footer_start=findfirst("</main>",html);footer_start===nothing && error("Missing report footer")
    chunks=Dict{String,String}()
    for (i,m) in enumerate(matches)
        finish=i<length(matches) ? matches[i+1].offset : first(footer_start)
        chunks[m.captures[1]]=html[m.offset:prevind(html,finish)]
    end
    prefix=html[1:prevind(html,first(matches).offset)]
    footer=html[first(footer_start):end]
    body=join(replace(chunks[id],r"^<h2[^>]*>[^<]+</h2>"=>"<h2 id='$id'>$i. $title</h2>";count=1) for (i,(id,title)) in enumerate(REPORT_SECTIONS))
    prefix*body*footer
end
function build_html(outdir,rundir)
    for path in readdir(joinpath(rundir,"report","figures");join=true)
        endswith(path,".png") && cp(path,joinpath(outdir,"figures",basename(path));force=true)
    end
    spec=CatalogSweep.read_csv(joinpath(outdir,"chart-data","charts.csv"))
    interactive=get(ENV,"COMPREHENSIVE_INTERACTIVE","0")=="1" || any(s->!isfile(joinpath(outdir,"figures",s["id"]*".png")),spec)
    if interactive
        assets=joinpath(outdir,"assets");mkpath(assets)
        library=joinpath(assets,"plotly.min.js")
        isfile(library) || Downloads.download("https://cdn.plot.ly/plotly-3.1.0.min.js",library;timeout=60)
    end
    allrows=CatalogSweep.read_csv(joinpath(outdir,"data","all-results.csv"));rows=primary_rows(allrows)
    length(allrows)==445 || error("This report requires the completed extended sweep; pass its directory to --prepare")
    inv=CatalogSweep.read_csv(joinpath(outdir,"data","matrix-inventory.csv"))
    d=lookup(rows);original=analysis_rows(rows);catalog=analysis_rows(rows;cohort="Catalog / 14-core CPU")
    legacy=analysis_rows(rows;cohort="Legacy rerun / 14-core CPU");dense=analysis_rows(rows;kind="synthetic")
    historical=historical_rows(allrows);paired=lookup(historical);scaling=scaling_rows(allrows)
    io=IOBuffer()
    print(io,"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Cholesky: comprehensive accuracy and performance analysis</title><style>
    :root{--ink:#172c43;--muted:#51647a;--accent:#126490;--line:#dce5ee}*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1400px;margin:auto;padding:36px 34px}h1{font-size:42px;line-height:1.12;max-width:1000px}h2{font-size:29px;margin:54px 0 15px;border-top:2px solid var(--line);padding-top:22px}h3{font-size:21px;margin-top:30px}h4{margin:24px 0 10px}p{max-width:1120px}a{color:#126490}.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:12px;color:var(--accent);font-weight:700}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{flex:1;background:white;border:1px solid var(--line);border-radius:12px;padding:18px;min-width:180px}.card strong{display:block;font-size:30px}.takeaway{border-left:5px solid var(--accent);background:#e5eff8;padding:14px 18px;max-width:none}.caution{border-left:5px solid #b96316;background:#fff3e4;padding:15px 20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:15px 0}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#e6edf5;position:sticky;top:0}td{max-width:380px;overflow-wrap:anywhere}figure{margin:28px 0;background:white;border:1px solid var(--line);border-radius:12px;padding:20px;break-inside:avoid}figure img{width:100%;height:auto;display:block}figcaption{color:var(--muted);font-size:14px;margin:8px 0}.small{font-size:13px;color:var(--muted)}nav{columns:2;max-width:1000px;background:white;border-radius:10px;padding:20px 28px}nav a{display:block;padding:3px 0}code{background:#e6edf4;border-radius:4px;padding:2px 5px}details{margin:24px 0;padding:15px 20px;background:white;border:1px solid var(--line);border-radius:10px}summary{font-weight:700;cursor:pointer}.matrix{border-top:1px solid var(--line);padding-top:20px}ul,ol{max-width:1120px}.sources{font-size:14px}.badge{display:inline-block;padding:2px 8px;background:#e4edf6;border-radius:5px;font-size:12px}
    @media print{main{padding:0}body{background:white}nav{display:none}h2{break-before:page}figure{padding:4px;border:0}th{position:static}.scroll{overflow:visible}details{border:0;padding:0}a{color:inherit}}
    </style></head><body><main><div class="eyebrow">Cholesky survey · full evidence and interpretation</div><h1>Where mixed precision helps,<br>where it loses accuracy,<br>and why depth alone is insufficient</h1>""")
    if interactive
        print(io,"<script>",replace(read(joinpath(outdir,"assets","plotly.min.js"),String),"</script"=>"<\\/script"),"</script><script>window.reportCharts=[];window.reportChartJobs=[];</script>")
    end
    print(io,"<p>26 real matrices with current raw results, five synthetic dense dimensions, six historical original/shifted pairs, and a completed 1–14-core sTiles sweep. Generated ",Dates.format(now(),"yyyy-mm-dd HH:MM"),".</p>")
    print(io,"<div class='cards'><div class='card'><strong>26</strong>current real inputs</div><div class='card'><strong>5</strong>synthetic dimensions</div><div class='card'><strong>",length(allrows),"</strong>observations retained</div><div class='card'><strong>15</strong>CPU scaling measurements</div></div>")
    paragraph(io,"<strong>Reading route:</strong> start with the findings and three overview plots: coverage, speed versus forward error, and speed versus backward error. Continue through measured performance, CPU scaling and the synthetic size trend. The later chapters unpack precision depth, error definitions, failures and quantization. Inventory, individual tables and provenance come last. <strong>Accuracy key:</strong> forward error measures recovery of a known solution; 10⁻³ means 0.1% relative vector error. Backward error measures a scaled equation residual and has a different interpretation. The screens are exploratory. M1/M2/M4/M6/M7 use progressively deeper FP16 block hierarchies with FP32 leaves.")
    print(io,"<nav>");for (i,(id,title)) in enumerate(REPORT_SECTIONS);print(io,"<a href='#",id,"'>",i,". ",title,"</a>");end;print(io,"</nav>")

    print(io,"<h2 id='summary'>1. Overall findings</h2>")
    paragraph(io,"The six older real matrices have now been rerun under the H200 setup with five timing repetitions and a 14-core sTiles baseline. Main original-input comparisons use these fresh raw CSVs alongside the 20-matrix catalog. The older paired shifting study and synthetic study remain separate, with their original labels. Main conclusions no longer depend on the earlier one-core real timing tables.";takeaway="Mixed precision has useful cases, but the fastest accurate schedule depends on the matrix and the baseline.")
    print(io,"<ol><li><strong>Accuracy narrows the useful set.</strong> Across all 26 current real inputs, M1 meets forward error ≤10⁻³ on 14, while each deeper schedule meets it on 11. M1 completes 25/26; M7 completes 16/26. Pure F64 and cuSOLVER F64 pass all 26. sTiles passes all 25 completed inputs; animal2 times out.</li>")
    print(io,"<li><strong>Input-storage accuracy and numerical factorization accuracy are separate bottlenecks.</strong> gyro_m loses substantial information in its mixed input representation. In contrast, animal1 has quantization probes near 10⁻¹⁵ but a growing factorization error as depth increases. Low input error does not make subsequent arithmetic exact.</li>")
    print(io,"<li><strong>Hierarchy depth has a size-dependent optimum.</strong> The fastest synthetic mixed schedule progresses M1 → M1 → M2 → M4 → M6 across n=4,096, 8,192, 16,384, 32,768 and 65,536. M7 is slower than M6 even at the largest synthetic dimension.</li>")
    print(io,"<li><strong>The baseline changes the conclusion.</strong> On catalog spacetime, M4 is 1.36× faster than cuSOLVER F32 and 1.08× faster than 14-core sTiles, while meeting the 10⁻³ forward-error screen. On sem_n5000, M4 is 1.79× faster than cuSOLVER F32 but about 20× slower than sTiles.</li>")
    print(io,"<li><strong>Animal2 remains a positive GPU result.</strong> Current M6 takes 357.48 ms versus cuSOLVER F32's 939.48 ms: 2.63× faster, with forward error 3.24×10⁻⁶. Its CPU worker hits the 300-second wall-time limit during analysis, so no numeric-factor CPU ratio is available. The separate earlier n=65,536 synthetic result reaches 4.80× versus cuSOLVER F32.</li>")
    print(io,"<li><strong>New reruns expose an accuracy cost behind fast times.</strong> Only M1 meets the 10⁻³ forward screen on bodyy4/5/6, and sTiles is faster. No mixed schedule meets it on original nasasrb or crankseg_2; M6/M7 fail on both. A fast runtime is not evidence of a useful solve.</li>")
    print(io,"<li><strong>More CPU cores do not always help.</strong> Animal1 improves from 415.60 ms at one requested core to 82.72 ms at 14 (5.02×). Spacetime is fastest at 8 (131.93 ms); sem_n5000 is fastest at 2 (2.46 ms), and 14 cores are slower than one. These are measured factor medians, not an assumed scaling law.</li>")
    print(io,"<li><strong>Shifting improves measured accuracy but changes the mathematical problem.</strong> It must not be described as fixing the unmodified input. Different error metrics can also disagree about which shifted schedule is adequate.</li></ol>")
    paragraph(io,"The 10⁻³ screen is a transparent exploratory choice, not a universal definition of useful accuracy. Below, both factor and solution criteria are reported, alongside 10⁻⁶ and 10⁻² sensitivity checks. A 10% or larger known-solution error is highlighted as a serious concern for applications requiring accurate solutions, without declaring every application has the same requirements.")
    figure(io,outdir,"29-current-coverage","Overview A — completion, forward-error and backward-error screens across all 26 current real matrices.","Backward error admits many more mixed results than forward error; passing one is not a guarantee of passing the other.","Each method has the same denominator of 26, including failures and the CPU timeout. Solid accuracy lines show forward error ≤10⁻³ and ≤10⁻⁶. Diamond-marked backward-error lines show ≤10⁻³ (dashed, filled diamonds) and ≤10⁻⁶ (dotted, open diamonds). M1 passes 25 backward versus 14 forward screens at 10⁻³, and 24 backward versus six forward screens at 10⁻⁶. M7 passes 15 versus 11 at 10⁻³, and 12 versus two at 10⁻⁶. These cutoffs have different meanings, so this is a criterion-sensitivity comparison, not equivalent acceptance standards. The method axis is categorical. All values are current raw real-input observations; historical and synthetic results are excluded.")
    figure(io,outdir,"30-current-tradeoff","Overview B — every completed current mixed case: solution error versus GPU-library speedup.","Useful speedups lie above the horizontal equal-time line and left of the exploratory error cutoff.","The horizontal axis is forward error, so farther left is more accurate. The vertical axis is cuSOLVER F32 median divided by mixed median; above one means faster. The dotted vertical line is forward error 10⁻³. Both axes are logarithmic. Colors identify schedules; the interactive version reveals the matrix on hover. Failed mixed cases have no valid point and remain in the coverage denominator in Overview A. This plot compares dense GPU methods, not sTiles.")
    figure(io,outdir,"35-current-backward-tradeoff","Overview C — the same current mixed runs, with backward error as the screening metric.","The apparent fast-and-passing region grows under a backward-error criterion, including some solutions with large forward errors.","The vertical axis and method colors match Overview B. The horizontal axis is now ||Ax̂−b||₂ / (||A||F||x̂||₂+||b||₂), maximized over four RHS. Above the horizontal line and left of the dotted 10⁻³ cutoff means faster than cuSOLVER F32 and passing this backward screen only. In the interactive report, hovering includes the corresponding forward error. Both axes are logarithmic; failed methods are excluded from the scatter but counted in Overview A. The Frobenius denominator can make this quantity small even when the recovered solution is inaccurate.")
    print(io,"<h3>Which metric should we show the advisors?</h3>")
    paragraph(io,"<strong>Show both, and state the task.</strong> For these manufactured known-solution tests, forward error directly answers whether x̂ recovered the supplied x_true. Backward error asks how small the scaled equation residual is, which remains useful when a true solution is unavailable. It does not guarantee a small distance to x_true on a sensitive problem. Factor residual answers a third question: whether the factor reproduces A. The report's main accuracy-qualified performance comparison continues to use forward error; the new backward view shows how the conclusion changes under a different criterion. An application should choose its own forward and backward tolerances separately, rather than assume the same numerical cutoff is equally stringent.")
    htable(io,["Current run / method","Forward error","Backward error","Forward ≤1e−3?","Backward ≤1e−3?","vs cuSOLVER F32"],
        [[name*" / "*method,nval(d[(name,"original",method)],"forward_error"),nval(d[(name,"original",method)],"backward_error"),
            passes(d[(name,"original",method)]) ? "yes" : "no",passes(d[(name,"original",method)],"backward_error") ? "yes" : "no",
            nval(d[(name,"original","cuSOLVER_F32")],"time_ms")/nval(d[(name,"original",method)],"time_ms")] for (name,method) in [("bodyy6","M4"),("nasasrb","M4"),("crankseg_2","M4")]])
    paragraph(io,"These examples move into the backward-screened fast region but fail the forward screen. Bodyy6 M4 has about 16.6% solution error despite backward error below 10⁻⁶. Nasasrb M4 has about 87.8% solution error with backward error 3.76×10⁻⁷. This is why a plot labeled only 'good enough error' would hide an important distinction.")

    print(io,"<h2 id='sources'>2. Experimental scope and provenance</h2>")
    htable(io,["Cohort","Inputs / observations","Accuracy source","Timing source and CPU baseline"],[
        ["Catalog","20 real × 10 methods = 200","Raw results.csv; eight Gaussian probes; four known RHS","Same run, median of five repetitions; sTiles 14 requested cores"],
        ["Legacy rerun","6 real × 10 = 60","Raw extended/legacy-14/results.csv; eight probes; four known RHS","Median of five, 14 requested cores; 55 OK, 4 ERROR, 1 CPU TIMEOUT"],
        ["CPU scaling","3 matrices × 5 core counts = 15","Raw extended/cores-*/results.csv; all OK","Five factor repetitions per setting, 1/2/4/8/14 requested cores; BLAS=1"],
        ["Earlier real","6 original + 6 shifted × 10 = 120","Rounded latest paired tables transcribed from supplied synthesis","Original timings transcribed from earlier synthesis; minimum GPU time; one-core sTiles; separate accuracy run"],
        ["Earlier synthetic","5 dimensions × 10 = 50","Raw accuracy_20261008_225048.csv dense rows only","Earlier synthesis timing tables; minimum GPU time; one-core sTiles; accuracy and timing run separately"]])
    print(io,"<p class='caution'><strong>Historical precision and reproducibility:</strong> the underlying latest original/shifted terminal transcript was not supplied. Its tables in the provided synthesis are available and have been transcribed with their reported precision. They are never labeled raw observations. The older CSV's 60 real rows are superseded and are not substituted for those newer paired tables. GPU timing minima from one cohort are not pooled with timing medians from another. Older one-core CPU speedups cannot establish a 14-core speedup.</p>")
    paragraph(io,"Current hardware is an NVIDIA H200 and Intel Xeon Platinum 8580; BLAS thread count is one and the main sTiles comparisons request 14 cores. The original catalog source commit is e04b604 and the extended sweep is 65cba86. Extended settings retain the input SHA-256 hashes, seed 20261029, five repeats, eight probes and four RHS. The latter worker preserves quantization diagnostics on factor failures and reduces peak allocation for large F64 cases; these are sequential runs under the same allocation, not a randomized interleaved experiment. The older synthesis identifies an H200 and sTiles cores=1 but does not record CPU model or BLAS count. The recorded CPU topology has two sockets, 60 physical cores per socket and two hardware threads per core; allocation and affinity are retained in data/extended/hardware.txt. Requested sTiles core counts do not prove every native thread ran on a dedicated physical core.")
    paragraph(io,"All-results retains 445 observations: 200 catalog + 60 rerun + 15 core sweep + 120 historical paired real + 50 synthetic. Primary-results has 370 rows: current original real (260), historical shifted (60), synthetic (50). Historical original real rows remain in all-results for the paired shift comparison; the core sweep is analyzed separately. Repeated matrices across experiments are not counted as new distinct inputs.")
    paragraph(io,"All GPU methods begin with dense storage, even when the input is sparse. Mixed timing includes SymmMixedPrec construction plus factorization; pure recursive and cuSOLVER timing covers factorization. Input conversion/transfer, reset, decoding and accuracy diagnostics are excluded. CPU sTiles comparisons use numeric factorization only; analysis is reported separately and may be reusable for repeated sparsity patterns. These figures do not measure complete end-to-end solves.")
    paragraph(io,"Mk denotes k FP16 hierarchy entries followed by FP32 leaves. The off-diagonal blocks created by recursive partitioning get lower precision; this is not simply a scalar-diagonal/off-diagonal partition. Matrix ordering and block geometry can therefore affect both error and performance. Pure_F32/F64 denote the uniform-precision recursive schedules.")

    print(io,"<h2 id='errors'>3. What each error measures—and why they disagree</h2>")
    htable(io,["Metric","Definition","Question answered","What it does not establish"],[
        ["Input quantization","|| (A_stored − A) Ω ||F / || A Ω ||F","How much damage exists before factorization?","A probe zero does not prove entrywise exactness; it says nothing about subsequent rounding."],
        ["Factor residual","|| (L Lᵀ − A) Ω ||F / || A Ω ||F","Does the decoded factor reproduce the tested input?","A small residual does not guarantee an accurate solution. sTiles factor residual is unavailable."],
        ["Forward error","max_j || x_hat_j − x_true_j ||2 / || x_true_j ||2","Did we recover the four known solutions accurately?","Not a bound for every right-hand side and not a condition-number measurement."],
        ["Equation residual","max_j || A x_hat_j − b_j ||2 / || b_j ||2","How well does the computed solution satisfy the equations?","A small equation residual need not mean a small solution error."],
        ["Frobenius-scaled backward error","max_j || A x_hat_j − b_j ||2 / (||A||F ||x_hat_j||2 + ||b_j||2)","A normwise, scaled residual diagnostic for the solve","It uses ||A||F, not ||A||2 or an infinity norm; sizes and norm choices affect interpretation."]])
    paragraph(io,"Eight fixed Gaussian probes estimate the first two quantities. They are not exact Frobenius norms or worst-case guarantees. Forward, equation and backward errors take the maximum across four known-solution RHS. GPU checks use FP64 triangular arithmetic on the decoded factor, without iterative refinement; sTiles uses its native solve. These observations evaluate the quality of the factor for FP64-assisted solves, not a native fully low-precision GPU solve pipeline.";takeaway="Use the factor metric to assess the factorization and forward error to assess recovered solutions; report both.")
    disagreements=filter(r->r["method"] in MIXED && okay(r,"factor_residual") && nval(r,"factor_residual")<=1e-3 && okay(r,"forward_error") && nval(r,"forward_error")>.1,original)
    htable(io,["Matrix","Method","Cohort","Factor residual","Forward error","Equation residual","Backward error"],
        [[r["matrix"],r["method"],r["cohort"],nval(r,"factor_residual"),nval(r,"forward_error"),nval(r,"solve_residual"),nval(r,"backward_error")] for r in disagreements])
    paragraph(io,"The table lists current original real observations with factor residual ≤10⁻³ but forward error &gt;0.1. On nd3k M1, factor residual 2.36×10⁻⁶ coexists with forward error 0.386. Current bodyy6 M6 has factor residual 1.51×10⁻⁴ but forward error 0.214; nasasrb M1 has factor residual 1.09×10⁻⁵ but forward error 1.79. Conditioning and perturbation direction are possible explanations, but κ(A) was not measured for these exact inputs. A large forward/backward ratio is not reported as κ(A). See <a href='https://nhigham.com/wp-content/uploads/2023/08/high90g.pdf'>Higham's numerical stability survey</a> for the general distinction.")
    figure(io,outdir,"25-residual-vs-forward","Factor residual versus known-solution error across original real and synthetic inputs. Historical and current cohorts are distinguished.","A small factor residual is not a sufficient solution-accuracy criterion.","The vertical spread shows that similar factor residuals can correspond to very different solution errors. Cohorts have different timing and input histories, but this plot concerns dimensionless accuracy only. Zero plotted errors are floored to 10⁻¹⁸; missing sTiles factor metrics are excluded.")
    figure(io,outdir,"26-backward-vs-forward","Frobenius-scaled backward error versus forward error.","Very small scaled backward errors can coexist with poor known-solution recovery.","The backward denominator includes ||A||F and ||x_hat||, so it can be small while forward error is appreciable. This comparison motivates retaining the raw equation residual and forward error instead of using a single scaled diagnostic as an accuracy certificate.")
    for (id,title,message) in [("02-forward-error","Catalog forward-error heatmap","The mixed schedules share ten good cases, while FP64 methods remain accurate on all 20."),
            ("03-factor-residual","Catalog factor-residual heatmap","The factor metric and forward metric separate different failure modes."),
            ("04-equation-residual","Catalog equation-residual heatmap","A small equation residual is a separate observation from accurate recovery of x."),
            ("05-backward-error","Catalog Frobenius-scaled backward-error heatmap","Backward-error colors must be interpreted using the recorded scaling convention.")]
        figure(io,outdir,id,title,message,"Rows are matrices, columns are methods; darker colors mean larger error. Colors use log10(error), clipped to [−18,0]. ERR cells are failures and do not equal zero. The per-matrix tables below retain the actual values. These heatmaps cover the 20-matrix catalog cohort only; the earlier six are included in their own dossiers.")
    end

    print(io,"<h2 id='inventory'>4. Every matrix and its properties</h2>")
    paragraph(io,"Density is 100×nnz(A)/n²; sparsity is 100−density. These refer to the full input after symmetry expansion and explicit-zero removal, not a lower-triangle storage count or factor fill. A 99.9% sparse input is only 0.1% dense. The memory column is one n×n FP64 dense allocation, not the complete solver peak.")
    htable(io,["Matrix","Kind / cohort","n","Input nnz","Density %","Sparsity %","FP64 dense GiB","Strict DD rows %","Application / construction"],
        [[r["matrix"],r["kind"]*" / "*r["cohort"],nval(r,"n"),nval(r,"nnz"),nval(r,"density_percent"),nval(r,"sparsity_percent"),nval(r,"fp64_dense_gib"),100nval(r,"dominant_fraction"),r["application"]] for r in inv])
    paragraph(io,"Synthetic inputs are the seeded symmetric random design with n added to the diagonal, fully populated by construction. This is a deliberately stabilized family, not an unmodified application matrix. Fresh rerun diagnostics now populate the six older real matrices' magnitudes and diagonal-dominance fractions. Dynamic range and diagonal dominance are diagnostics, not substitutes for eigenvalues or condition numbers. The inventory uses each real matrix's current raw original-input observation, not the historical shifted variant.")
    print(io,"<details><summary>Recorded magnitude and diagonal-dominance diagnostics</summary>")
    htable(io,["Matrix","max |Aij|","min nonzero |Aij|","Absolute dynamic range","Minimum DD margin"],
        [[r["matrix"],nval(r,"max_abs"),nval(r,"min_abs_nonzero"),nval(r,"dynamic_range"),nval(r,"worst_margin")] for r in inv]);print(io,"</details>")

    print(io,"<h2 id='failures'>5. Where runs failed and where finite results were inaccurate</h2>")
    paragraph(io,"A worker status DONE only means the process returned; individual methods may report ERROR. The catalog has 171 OK and 29 ERROR rows. The rerun adds 55 OK, four ERROR (M6/M7 on nasasrb and crankseg_2) and one TIMEOUT (animal2 sTiles), giving 226 OK, 33 ERROR and one TIMEOUT across 260 current real method rows. All 33 errors report Invalid factor diagonal. That message alone does not distinguish underflow, loss of positive definiteness, breakdown or a scale-handling defect. A timeout is a separate execution outcome, not a finite inaccurate solution.";takeaway="Separate execution failure, finite but inaccurate results, and results satisfying a declared tolerance.")
    summaryrows=Vector{Any}[]
    for (label,rr) in [("All current real (26)",original),("Catalog original (20)",catalog),("Legacy rerun (6)",legacy),("Synthetic (5)",dense)],method in METHODS
        selected=filter(r->r["method"]==method,rr)
        push!(summaryrows,[label,method,length(selected),count(r->r["status"]=="OK",selected),count(r->r["status"]=="ERROR",selected),
            count(r->r["status"]=="SKIP",selected),count(r->r["status"]=="TIMEOUT",selected),count(r->passes(r,"factor_residual"),selected),count(r->passes(r),selected),
            count(r->passes(r,"forward_error",1e-6),selected),count(r->passes(r,"forward_error",1e-2),selected),
            count(r->okay(r,"forward_error") && nval(r,"forward_error")>=.1,selected)])
    end
    htable(io,["Cohort","Method","All cases","OK","ERROR","SKIP","TIMEOUT","Factor ≤1e−3","Forward ≤1e−3","Forward ≤1e−6","Forward ≤1e−2","Forward ≥0.1"],summaryrows)
    paragraph(io,"Missing metrics cannot pass a criterion: sTiles has no reconstructed factor residual in this harness. Every coverage denominator includes all selected cases in that cohort. A forward error of 0.1 is a 10% relative vector error; 1 or more is at least 100%. Such results fail stringent solution-accuracy goals even if their factorization ran quickly. Application usefulness must ultimately be judged against an explicitly chosen requirement.")
    figure(io,outdir,"01-completion","Catalog method completion and failures.","Increasing depth reduces completed catalog factorizations from M1's 19/20 to M7's 12/20.","The reduction in completions is distinct from accuracy coverage: all mixed schedules still meet the 10⁻³ forward-error screen on the same ten matrices. No catalog timeout or memory-skip rows were recorded in this completed run.")
    failurecases=filter(r->r["status"]!="OK",original)
    htable(io,["Matrix","Method","Cohort","Status","Recorded failure","Input quantization diagnostic"],
        [[r["matrix"],r["method"],r["cohort"],r["status"],r["detail"],nval(r,"quantization_error")] for r in failurecases])
    paragraph(io,"The original catalog worker exported quantization diagnostics only after successful factorization, so its failed rows often lack that measurement. The rerun preserves it: nasasrb M6/M7 have input quantization probes 7.56×10⁻⁵/8.07×10⁻⁵; crankseg_2 M6/M7 have 2.76×10⁻⁵/2.77×10⁻⁵. Small normwise input perturbations can still precede numerical breakdown. No factor/forward errors are invented for those failed methods. Animal2's timeout log shows termination inside sTiles analysis; 300 seconds is a whole-worker cap including loading and diagnostics, not a lower bound on numeric factor time.")
    figure(io,outdir,"14-accuracy-coverage","Catalog accuracy coverage versus forward-error tolerance.","A schedule's apparent success rate depends on the requested accuracy.","The denominator stays 20, including failures. Moving the cutoff is a sensitivity analysis, not permission to choose a threshold after seeing which method wins. Use an application-defined target for a final method recommendation.")

    print(io,"<h2 id='depth'>6. Why deeper mixed precision was not always better</h2>")
    paragraph(io,"Increasing k changes the algorithm's partition as well as its storage precision. More off-diagonal blocks are assigned FP16, the FP32 leaves get smaller, and the recursion creates more blocks, allocations, conversions and kernel calls. It is therefore not a monotonic improvement knob for either accuracy or speed.";takeaway="More FP16 work can save arithmetic while the growing hierarchy cost consumes the saving.")
    geometryrows=Vector{Any}[]
    for n in (5126,17546,65536),k in (1,2,4,6,7)
        g=geometry(n,k)
        push!(geometryrows,[n,"M$k",g.leaves,g.offdiag,g.minleaf,g.maxleaf,100g.fp16_entries/(n*(n+1)/2)])
    end
    htable(io,["n","Schedule","FP32 leaves","FP16 off-diagonal blocks","Smallest leaf","Largest leaf","Stored lower-triangle entries assigned FP16 %"],geometryrows)
    paragraph(io,"These geometry counts are computed from the source's partition rule, not measured timings. For a balanced power-of-two input, M6 has 64 leaves and M7 128. The extra level roughly doubles the block count while moving only a small remaining fraction of stored entries from FP32 to FP16. These are container leaves: a large FP32 leaf can recurse again inside its factorization down to the 4,096 cutoff, so container counts are not a measured kernel count. Non-power-of-two inputs use unequal power-of-two splits and can create very small leaves. The percentage counts dense stored positions, including zeros; it is not the percentage of nonzero information quantized.")
    print(io,"<ol><li><strong>Less arithmetic benefit at small sizes.</strong> At synthetic n=4,096, M7 takes 47.321 ms versus M1's 2.640 ms and cuSOLVER F32's 1.553 ms. The extra hierarchy has no time advantage there.</li>")
    print(io,"<li><strong>Large dense work can amortize that cost.</strong> At n=65,536, M6 takes 413.606 ms versus M1's 1,308.680 ms. M7 rises to 438.556 ms: deeper still is already beyond the best observed balance.</li>")
    print(io,"<li><strong>Accuracy may worsen even if the input quantizes almost exactly.</strong> On catalog animal1, input quantization is about 8.98×10⁻¹⁶ for every mixed schedule, but factor residual rises from about 8.90×10⁻⁷ for M1 to 7.31×10⁻⁶ for M7. Arithmetic and update errors remain.</li>")
    print(io,"<li><strong>More blocks expose more data to limited precision.</strong> On gyro_m, input quantization grows from 0.0693 to 0.586 from M1 to M7. Reduced representational accuracy can dominate any speed gain.</li></ol>")
    paragraph(io,"The implementation times container construction with factorization, so the total cannot isolate which overhead dominates. Smaller GEMMs/TRSMs, unequal blocks, allocation costs and reduced GPU utilization are plausible mechanisms, not profiler-confirmed diagnoses. <a href='https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html'>NVIDIA's GEMM performance guide</a> explains why operation size and tiling affect utilization. A profile separating construction, POTRF, TRSM, SYRK and synchronization is needed to attribute this implementation's slowdown.")
    for (name,meaning) in [("dense_4096","Deep hierarchy costs dominate the smallest synthetic case."),("dense_16384","The best mixed depth has moved to M2 at the intermediate size."),("dense_65536","M6 is best observed at the largest synthetic size; M7 adds cost."),("animal2","The fresh animal2 rerun also favors M6 by runtime."),("bodyy6","The bodyy runtime optimum is M4, but its solution error must still be checked.")]
        figure(io,outdir,"22-depth-"*name,(name in OLD_NAMES ? "Current rerun median runtimes: " : "Earlier synthetic minimum runtimes: ")*name,meaning,name in OLD_NAMES ? "The real-input curve now uses current raw rerun medians. Its accuracy and runtime are from the same experiment. A low time alone does not qualify a schedule: consult the solution-error plot and tables." : "Points within this synthetic curve come from the same earlier timing table. The separate raw accuracy run determines whether each schedule is numerically useful; timings are historical minima.")
    end
    figure(io,outdir,"34-rerun-depth-forward","Current legacy rerun: known-solution error as precision depth increases.","Animal2 tolerates deeper schedules; bodyy4/5/6 lose the 10⁻³ solution pass beyond M1; nasasrb and crankseg_2 have no mixed pass.","The axis is FP16 hierarchy depth k, not a uniform percentage of low-precision arithmetic. Failure points (M6/M7 for nasasrb/crankseg_2) are absent, not zero. Bodyy6 M2 already reaches forward error 0.177 despite a factor residual near 9.61×10⁻⁵. This demonstrates why factor and solution error must both be checked.")

    print(io,"<h2 id='synthetic'>7. Synthetic dense matrices: the positive scaling case</h2>")
    dense_summary=Vector{Any}[]
    for n in (4096,8192,16384,32768,65536)
        rr=filter(r->r["matrix"]=="dense_$n",dense);best=bestmixed(rr;qualified=false)
        cu=d[("dense_$n","seeded dense + nI","cuSOLVER_F32")]
        push!(dense_summary,[n,best["method"],nval(best,"time_ms"),nval(cu,"time_ms")/nval(best,"time_ms"),
            nval(best,"factor_residual"),nval(best,"forward_error"),nval(best,"solve_residual")])
    end
    htable(io,["Dimension","Fastest mixed","Time ms","vs cuSOLVER F32","Factor residual","Forward error","Equation residual"],dense_summary)
    paragraph(io,"The synthetic family supports a specific size trend: M1 is slower than cuSOLVER at 4,096, slightly faster at 8,192, and deeper schedules become optimal as n grows. This five-point family does not define the crossover for arbitrary sparse application matrices. The added nI is part of the synthetic input design and must be disclosed alongside the accuracy result.";takeaway="Depth helps most clearly when there is enough large dense work to amortize the hierarchy and the input remains numerically favorable.")
    figure(io,outdir,"20-dense-timing","Earlier synthetic timing curves. GPU points are minima; CPU points are selected numeric factor times.","Dense GPU speed grows substantially at large n in this earlier experiment.","The sTiles curves use fully populated CSC matrices, a particularly expensive workload for a sparse solver. At 65,536, the earlier installed sTiles reports an indexing limit and has no timing point. The missing point is not an infinite measured runtime.")
    figure(io,outdir,"21-dense-forward","Raw synthetic known-solution errors across sizes.","The observed mixed speedups retain small solution errors on this stabilized synthetic family.","The plotted errors come from the raw dense accuracy CSV, independently from the earlier timing table. Differences between methods remain visible even where every method passes the exploratory 10⁻³ screen. sTiles at 65,536 is skipped rather than assigned an error of zero.")
    paragraph(io,"The 65,536 lower triangle has n(n+1)/2 = 2,147,516,416 positions, exceeding a signed 32-bit index maximum by 32,769. The earlier report attributes the installed sTiles limit to that indexing path. Adding CPU threads would not remove an index-width limit. This is a version-specific observed restriction, not a claim that all sparse Cholesky software cannot solve such a matrix.")

    print(io,"<h2 id='quantization'>8. Where input quantization works and where it does not</h2>")
    qgroups=[("Negligible probe (≤1e−12)",x->x<=1e-12),("Small/intermediate (1e−12,1e−3]",x->1e-12<x<=1e-3),("Large probe (>1e−3)",x->x>1e-3)]
    qtable=Vector{Any}[]
    for cohort in ("Catalog / 14-core CPU","Legacy rerun / 14-core CPU"),(label,predicate) in qgroups
        rr=filter(r->r["cohort"]==cohort && r["method"] in MIXED && okay(r,"quantization_error") && predicate(nval(r,"quantization_error")),original)
        push!(qtable,[cohort,label,length(rr),count(r->passes(r,"factor_residual"),rr),count(r->passes(r),rr),
            join(sort(unique(getindex.(rr,"matrix"))),", ")])
    end
    htable(io,["Original-real cohort","Input quantization group","Recorded OK cases","Factor passes 1e−3","Forward passes 1e−3","Matrices represented"],qtable)
    paragraph(io,"The catalog's ten mixed-accurate cases have quantization probes near the FP64 computation floor, as does current animal2 (8.91×10⁻¹⁷ for every mixed schedule). This suggests many important stored values are nearly representable, but probes do not establish entrywise exactness. Bodyy is different: mixed quantization is nonzero and increases with depth (bodyy6 M1 5.28×10⁻⁷; M7 1.24×10⁻⁴). Bodyy6's forward error then grows to 0.224 despite a factor residual of only 1.64×10⁻⁴. Gyro_m exhibits large direct representation damage. Nasasrb's finite shallow factor residuals are small but its solution errors are large, while its deep methods fail. These are different numerical regimes.";takeaway="Distinguish large representation damage, subsequent arithmetic error, and sensitivity of the recovered solution.")
    figure(io,outdir,"11-storage-error","Catalog input quantization versus estimated factor residual for mixed methods.","Points near the diagonal suggest input storage is an important contributor; points far above it indicate additional error.","On gyro_m M1, quantization and factor residual are both about 0.0693; by M7 both are about 0.586. animal1 lies far above its near-zero input probe but still has small factor residuals in absolute terms. The diagonal is an equal-error reference, not a theorem that decomposes or bounds every error source. Failures with missing diagnostics are omitted.")
    paragraph(io,"Per-block scaling in the source activates when a block maximum exceeds the FP16 finite bound, 65,504. Scaling avoids one overflow mechanism; it does not increase the significand precision or guarantee that very small values survive. The fixed source no longer performs the earlier extra integer rounding before the Float16 conversion. Remaining failures should be investigated rather than automatically attributed to that previously fixed bug. Dynamic range of the entire matrix also differs from the range within each recursively assigned block.")
    figure(io,outdir,"12-dynamic-range","Catalog mixed-schedule solution error versus global absolute-value dynamic range.","Global dynamic range alone does not identify a safe precision schedule.","This is an association plot, not a measurement of κ(A). A tiny isolated nonzero can make max|Aij|/min_nonzero|Aij| enormous while contributing little to a normwise probe. The assigned block ranges, eigenvalue sensitivity and arithmetic path are not summarized by this one scalar.")
    figure(io,outdir,"13-diagonal-dominance","Catalog mixed-schedule solution error versus the fraction of strictly diagonally dominant rows.","Diagonal dominance is a useful diagnostic, not a complete predictor of mixed-precision performance.","The fraction refers to the original input. It does not report the spectrum of the quantized input or every Schur complement. The earlier bodyy and animal2 observations motivate further tests, while the shifting experiment below shows a controlled change in both the problem and relative-error denominators.")

    print(io,"<h2 id='crossover'>9. When are we faster than sTiles? Size and percent sparsity</h2>")
    paragraph(io,"There is no measured universal size or sparsity boundary in these data. The most defensible answer is an empirical list of wins and losses, stratified by CPU cohort and by accuracy requirement. Sparse factorization cost also depends on ordering, elimination fill and supernodal structure, none of which is recorded here.";takeaway="Dimension and input sparsity are informative covariates; they are insufficient to define a universal CPU/GPU crossover.")
    comparison=comparison_table(rows,inv)
    htable(io,["Matrix","CPU cohort","n","Density %","Sparsity %","Fastest mixed by time","Fastest mixed meeting forward ≤1e−3","Its forward error","vs sTiles factor","vs cuSOLVER F32"],comparison)
    paragraph(io,"Across the 26 current original inputs, 14 have an accuracy-qualified mixed schedule. Four have a measured numeric-factor win against sTiles: animal1, lgm_10010_bw2, lidense and spacetime. Nine eligible inputs lose to sTiles, and one (animal2) has a CPU timeout with no comparable factor time. The remaining 12 have no mixed forward-error pass. Thus the result is four measured wins out of all 26 selected inputs, with a separate unresolved CPU comparison; among eligible inputs with completed CPU timings it is 4/13. Bodyy4/5/6 qualify only at M1, whose times are 62.99/70.36/74.50 ms versus CPU 16.52/8.53/15.88 ms. Nasasrb and crankseg_2 have no mixed solution pass at 10⁻³.")
    for (cohort,rr) in [("Catalog / 14-core CPU",catalog),("Legacy rerun / 14-core CPU",legacy)]
        names=unique(getindex.(rr,"matrix"));binrows=Vector{Any}[]
        for (label,lo,hi) in [("≤0.1% density",0.0,.1),("(0.1,1]% density",.1,1.0),("(1,5]% density",1.0,5.0),(">5% density",5.0,101.0)]
            metas=filter(m->m["matrix"] in names && lo<100nval(m,"density")<=hi,inv)
            eligible,wins=0,0
            for m in metas
                subset=filter(r->r["matrix"]==m["matrix"],rr);best=bestmixed(subset);cpu=d[(m["matrix"],"original","sTiles_F64")]
                if best!==nothing
                    eligible+=1;validtime(cpu) && nval(cpu,"time_ms")>nval(best,"time_ms") && (wins+=1)
                end
            end
            push!(binrows,[label,length(metas),eligible,wins])
        end
        print(io,"<h3>",esc(cohort),": density bins</h3>");htable(io,["Input density bin","All matrices","Have a mixed accuracy pass","Eligible mixed beats sTiles"],binrows)
    end
    paragraph(io,"The observed 14-core mixed wins occur at density 20.109% (animal1, n=5,126), 22.095% (lidense, n=12,509), 0.6434% (lgm_10010_bw2, n=10,010) and 0.2033% (spacetime, n=24,931). Their sparsities are about 79.891%, 77.905%, 99.357% and 99.797%. However, other matrices of similar or greater density either lose to sTiles or lack an acceptable mixed solution. This prevents inferring a single density cutoff from the winning points.")
    paragraph(io,"The largest catalog input, nd12k at n=36,000, has no mixed forward-error pass, while much smaller animal1 is a large CPU/GPU timing win. Current animal2 at n=50,126 and about 20.01% density has a 2.63× GPU-library gain, but its CPU comparison is unresolved. Larger n can favor dense throughput while simultaneously increasing the need to check numerical suitability and sparse structure. Historical animal2's one-core factor time is retained as historical evidence only; it is not substituted for the current timeout.")
    figure(io,outdir,"27-density-crossing","Accuracy-qualified best mixed versus 14-core sTiles across current real inputs, by density.","Measured wins span very different sparsities; density alone does not identify a safe crossover.","Above one means mixed is faster. Only the 13 matrices with both a qualified mixed method and a completed CPU factor timing appear: four wins and nine losses. Twelve have no eligible mixed method and animal2 has a CPU timeout; these 13 omissions are not failures hidden as zeros. Colors distinguish the sequential catalog and rerun experiments; both use medians and 14 requested CPU cores.")
    figure(io,outdir,"28-size-crossing","The same current accuracy-qualified CPU comparison by matrix dimension.","Size alone does not explain the wins and failures.","The chart uses the fastest observed qualifying schedule on each input, an oracle selection for this dataset rather than a validated adaptive rule. Animal2 has no numeric CPU point because its worker timed out in analysis. All 26 raw-input outcomes remain in the comparison table.")
    for (id,caption,take,body) in [
        ("08a-tradeoff-cuSOLVER_F32","Catalog accuracy versus cuSOLVER F32 speedup","Mixed precision has selective GPU speedups with an accuracy cost.","Upper left is preferable: lower known-solution error and greater runtime gain. The dashed line is equal time. A fast point far to the right can be unusable for accurate solves. Each point represents a matrix/method, not a confidence interval."),
        ("08b-tradeoff-cuSOLVER_F64","Catalog accuracy versus cuSOLVER F64 speedup","FP64 is a meaningful baseline on this H200 run.","The available H200 F64 timings are often close to F32, so a comparison only against F32 hides an important accuracy/performance option. Inspect each matrix's F64 result and the exclusion of input transfer before choosing a mixed schedule."),
        ("08c-tradeoff-sTiles_F64","Catalog accuracy versus 14-core sTiles speedup","A GPU gain over cuSOLVER does not guarantee a gain over a sparse CPU solver.","On sem_n5000, M4 takes about 123.5 ms, cuSOLVER F32 220.7 ms and sTiles 6.20 ms. The GPU-library speedup and CPU loss are both true. Accuracy thresholds and sparse/dense scopes remain essential."),
        ("09-size-trend","Catalog GPU speedup versus dimension","The observed runtime trend has both size effects and matrix-specific exceptions.","This scatter does not filter by the accuracy cutoff. Consult the forward-error heatmap and matrix tables before interpreting a large speedup as an acceptable method. The baseline is cuSOLVER F32, not sTiles."),
        ("10-density-trend","Catalog GPU speedup versus input density","Input density is not the same as GPU work density.","All GPU kernels operate on dense storage, so the pattern does not measure a GPU sparse algorithm. Accuracy, block geometry and size co-vary with the original matrix density."),
        ("06-runtime","Catalog runtime heatmap","Absolute milliseconds are needed alongside speedup ratios.","A twofold ratio on a small matrix and a twofold ratio on a large matrix have different practical impact. Only status-OK timings are colored. Mixed construction is included and CPU analysis is excluded."),
        ("07a-speedup-cuSOLVER_F32","Catalog per-matrix speedup against cuSOLVER F32","Runtime improvements are uneven across schedules and inputs.","Blue means faster, orange slower and white equal. The scale is log2(speedup): +1 means 2× faster and −1 means 2× slower. Failed and unavailable pairs do not have a valid ratio."),
        ("07b-speedup-cuSOLVER_F64","Catalog per-matrix speedup against cuSOLVER F64","Compare against a high-accuracy GPU baseline as well.","The same color interpretation applies. Accuracy-qualified recommendations require both this runtime information and forward/factor error requirements; a blue cell alone is not a recommendation."),
        ("07c-speedup-sTiles_F64","Catalog per-matrix speedup against 14-core sTiles","Sparse CPU performance creates a different winning set.","Ratios use CPU numeric factor time and the same matrix's method median. sTiles analysis is displayed separately below, so this is appropriate to a numeric-factor comparison, not an end-to-end workflow guarantee.")]
        figure(io,outdir,id,caption,take,body)
    end
    figure(io,outdir,"15-stiles-phases","Catalog CPU analysis versus numeric factorization cost.","Including or reusing analysis can materially change the relevant comparison.","Shares come from the repetition with minimum combined analysis+factor time, rather than the separate medians used in the speedup plots. For a one-off factorization, include analysis; for repeated values with a reusable sparsity pattern, numeric factorization may be the relevant cost. This report provides the separate repeated phase fields rather than assuming one scenario.")

    print(io,"<h2 id='scaling'>CPU core scaling: the completed controlled sweep</h2>")
    paragraph(io,"The sweep reruns the same three inputs under one allocation, requesting 14, 1, 8, 2 and 4 sTiles cores in that order, with BLAS threads fixed at one. Each setting records five numeric-factor and five analysis timings. Factor speedup is T₁/Tₚ using medians; requested-core efficiency is (T₁/Tₚ)/p. These are observed summaries of repeated operations in one experiment, not confidence intervals or five independently allocated jobs.";takeaway="Choose CPU parallelism for the workload: 14 cores is not uniformly the strongest baseline.")
    htable(io,["Matrix","Requested cores","Median factor ms","Observed repeat min ms","Observed repeat max ms","Factor speedup T1/Tp","Efficiency %","Selected analysis ms"],
        [[r["matrix"],nval(r,"stiles_cores"),nval(r,"time_ms"),nval(r,"repeat_min"),nval(r,"repeat_max"),nval(r,"speedup"),100nval(r,"efficiency"),nval(r,"analysis_ms")] for r in scaling])
    paragraph(io,"Animal1 improves from 415.60 to 82.72 ms at 14 cores (5.02×, 35.9% requested-core efficiency). Spacetime improves from 467.30 to 131.93 ms at 8 cores (3.54×); 14 gives 138.74 ms (3.37×), so 8 is fastest by the measured median. Sem_n5000 goes from 5.03 ms at one to 2.46 ms at two (2.05×); at 14 it takes 6.00 ms, 1.19× slower than one and 2.44× slower than its best observed setting. Slightly superlinear two-core speedup is an observation, not evidence of a general scaling law.")
    figure(io,outdir,"31-core-runtime","sTiles factor runtime versus requested cores for the three representative inputs.","Large factor workloads benefit from more parallelism; the tiny sem_n5000 factor saturates early.","Lower is better; the runtime axis is logarithmic so the 2–6 ms sem_n5000 series is visible alongside the 80–470 ms cases. All five settings succeeded with small solution errors. Differences can reflect workload structure, overhead, affinity and run variability; no kernel profile establishes a single cause.")
    figure(io,outdir,"32-core-speedup","Measured factor speedup relative to one requested core.","Efficiency falls as requested parallelism increases, and speedup can reverse.","Every series starts at one by definition. Divide the plotted speedup by requested cores to obtain the efficiency in the table. The core counts are sTiles requests within a 14-CPU allocation. Hardware threads, physical cores, actual native-thread utilization and per-setting affinity were not independently measured, so these curves are not proof of isolated physical-core scaling.")
    figure(io,outdir,"33-core-analysis","Recorded sTiles analysis phase versus requested cores.","Analysis remains expensive and does not track numeric factor speedup.","The chart uses analysis_ms: the analysis time from the repetition having the minimum combined analysis+factor time, not the analysis median. Raw analysis_repetition_ms is retained for all five repetitions. Animal1 stays near 1.47–1.51 s and spacetime near 2.04–2.18 s, much larger than their best factor times. For a fresh matrix pattern, include analysis; for a reusable pattern, a factor-only comparison answers a different practical question.")
    paragraph(io,"The repeated 14-core factor medians differ from the earlier catalog: animal1 82.72 versus 68.11 ms, spacetime 138.74 versus 103.31 ms, and sem_n5000 6.00 versus 6.20 ms. Those sequential differences caution against reading a small CPU/GPU margin as stable. Main catalog ratios retain their same-run CPU reference rather than replacing it with the later measurement. GPU work was not rerun at each CPU core setting, so no synchronized GPU-versus-every-core-count experiment is claimed.")

    print(io,"<h2 id='shifting'>10. Effects of diagonal shifting</h2>")
    paragraph(io,"<strong>Pairing rule:</strong> this entire chapter compares the historical original and shifted observations from the same supplied synthesis. It does not compare a fresh rerun original against an older shifted result. The new experiment reran only unmodified inputs; it supplies no new shifted timings or errors. Historical original tables remain available below each of the six matrix dossiers for this reason.")
    paragraph(io,"The paired earlier experiment tests A and A+sI with s at least n and large enough to enforce a positive strictly diagonally dominant diagonal according to the logged formula. It forms b from the matrix actually being tested in each mode. This is a different linear system, not a storage-only rescaling or a solution of the original system through refinement.";takeaway="Shifting substantially improves several results, but it changes both the problem and the interpretation of relative errors.")
    shiftrows=Vector{Any}[]
    for (j,name) in enumerate(OLD_NAMES)
        a,b=paired[(name,"original","M6")],paired[(name,"shifted","M6")]
        push!(shiftrows,[name,SHIFTS[j],nval(a,"quantization_error"),nval(b,"quantization_error"),nval(a,"factor_residual"),nval(b,"factor_residual"),
            nval(a,"forward_error"),nval(b,"forward_error"),nval(a,"factor_residual")/nval(b,"factor_residual")])
    end
    htable(io,["Matrix / M6","Added shift s","Input quant. original","Input quant. shifted","Factor original","Factor shifted","Forward original","Forward shifted","Factor-error reduction ×"],shiftrows)
    for (id,caption,take) in [("23-shift-factor","Paired M6 factor residual before and after shifting","Relative factor residuals decrease on all six paired inputs."),
            ("24-shift-forward","Paired M6 solution forward error before and after shifting","Improved factor residual and improved recovered solution are related but separately measured.")]
        figure(io,outdir,id,caption,take,"The cases share probes and known-solution design within the earlier paired experiment. Values are transcribed from the supplied rounded tables. An absent or failed observation is never drawn as zero. The diagonal shift can alter eigenvalue separation as well as the denominator ||AΩ||F.")
    end
    paragraph(io,"For M6, nasasrb's factor residual falls from 0.272 to 0.00257 and crankseg_2's from 0.793 to 0.000625. These are large improvements, but shifted nasasrb M6 still exceeds a 10⁻³ factor screen. Shallow shifted schedules can satisfy a stricter criterion than deeper ones. Original nasasrb M7 fails its diagonal check; its shifted counterpart completes, which is a change in robustness on a changed problem.")
    shiftedcoverage=Vector{Any}[]
    for method in MIXED,metric in ("factor_residual","forward_error")
        before=[paired[(name,"original",method)] for name in OLD_NAMES];after=[paired[(name,"shifted",method)] for name in OLD_NAMES]
        push!(shiftedcoverage,[method,metric,count(r->passes(r,metric),before),count(r->passes(r,metric),after),6])
    end
    htable(io,["Method","Criterion ≤1e−3","Original passes","Shifted passes","All cases"],shiftedcoverage)
    paragraph(io,"A+sI has eigenvalues λᵢ(A)+s for a symmetric A. For SPD A and positive s this moves the spectral condition ratio toward one, but no eigenvalues or run-specific condition numbers were measured here. A smaller relative residual can also reflect a larger normalization denominator; absolute residuals were not exported. Global normalization A/α is a different operation and does not have the same spectral effect. Old 16-probe normalization/raw-storage diagnostics are retained only in the source document and are not substituted for the paired physical-factor results. <a href='https://eprints.maths.manchester.ac.uk/2771/1/paper.pdf'>Research on lower-precision SPD solves</a> provides relevant theory, not an empirical guarantee for these runs.")
    paragraph(io,"A further distinction matters for quantization: adding sI changes the diagonal, while the original off-diagonal entries and their recursive FP16 block locations stay the same. A reduced relative input-quantization probe after shifting therefore need not mean the off-diagonal values were stored more accurately. Changes in the denominator and any FP32 diagonal rounding also contribute. In exact arithmetic, scaling the whole input by a nonzero scalar preserves its spectral condition number; it can still change finite-precision overflow, underflow and block-scale activation. These mechanisms should be measured separately.")

    print(io,"<h2 id='dossiers'>11. Matrix-by-matrix results and interpretation</h2>")
    paragraph(io,"Every input is listed below, including earlier animal2, bodyy4/5/6, nasasrb, crankseg_2 and all five synthetic dimensions. Results are original-input observations unless the table is explicitly labeled shifted. Times retain their cohort's statistic. Missing and failed fields remain unavailable rather than filled with a fabricated value.")
    cols=["Method","Status","Time ms","Input quant.","Factor residual","Forward error","Equation residual","Backward error","vs sTiles factor"]
    special=Dict(
        "gyro_m"=>"The input quantization error closely tracks factor residual. M1 to M7 raises quantization from about 0.0693 to 0.586 and forward error from 0.110 to 0.961. This is a strong representation-damage case; ordinary FP32 and FP64 are essential controls.",
        "nd3k"=>"M1 has a relatively small factor residual but forward error about 0.386; M2 and deeper schedules fail. cuSOLVER F32 also has forward error about 0.0161, so this input is already sensitive at uniform single precision. F64 controls are much more accurate.",
        "inla_graph_animal1"=>"All mixed schedules meet the exploratory solution cutoff, with negligible input quantization. M1 is the fastest mixed option. Increasing depth adds time and raises arithmetic/factor error. The CPU comparison is favorable, but cuSOLVER F32 is faster than M1.",
        "inla_graph_spacetime"=>"M4 is a useful measured tradeoff: about 96.1 ms, forward error 1.06×10⁻⁶, 1.36× faster than cuSOLVER F32 and 1.08× faster than 14-core sTiles. Its margin over sTiles is modest and should be checked with repeated experiments.",
        "inla_graph_sem_n5000"=>"M4 improves on the dense GPU baseline while sTiles remains much faster. This is the clearest example of why a cuSOLVER speedup does not imply a CPU sparse speedup.",
        "inla_graph_stcov"=>"The INLA name alone does not predict a safe mixed schedule. This input is outside the ten catalog mixed forward-error passes, despite the success of many other INLA cases. Inspect the metric table instead of generalizing from the family name.",
        "msc23052"=>"The uniform F32 recursive and cuSOLVER F32 cases fail here along with all mixed schedules. The F64 methods complete with small solution errors. This supports investigating low-precision sensitivity, while the specific breakdown mechanism still requires diagnostics.",
        "animal2"=>"Current M6 is the fastest mixed schedule at 357.483 ms: 2.63× faster than cuSOLVER F32 (939.479 ms), factor residual 1.94×10⁻⁶ and forward error 3.24×10⁻⁶. All five mixed schedules pass 10⁻³. The sTiles worker timed out after 300 seconds while in analysis; no current numeric-factor speedup is available. Historical CPU timing is not used to fill that gap.",
        "nasasrb"=>"Current M1/M2/M4 complete with factor residuals 1.09×10⁻⁵, 1.81×10⁻⁵ and 9.00×10⁻⁵, yet forward errors 1.79, 1.04 and 0.878. M6/M7 fail. Uniform F32 also has about 1.1% forward error; F64 and sTiles are accurate. The earlier M6 was reported finite with factor residual 0.272, so the fresh run changes the observed breakdown boundary. Exact historical source/input hashes are unavailable; the report does not identify a proven cause.",
        "crankseg_2"=>"Current M1/M2/M4 complete with factor residuals around 2–4×10⁻⁵ but forward errors 0.00696–0.00966, above 10⁻³. M6/M7 fail. The fastest completed mixed schedule M4 is about 2.73× faster than cuSOLVER F32, but is neither accurate enough for this screen nor faster than sTiles. Historical M6's finite large factor error is a different experiment, not substituted for this failure.")
    for meta in inv
        name=meta["matrix"];policy=meta["kind"]=="synthetic" ? "seeded dense + nI" : "original"
        rr=[d[(name,policy,m)] for m in METHODS]
        print(io,"<section class='matrix'><h3>",esc(name)," <span class='badge'>",esc(meta["cohort"]),"</span></h3>")
        paragraph(io,"n="*fnum(nval(meta,"n"))*"; input nnz="*fnum(nval(meta,"nnz"))*"; density="*fnum(nval(meta,"density_percent"))*"%; sparsity="*fnum(nval(meta,"sparsity_percent"))*"%; one FP64 dense allocation="*fnum(nval(meta,"fp64_dense_gib"))*" GiB. "*esc(meta["application"])*".")
        mixed=filter(r->r["method"] in MIXED,rr);best=bestmixed(rr);fast=bestmixed(rr;qualified=false)
        complete=count(r->r["status"]=="OK",mixed);factorpass=count(r->passes(r,"factor_residual"),mixed);forwardpass=count(r->passes(r),mixed)
        body="Mixed schedules: $complete/5 complete; $factorpass/5 meet factor residual ≤10⁻³; $forwardpass/5 meet forward error ≤10⁻³. "
        if best===nothing
            body*="No recorded mixed schedule meets the solution criterion with a valid timing. "
        else
            label,ms=esc(best["method"]),fnum(nval(best,"time_ms"))
            body*="Fastest observed solution-qualified mixed schedule: <strong>$label</strong> ($ms ms). "
        end
        if haskey(special,name);body*=special[name]
        elseif startswith(name,"bodyy")
            body*="Current M4 is fastest among mixed schedules by time, but only M1 meets the 10⁻³ solution screen. Fourteen-core sTiles numeric factorization is faster than every mixed schedule. Deep methods have modest factor residuals but appreciably larger solution error. Historical shifted observations are shown as a separate paired experiment below."
        elseif meta["kind"]=="synthetic"
            body*="This stabilized dense design illustrates depth amortization with size. Its timing-optimal schedule is not automatically optimal for an arbitrary real matrix of the same dimension."
        elseif all(r->passes(r),mixed)
            body*="All mixed schedules pass the exploratory solution cutoff here. Compare the shallow and deeper times, the CPU sparse baseline and the high-accuracy F64 alternatives before selecting a schedule."
        else
            body*="At least one mixed observation fails or exceeds the solution cutoff. Factor and solve metrics in the table distinguish representation damage, numerical factor error and solution sensitivity; they do not identify a root cause by themselves."
        end
        paragraph(io,body;takeaway=best===nothing ? "Mixed precision does not meet the illustrative solution-accuracy screen on this original input." : "A mixed solution-accuracy pass exists; compare runtime against both GPU and CPU baselines.")
        cpu=d[(name,policy,"sTiles_F64")]
        function metricrecords(cases,reference)
            [[r["method"],r["status"],nval(r,"time_ms"),nval(r,"quantization_error"),nval(r,"factor_residual"),nval(r,"forward_error"),nval(r,"solve_residual"),nval(r,"backward_error"),
                validtime(r)&&validtime(reference) ? nval(reference,"time_ms")/nval(r,"time_ms") : NaN] for r in cases]
        end
        htable(io,cols,metricrecords(rr,cpu))
        if name in OLD_NAMES
            oldoriginal=[paired[(name,"original",m)] for m in METHODS]
            print(io,"<details><summary>Historical original input — reference for the historical shifted pair</summary>")
            paragraph(io,"These are rounded report transcriptions, with earlier GPU minimum times and a one-core CPU baseline. They are preserved to explain the earlier paired experiment and are not the main current results above. Differences across the two runs can involve code, probe/RHS seeds, settings and timing statistic; historical input hashes and complete source records are unavailable, so no single cause is asserted.")
            htable(io,cols,metricrecords(oldoriginal,last(oldoriginal)));print(io,"</details>")
            shifted=[paired[(name,"shifted",m)] for m in METHODS]
            print(io,"<details><summary>Historical paired shifted input: A + ",fnum(nval(first(shifted),"shift"))," I</summary>")
            paragraph(io,"Compare these values with the historical original table immediately above, not with the fresh rerun. Matching shifted timings are unavailable and remain blank.")
            htable(io,cols,metricrecords(shifted,last(shifted)));print(io,"</details>")
        end
        print(io,"</section>")
    end

    print(io,"<h2 id='next'>12. Reruns and additional measurements that would resolve the uncertainty</h2>")
    print(io,"<ol><li><strong>Resolve animal2 CPU analysis.</strong> The matched six-input rerun is complete, with input hashes and raw measurements. Animal2 alone lacks a CPU factor timing because its worker times out in analysis. If this comparison is essential, run sTiles alone with a deliberately longer wall-time cap and retain separate analysis and factor progress. Do not treat 300 seconds as a factor-time lower bound.</li>")
    print(io,"<li><strong>Repeat the CPU sweep and modest crossover margins.</strong> The first 1/2/4/8/14 sweep is complete. Repeat in randomized or interleaved order, record per-setting affinity and actual thread utilization, and rerun GPU controls near comparisons such as spacetime. Sem_n5000 is fastest at two requested cores; use measured workload-specific CPU configurations when presenting a best-tuned CPU baseline.</li>")
    print(io,"<li><strong>Preserve diagnostics on failure.</strong> Retain input quantization and scale statistics even if POTRF fails, record the failing stage and any solver info code, and distinguish nonfinite data, a nonpositive diagonal and allocation failures. The updated worker preserves pre-factorization diagnostics through ordinary exceptions.</li>")
    print(io,"<li><strong>Profile the cost of hierarchy.</strong> Measure construction, conversions, POTRF leaves, TRSM, SYRK and synchronization separately without replacing the existing total timing. Record block shapes and kernel counts to test the overhead hypothesis.</li>")
    print(io,"<li><strong>Quantization diagnostics inside assigned blocks.</strong> Count underflow-to-zero, saturation, scale activations, per-block ranges and changes in off-diagonal nonzeros. Check a small controlled scaling-active SPD case against an exact reconstructed residual, including triangular and update stages. Passing one small test does not establish large-matrix stability.</li>")
    print(io,"<li><strong>Numerical sensitivity and sparse structure.</strong> Measure or estimate spectral conditioning when feasible; record sparse ordering, symbolic/factor nonzeros and fill ratio. Compare absolute as well as relative residuals before/after shifting. Test iterative refinement as a separate algorithm, retaining the unrefined results.</li>")
    print(io,"<li><strong>A defensible schedule policy.</strong> Select an accuracy target before tuning; compare candidate depths on common matrices, retain failures, and validate any size/density/quantization rule on held-out matrices. A per-matrix best observed schedule is an oracle description of this dataset, not a deployed adaptive policy.</li></ol>")
    paragraph(io,"The completed extended experiment is included throughout this report. It supplies new original-input measurements and actual CPU scaling, but not new shifted or synthetic runs. A controlled fresh original/shifted pair would help explain the changed nasasrb/crankseg_2 depth-failure boundary relative to the older rounded report; retain the same probes, RHS and exact source/input hashes for that comparison.")

    print(io,"<h2 id='conclusion'>13. Overall conclusion</h2>")
    paragraph(io,"The evidence supports mixed precision as a selective method whose useful depth depends on workload size, block geometry, numerical sensitivity and the comparison baseline. The largest historical stabilized dense problem and current animal2 show substantial GPU-library gains with small measured errors. The catalog and legacy rerun also show many inputs on which deeper mixed precision becomes less reliable or slower. These observations describe distinct operating regimes.";takeaway="Choose precision depth using both numerical eligibility and measured total cost; depth alone is not a quality or speed guarantee.")
    paragraph(io,"For accuracy, the central distinction is between damaging the stored input, producing an inaccurate factor, and recovering an inaccurate solution. All three can matter. On some inputs negligible quantization still leaves arithmetic error; on gyro_m representation error is large; on nd3k a small factor residual coexists with substantial forward error. Numerical breakdowns and finite inaccurate cases must remain in the analysis.")
    paragraph(io,"For performance, there is no justified universal sparsity percentage or dimension threshold. Current animal2 confirms a 2.63× GPU-library gain with small forward error, while its current CPU factor comparison remains unresolved. Four current real inputs have measured accuracy-qualified wins against the recorded 14-core CPU baseline. The controlled core sweep shows that best CPU settings differ by workload: two for sem_n5000, eight for spacetime and 14 among the tested settings for animal1. Earlier one-core ratios are no longer used for the main current real comparison.")
    paragraph(io,"Shifting is a valuable controlled sensitivity experiment and can make many measurements better. It solves a changed system, however, and a smaller relative error can partly reflect the changed denominator. A practical next result would be a matched, accuracy-qualified schedule comparison supported by actual core scaling, kernel-stage profiles, scale-aware diagnostics on failed inputs and a validated precision-selection policy.")
    print(io,"<h3>Data, source records and audit trail</h3><p>This HTML embeds all displayed figures and works offline. data/all-results.csv retains all 445 normalized observations and per-cell source labels; data/primary-results.csv contains 370 selected observations without duplicating historical originals; data/cpu-scaling.csv includes the 15 measurements and derived ratios. data/matrix-inventory.csv contains 31 distinct inputs. data/extended contains the unchanged six raw result CSVs, settings, worker logs, per-case outputs and hardware record. chart-data holds numerical figure data. Original matrix files, factors, the missing historical raw transcript and a full historical Manifest are not bundled.</p>")
    hashes=[[name,bytes2hex(sha256(read(joinpath(outdir,"data",name))))] for name in ["catalog-results-original.csv","earlier-accuracy-original.csv","earlier-synthesis.txt","extended/legacy-14/results.csv","extended/cores-1/results.csv","extended/cores-2/results.csv","extended/cores-4/results.csv","extended/cores-8/results.csv","extended/cores-14/results.csv","all-results.csv","primary-results.csv"]]
    htable(io,["Evidence file","SHA-256"],hashes)
    print(io,"<p class='sources'>Theory and implementation references: <a href='https://nhigham.com/wp-content/uploads/2023/08/high90g.pdf'>Higham, numerical stability and forward/backward error</a>; <a href='https://eprints.maths.manchester.ac.uk/2771/1/paper.pdf'>lower-precision SPD solves</a>; <a href='https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html'>NVIDIA matrix multiplication performance guide</a>. Source-code interpretation uses src/symmmixedprec.jl, src/cholesky_tree_subdiv.jl and test/catalog/worker.jl. Theoretical explanations are distinguished from unprofiled implementation hypotheses throughout.</p>")
    print(io,"<p class='small'>Zero errors are floored only for log plotting. Table dashes mean unavailable, not zero. Synthetic sTiles n=65,536 is skipped. Unknown condition numbers, confidence intervals and absolute shift residuals are not inferred.</p>")
    if interactive
        print(io,"""<script>Promise.all(window.reportChartJobs).then(async()=>{
        document.documentElement.dataset.chartsReady='true';
        if(new URLSearchParams(location.search).get('export')==='1'){
          const images=[];
          for(const c of window.reportCharts){images.push({id:c.id,url:await Plotly.toImage(c.g,{format:'png',width:1200,height:680,scale:1.5})});}
          const node=document.createElement('script');node.id='exported-charts';node.type='application/json';node.textContent=JSON.stringify(images);document.body.appendChild(node);
          document.documentElement.dataset.exportsReady='true';
        }
        }).catch(e=>{document.documentElement.dataset.chartError=String(e)});</script>""")
    end
    print(io,"</main></body></html>")
    path=joinpath(outdir,"report.html");write(path,order_report(String(take!(io))))
    println("Comprehensive report: ",abspath(path))
end
