# Comprehensive cross-experiment analysis. Pure Julia standard libraries.
# --prepare: normalize sources and export plotting datasets.
# --html: assemble an offline report with existing PNGs and supplemental Plotly charts.
module ComprehensiveCholesky
include("common.jl")
using .CatalogSweep, Statistics, Printf, SHA, Base64, TOML, Dates, Downloads
const METHODS = vcat(CatalogSweep.GPU_METHODS,["sTiles_F64"])
const MIXED = ["M1","M2","M4","M6","M7"]
const OLD_NAMES = ["nasasrb","crankseg_2","bodyy4","bodyy5","bodyy6","animal2"]
const OLD_N = [54870,63838,17546,18589,19366,50126]
const OLD_NNZ = [2677324,14148858,121550,128853,134208,502803876]
const SHIFTS = [2381559229.462728,2771136879.892498,17546.0,18589.0,19366.0,50126.0]
const METRICS = ["quantization_error","factor_residual","forward_error","solve_residual","backward_error"]
const FIELDS = ["cohort","kind","matrix","n","policy","shift","method","status","time_ms","timing_statistic",
    "min_ms","median_ms","analysis_ms","factor_ms","stiles_cores","nnz","density","application","dominant_fraction",
    "max_abs","min_abs_nonzero","worst_margin","scaled_blocks","max_scale","quantization_error","factor_residual",
    "forward_error","solve_residual","backward_error","detail","accuracy_source","timing_source","source_precision",
    "repetition_ms","analysis_repetition_ms","git_commit","source_hash","matrix_sha256","probes","nrhs","repeats","seed","blas_threads"]
nval(r,k) = something(tryparse(Float64,get(r,k,"")),NaN)
fnum(x) = x isa Number ? (isfinite(x) ? (isinteger(x) && abs(x)<1e12 ? @sprintf("%.0f",x) : @sprintf("%.4g",x)) : "—") : string(x)
esc(x) = replace(string(x),'&'=>"&amp;",'<'=>"&lt;",'>'=>"&gt;",'"'=>"&quot;",'\''=>"&#39;")
okay(r,k) = r !== nothing && get(r,"status","")=="OK" && isfinite(nval(r,k)) && nval(r,k)>=0
passes(r,metric="forward_error",tol=1e-3) = okay(r,metric) && nval(r,metric)<=tol
validtime(r) = okay(r,"time_ms") && nval(r,"time_ms")>0
canonical(s) = s=="inla_graph_animal2" ? "animal2" : replace(s,"Pure F"=>"Pure_F","cuSOLVER F"=>"cuSOLVER_F","sTiles F"=>"sTiles_F")
function method_name(s)
    occursin("Mixed F16x",s) && return "M" * match(r"F16x(\d+)",s).captures[1]
    canonical(s)
end
function numeric_text(s)
    s=strip(s)
    s in ("—","Unsupported","Not measured","SKIP","ERROR","NaN","") && return ""
    supers=Dict('⁰'=>'0','¹'=>'1','²'=>'2','³'=>'3','⁴'=>'4','⁵'=>'5','⁶'=>'6','⁷'=>'7','⁸'=>'8','⁹'=>'9','⁻'=>'-','⁺'=>'+')
    value=replace(s,"×10"=>"e",","=>"")
    value=join(get(supers,c,c) for c in value)
    tryparse(Float64,value) === nothing && error("Unrecognized report number: $s")
    value
end
function between(doc,start,finish)
    a=findfirst(start,doc); a===nothing && error("Missing section $start")
    rest=SubString(doc,last(a)+1)
    b=findfirst(finish,rest); b===nothing && error("Missing end section $finish")
    String(SubString(rest,1,first(b)-1))
end
tokens(s)=filter(!isempty,strip.(split(s,'\n')))
function timing_table(block,names)
    values=tokens(block); out=Dict{Tuple{String,String},String}()
    for (i,label) in enumerate(values)
        m=canonical(label)
        m in METHODS[1:9] || continue
        for (j,name) in enumerate(names); out[(name,m)]=numeric_text(values[i+j]); end
    end
    length(out)==9length(names) || error("Incomplete timing table for $names")
    out
end
function legacy_real(doc)
    # The real timing section contains two blocks. Parse each using its own header.
    section=tokens(between(doc,"Real timings on unmodified inputs","Real sTiles timings"))
    starts=findall(==("Method"),section)
    length(starts)==2 || error("Expected two real timing panels")
    times=merge(timing_table(join(section[starts[1]:starts[2]-1],'\n'),OLD_NAMES[1:3]),
        timing_table(join(section[starts[2]:end],'\n'),OLD_NAMES[4:6]))
    phases=tokens(between(doc,"Real sTiles timings and hardware comparison","What to report as the headline"))
    phase_lookup=Dict{String,Tuple{String,String}}()
    for (i,s) in enumerate(phases)
        s in OLD_NAMES || continue
        # Name-only cells identify table rows; prose is not equal to a name.
        phase_lookup[s]=(numeric_text(phases[i+1]),numeric_text(phases[i+2]))
    end
    rows=Dict{String,String}[]
    for (j,name) in enumerate(OLD_NAMES)
        nextheading=j<length(OLD_NAMES) ? OLD_NAMES[j+1]*" accuracy" : "Dense accuracy"
        block=between(doc,name*" accuracy",nextheading)
        halves=split(block,"Shifted input";limit=2)
        length(halves)==2 || error("Missing paired inputs for $name")
        for (policy,part) in zip(("original","shifted"),halves)
            items=tokens(part)
            countrows=0
            for (i,item) in enumerate(items)
                method=canonical(item); method in METHODS || continue
                countrows+=1
                vals=numeric_text.(items[i+1:i+5])
                state=items[i+1]=="ERROR" ? "ERROR" : "OK"
                row=Dict{String,String}("cohort"=>"Earlier / 1-core CPU","kind"=>"real","matrix"=>name,"n"=>string(OLD_N[j]),
                    "policy"=>policy,"shift"=>string(policy=="original" ? 0.0 : SHIFTS[j]),"method"=>method,"status"=>state,
                    "nnz"=>string(OLD_NNZ[j]),"density"=>string(OLD_NNZ[j]/Float64(OLD_N[j])^2),
                    "application"=>name=="animal2" ? "Bayesian Statistics" : "Structural Engineering","stiles_cores"=>"1","source_precision"=>"rounded report transcription",
                    "accuracy_source"=>"Supplied synthesis: latest paired comparison tables [S1]; underlying transcript unavailable",
                    "timing_source"=>policy=="original" ? "Supplied synthesis: unmodified timing tables [S1]" : "not measured in paired accuracy source",
                    "detail"=>state=="ERROR" ? "Earlier report: invalid diagonal" : "")
                for (key,value) in zip(["factor_residual","forward_error","solve_residual","quantization_error","backward_error"],vals);row[key]=value;end
                if policy=="original"
                    if method=="sTiles_F64"
                        row["analysis_ms"],row["factor_ms"]=phase_lookup[name]
                        row["time_ms"]=row["factor_ms"]
                        row["timing_statistic"]="numeric factor time from minimum combined phase pair"
                    else
                        row["time_ms"]=times[(name,method)]
                        row["min_ms"]=row["time_ms"]
                        row["timing_statistic"]="minimum of five repetitions (separate accuracy run)"
                    end
                end
                name in OLD_NAMES[3:6] && (row["dominant_fraction"]="1.0")
                push!(rows,row)
            end
            countrows==10 || error("Expected 10 rows for $name / $policy; got $countrows")
        end
    end
    rows
end
function legacy_dense(doc,csv)
    names=["dense_$n" for n in (4096,8192,16384,32768,65536)]
    times=timing_table(between(doc,"Dense timings","Best mixed schedule by size"),names)
    phase_tokens=tokens(between(doc,"sTiles F64 on fully populated CSC inputs","Real timings on unmodified inputs"))
    phases=Dict{String,Tuple{String,String}}()
    for (i,item) in enumerate(phase_tokens)
        replace(item,","=>"") in string.((4096,8192,16384,32768,65536)) || continue
        phases["dense_"*replace(item,","=>"")]=(numeric_text(phase_tokens[i+1]),numeric_text(phase_tokens[i+2]))
    end
    rows=Dict{String,String}[]
    for old in CatalogSweep.read_csv(csv)
        startswith(old["matrix"],"dense_") || continue
        n=parse(Int,old["n"]); method=method_name(old["method"])
        row=Dict{String,String}("cohort"=>"Earlier / 1-core CPU","kind"=>"synthetic","matrix"=>old["matrix"],"n"=>old["n"],
            "policy"=>"seeded dense + nI","shift"=>old["shift"],"method"=>method,"status"=>old["status"]=="measured" ? "OK" : old["status"],
            "nnz"=>string(n^2),"density"=>"1.0","application"=>"Seeded symmetric random + nI (dense by construction)",
            "stiles_cores"=>"1","accuracy_source"=>"accuracy_20261008_225048.csv (raw dense rows)",
            "timing_source"=>"Supplied synthesis: dense timing tables [S1], separate accuracy run",
            "source_precision"=>"raw accuracy CSV; report-transcribed timings","detail"=>get(old,"note",""))
        for (key,field) in zip(METRICS,["input_quant_probe","factor_probe","forward_error","relative_residual","backward_error_fro"])
            row[key]=get(old,field,"")
        end
        if method=="sTiles_F64"
            row["analysis_ms"],row["factor_ms"]=phases[row["matrix"]]
            row["time_ms"]=row["factor_ms"]
            row["timing_statistic"]="numeric factor time from minimum combined phase pair"
        else
            row["time_ms"]=times[(row["matrix"],method)];row["min_ms"]=row["time_ms"]
            row["timing_statistic"]="minimum of five repetitions (separate accuracy run)"
        end
        push!(rows,row)
    end
    length(rows)==50 || error("Expected 50 raw dense accuracy rows")
    rows
end
function current_rows(path;cohort="Catalog / 14-core CPU",expected=200)
    out=Dict{String,String}[]
    for row in CatalogSweep.read_csv(path)
        r=Dict{String,String}(k=>get(row,k,"") for k in FIELDS)
        r["matrix"]=canonical(row["matrix"])
        merge!(r,Dict("cohort"=>cohort,"kind"=>"real","policy"=>"original","time_ms"=>row["median_ms"],
            "timing_statistic"=>"median of five repetitions","accuracy_source"=>cohort*": results.csv (raw)",
            "timing_source"=>cohort*": results.csv (same run)","source_precision"=>"raw CSV"))
        push!(out,r)
    end
    length(out)==expected || error("Expected $expected rows in $path")
    out
end
# Retain every observation in all-results.csv, but use fresh original-input reruns
# for current comparisons. Never combine a rerun original with a historical shift.
function primary_rows(rows)
    rerun=Set(r["matrix"] for r in rows if r["cohort"]=="Legacy rerun / 14-core CPU")
    filter(r->!startswith(r["cohort"],"Core sweep /") &&
        !(r["cohort"]=="Earlier / 1-core CPU" && r["policy"]=="original" && r["matrix"] in rerun),rows)
end
historical_rows(rows)=filter(r->r["cohort"]=="Earlier / 1-core CPU",rows)
core_rows(rows)=filter(r->startswith(r["cohort"],"Core sweep /"),rows)
function scaling_rows(rows)
    rr=core_rows(rows);out=Dict{String,String}[]
    for name in sort(unique(getindex.(rr,"matrix")))
        group=sort(filter(r->r["matrix"]==name,rr);by=r->nval(r,"stiles_cores"))
        baseline=only(filter(r->nval(r,"stiles_cores")==1,group))
        for r in group
            speed=validtime(r)&&validtime(baseline) ? nval(baseline,"time_ms")/nval(r,"time_ms") : NaN
            reps=filter(isfinite,[something(tryparse(Float64,x),NaN) for x in split(get(r,"repetition_ms",""),';')])
            push!(out,merge(copy(r),Dict("speedup"=>string(speed),"efficiency"=>string(speed/nval(r,"stiles_cores")),
                "repeat_min"=>string(isempty(reps) ? NaN : minimum(reps)),"repeat_max"=>string(isempty(reps) ? NaN : maximum(reps)))))
        end
    end
    out
end
function inventory(rows)
    out=Dict{String,String}[]
    for name in unique(getindex.(rows,"matrix"))
        rr=filter(r->r["matrix"]==name && r["policy"]!="shifted",rows)
        firstrow=first(rr)
        row=Dict(k=>get(firstrow,k,"") for k in ["matrix","n","nnz","density","application","cohort","kind","dominant_fraction","max_abs","min_abs_nonzero","worst_margin"])
        # Current diagnostics are absent on ERROR rows; take any recorded value.
        for key in keys(row)
            values=[r[key] for r in rr if !isempty(get(r,key,""))]
            isempty(values) || (row[key]=first(values))
        end
        n=nval(row,"n");row["density_percent"]=string(100nval(row,"density"));row["sparsity_percent"]=string(100*(1-nval(row,"density")))
        row["fp64_dense_gib"]=string(8n^2/2.0^30)
        row["dynamic_range"]=string(nval(row,"max_abs")/nval(row,"min_abs_nonzero"))
        row["original_mixed_ok"]=string(count(r->r["method"] in MIXED && r["status"]=="OK",rr))
        row["original_mixed_forward_pass"]=string(count(r->r["method"] in MIXED && passes(r),rr))
        row["original_mixed_factor_pass"]=string(count(r->r["method"] in MIXED && passes(r,"factor_residual"),rr))
        push!(out,row)
    end
    sort!(out;by=r->(r["kind"]=="synthetic",nval(r,"n")))
end
function lookup(rows)
    d=Dict((r["matrix"],r["policy"],r["method"])=>r for r in rows)
    length(d)==length(rows) || error("Duplicate observations after normalizing sources")
    d
end
const OUTCOME_GROUPS=["All five depths accurate","Some depths accurate","Breakdown; no accurate depth","All depths complete; none accurate"]
function outcome_rows(rows,inv;tol=1e-3)
    out=Dict{String,String}[]
    for meta in inv
        meta["kind"]=="real" || continue
        rr=filter(r->r["matrix"]==meta["matrix"] && r["policy"]=="original" && r["method"] in MIXED,rows)
        length(rr)==5 || error("Expected five current mixed observations for $(meta["matrix"])")
        good=filter(r->passes(r,"forward_error",tol),rr)
        failed=filter(r->r["status"]!="OK",rr)
        group=length(good)==5 ? OUTCOME_GROUPS[1] : !isempty(good) ? OUTCOME_GROUPS[2] : !isempty(failed) ? OUTCOME_GROUPS[3] : OUTCOME_GROUPS[4]
        finitevalues(key)=[nval(r,key) for r in rr if isfinite(nval(r,key))]
        q=finitevalues("quantization_error");f=finitevalues("forward_error")
        row=copy(meta)
        merge!(row,Dict("outcome"=>group,"forward_cutoff"=>string(tol),"accurate_methods"=>join(getindex.(good,"method"),";"),
            "failed_methods"=>join([r["method"]*":"*r["status"] for r in failed],";"),
            "mixed_forward_passes"=>string(length(good)),"mixed_completed"=>string(5-length(failed)),
            "severe_forward_methods"=>string(count(>=(.1),f)),"quantization_observed_methods"=>string(length(q)),
            "quantization_min"=>string(isempty(q) ? NaN : minimum(q)),"quantization_max"=>string(isempty(q) ? NaN : maximum(q)),
            "best_forward"=>string(isempty(f) ? NaN : minimum(f)),"worst_forward"=>string(isempty(f) ? NaN : maximum(f))))
        push!(out,row)
    end
    out
end
function bestmixed(rr; metric="forward_error",tol=1e-3,qualified=true)
    eligible=filter(r->r["method"] in MIXED && validtime(r) && (!qualified || passes(r,metric,tol)),rr)
    isempty(eligible) ? nothing : eligible[argmin(nval.(eligible,"time_ms"))]
end
function geometry(n,k)
    if k==0 || n<=1; return (leaves=1,offdiag=0,minleaf=n,maxleaf=n,fp16_entries=0);end
    mid=ispow2(n) ? n÷2 : prevpow(2,n)
    a,b=geometry(mid,k-1),geometry(n-mid,k-1)
    (leaves=a.leaves+b.leaves,offdiag=1+a.offdiag+b.offdiag,minleaf=min(a.minleaf,b.minleaf),maxleaf=max(a.maxleaf,b.maxleaf),
        fp16_entries=mid*(n-mid)+a.fp16_entries+b.fp16_entries)
end
function prepare(docfile,densefile,rundir,outdir,extended=nothing)
    mkpath(joinpath(outdir,"data"));mkpath(joinpath(outdir,"figures"));mkpath(joinpath(outdir,"chart-data"))
    doc=read(docfile,String)
    rows=vcat(current_rows(joinpath(rundir,"results.csv")),legacy_real(doc),legacy_dense(doc,densefile))
    lookup(rows);length(rows)==370 || error("Expected 370 original normalized observations")
    if extended!==nothing
        for (folder,cohort,expected) in vcat([("legacy-14","Legacy rerun / 14-core CPU",60)],
                [("cores-$p","Core sweep / $p requested cores",3) for p in (1,2,4,8,14)])
            path=joinpath(extended,folder)
            append!(rows,current_rows(joinpath(path,"results.csv");cohort=cohort,expected=expected))
            dest=joinpath(outdir,"data","extended",folder);mkpath(dest)
            for filename in ("results.csv","settings.toml","cases-summary.csv")
                isfile(joinpath(path,filename)) && cp(joinpath(path,filename),joinpath(dest,filename);force=true)
            end
            # Preserve failure stage evidence and the run's worker output.
            for foldername in ("logs","cases")
                isdir(joinpath(path,foldername)) && cp(joinpath(path,foldername),joinpath(dest,foldername);force=true)
            end
        end
        cp(joinpath(extended,"hardware.txt"),joinpath(outdir,"data","extended","hardware.txt");force=true)
    end
    CatalogSweep.write_csv(joinpath(outdir,"data","all-results.csv"),rows;columns=FIELDS)
    primary=primary_rows(rows);lookup(primary)
    CatalogSweep.write_csv(joinpath(outdir,"data","primary-results.csv"),primary;columns=FIELDS)
    scaling=scaling_rows(rows)
    CatalogSweep.write_csv(joinpath(outdir,"data","cpu-scaling.csv"),scaling;columns=vcat(FIELDS,["speedup","efficiency","repeat_min","repeat_max"]))
    inv=inventory(primary)
    CatalogSweep.write_csv(joinpath(outdir,"data","matrix-inventory.csv"),inv;columns=sort(collect(keys(first(inv)))))
    outcomes=outcome_rows(primary,inv)
    CatalogSweep.write_csv(joinpath(outdir,"data","matrix-outcomes.csv"),outcomes;columns=sort(collect(keys(first(outcomes)))))
    for (source,target) in [(docfile,"earlier-synthesis.txt"),(densefile,"earlier-accuracy-original.csv"),(joinpath(rundir,"results.csv"),"catalog-results-original.csv"),(joinpath(rundir,"settings.toml"),"catalog-settings.toml")]
        cp(source,joinpath(outdir,"data",target);force=true)
    end
    specs=chart_specs(primary,inv;history=historical_rows(rows),scaling=scaling)
    for spec in specs
        CatalogSweep.write_csv(joinpath(outdir,"chart-data",spec.id*".csv"),spec.points;columns=["series","x","y","label"])
    end
    CatalogSweep.write_csv(joinpath(outdir,"chart-data","charts.csv"),[Dict("id"=>s.id,"title"=>s.title,"xlabel"=>s.xlabel,"ylabel"=>s.ylabel,
        "xlog"=>s.xlog,"ylog"=>s.ylog,"kind"=>s.kind) for s in specs];columns=["id","title","xlabel","ylabel","xlog","ylog","kind"])
    @info "Prepared comprehensive report data" observations=length(rows) matrices=length(inv) charts=length(specs)
end

# Standard plotting backends consume these numeric datasets; no chart contains invented observations.
function chart_specs(rows,inv;history=historical_rows(rows),scaling=Dict{String,String}[])
    specs=NamedTuple[]
    add(id,title,xlabel,ylabel,points;xlog=false,ylog=false,kind="scatter")=push!(specs,(id=id,title=title,xlabel=xlabel,ylabel=ylabel,points=points,xlog=xlog,ylog=ylog,kind=kind))
    pt(series,x,y,label)=Dict{String,Any}("series"=>series,"x"=>x,"y"=>y,"label"=>label)
    original=filter(r->r["policy"]!="shifted",rows); dense=filter(r->r["kind"]=="synthetic",rows)
    for (id,field,title) in [("20-dense-timing","time_ms","Synthetic dense: timing versus dimension"),("21-dense-forward","forward_error","Synthetic dense: solution error versus dimension")]
        points=[pt(r["method"],nval(r,"n"),max(nval(r,field),1e-18),r["matrix"]) for r in dense if okay(r,field) && (field!="time_ms" || validtime(r))]
        add(id,title,"Dimension n",field=="time_ms" ? "Minimum GPU / selected CPU factor time (ms)" : "Known-solution forward error",points;xlog=true,ylog=true,kind="line")
    end
    for name in ("dense_4096","dense_16384","dense_65536","animal2","bodyy6")
        rr=filter(r->r["matrix"]==name && r["policy"]!="shifted" && r["method"] in MIXED,rows)
        points=[pt("Time (ms)",parse(Int,r["method"][2:end]),nval(r,"time_ms"),r["method"]) for r in rr if validtime(r)]
        add("22-depth-"*name,"Hierarchy depth versus runtime: "*name,"FP16 hierarchy entries k",name in OLD_NAMES ? "Current median runtime (ms)" : "Historical minimum runtime (ms)",points;ylog=true,kind="line")
    end
    d=lookup(rows);paired=lookup(history)
    for (id,metric,title) in [("23-shift-factor","factor_residual","Diagonal shift: M6 factor residual"),("24-shift-forward","forward_error","Diagonal shift: M6 solution error")]
        points=Dict{String,Any}[]
        for (i,name) in enumerate(OLD_NAMES),policy in ("original","shifted")
            row=get(paired,(name,policy,"M6"),nothing)
            okay(row,metric) && push!(points,pt(policy,i,max(nval(row,metric),1e-18),name))
        end
        add(id,title,"Matrix (labels below)",metric,points;ylog=true,kind="line")
    end
    for (id,metric,title) in [("25-residual-vs-forward","factor_residual","Small factor residual versus solution error"),("26-backward-vs-forward","backward_error","Small backward error versus solution error")]
        points=[pt(r["cohort"]*" / "*(r["method"] in MIXED ? "mixed" : "baseline"),max(nval(r,metric),1e-18),max(nval(r,"forward_error"),1e-18),r["matrix"]*" / "*r["method"]) for r in original if okay(r,metric) && okay(r,"forward_error")]
        add(id,title,metric,"Known-solution forward error",points;xlog=true,ylog=true)
    end
    for (id,metric,title) in [("27-density-crossing","density","Accuracy-qualified mixed versus sTiles by density"),("28-size-crossing","n","Accuracy-qualified mixed versus sTiles by dimension")]
        points=Dict{String,Any}[]
        for row in inv
            row["kind"]=="real" || continue
            name=row["matrix"]; rr=filter(r->r["matrix"]==name && r["policy"]=="original",rows)
            best=bestmixed(rr);cpu=get(d,(name,"original","sTiles_F64"),nothing)
            best===nothing && continue
            validtime(cpu) || continue
            push!(points,pt(row["cohort"],metric=="density" ? 100nval(row,metric) : nval(row,metric),nval(cpu,"time_ms")/nval(best,"time_ms"),name*" / "*best["method"]))
        end
        add(id,title,metric=="density" ? "Nonzero density (%)" : "Dimension n","CPU numeric factor / mixed time",points;xlog=true,ylog=true)
    end
    current=filter(r->r["kind"]=="real" && r["policy"]=="original" && r["source_precision"]=="raw CSV",rows)
    points=Dict{String,Any}[]
    for (i,method) in enumerate(METHODS)
        rr=filter(r->r["method"]==method,current)
        for (label,metric,tol) in [("Completed","time_ms",Inf),("Forward ≤ 1e-3","forward_error",1e-3),("Forward ≤ 1e-6","forward_error",1e-6),("Backward ≤ 1e-3","backward_error",1e-3),("Backward ≤ 1e-6","backward_error",1e-6)]
            push!(points,pt(label,i,count(r->passes(r,metric,tol),rr),method))
        end
    end
    add("29-current-coverage","All 26 real matrices: completion, forward and backward screens","Method","Number of matrices (out of 26)",points;kind="line")
    points=Dict{String,Any}[]
    for r in current
        r["method"] in MIXED && validtime(r) && okay(r,"forward_error") || continue
        ref=d[(r["matrix"],"original","cuSOLVER_F32")];validtime(ref) || continue
        push!(points,pt(r["method"],max(nval(r,"forward_error"),1e-18),nval(ref,"time_ms")/nval(r,"time_ms"),r["matrix"]*" / "*r["method"]))
    end
    add("30-current-tradeoff","All 26 real matrices: speed gain versus solution error","Known-solution forward error (left is better)","cuSOLVER F32 / mixed median time (above 1 is faster)",points;xlog=true,ylog=true)
    points=Dict{String,Any}[]
    for r in current
        r["method"] in MIXED && validtime(r) && okay(r,"backward_error") || continue
        ref=d[(r["matrix"],"original","cuSOLVER_F32")];validtime(ref) || continue
        push!(points,pt(r["method"],max(nval(r,"backward_error"),1e-18),nval(ref,"time_ms")/nval(r,"time_ms"),
            r["matrix"]*" / "*r["method"]*"; forward="*fnum(nval(r,"forward_error"))))
    end
    add("35-current-backward-tradeoff","All 26 real matrices: speed gain versus backward error","Frobenius-scaled backward error (left is smaller)","cuSOLVER F32 / mixed median time (above 1 is faster)",points;xlog=true,ylog=true)
    if !isempty(scaling)
        for (id,field,title,ylabel) in [("31-core-runtime","time_ms","sTiles numeric factorization: measured core sweep","Median factor time (ms)"),
                ("32-core-speedup","speedup","sTiles factor speedup relative to one requested core","T1 / Tp (factor medians)"),
                ("33-core-analysis","analysis_ms","sTiles analysis cost across requested core counts","Recorded selected analysis time (ms)")]
            points=[pt(r["matrix"],nval(r,"stiles_cores"),nval(r,field),r["matrix"]*" / "*r["stiles_cores"]*" cores") for r in scaling if isfinite(nval(r,field))]
            add(id,title,"Requested sTiles cores (BLAS threads = 1)",ylabel,points;ylog=field!="speedup",kind="line")
        end
        points=[pt(r["matrix"],parse(Int,r["method"][2:end]),max(nval(r,"forward_error"),1e-18),r["method"]) for r in current if r["matrix"] in OLD_NAMES && r["method"] in MIXED && okay(r,"forward_error")]
        add("34-rerun-depth-forward","Six legacy inputs rerun: deeper precision versus solution error","FP16 hierarchy entries k","Known-solution forward error",points;ylog=true,kind="line")
    end
    specs
end

function main(args=ARGS)
    length(args)>=1 || error("Usage: comprehensive.jl --prepare doc.txt dense.csv run-dir output-dir | --html output-dir run-dir")
    if args[1]=="--prepare"; length(args) in (5,6) || error("Expected four paths and optional extended-run directory");prepare(args[2:end]...)
    elseif args[1]=="--html"; length(args)==3 || error("Expected output and run directory");build_html(args[2],args[3])
    else; error("Unknown mode")
    end
end
include("comprehensive_html.jl")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && ComprehensiveCholesky.main()
