using Test, TOML
include("report.jl")
const R = CatalogReport

# Deliberately synthetic: these numbers verify report calculations, not solvers.
function sample_row(name,method; state="OK", ms=8.0, error=1e-6)
    Dict("matrix"=>name,"method"=>method,"status"=>state,"n"=>name=="a&b" ? "4000" : "8000",
        "group"=>"fixture","min_ms"=>string(ms/2),"median_ms"=>string(ms),"forward_error"=>string(error),
        "factor_residual"=>method=="sTiles_F64" ? "" : string(error/10),"solve_residual"=>string(error/100),
        "backward_error"=>string(error/1000),"quantization_error"=>string(error/20),
        "density"=>"0.01","max_abs"=>"1e6","min_abs_nonzero"=>"0.01","dominant_fraction"=>"0.5",
        "analysis_ms"=>"0.5","factor_ms"=>string(ms-.5),"source_hash"=>"fixture-source",
        "git_commit"=>"fixture-commit","stiles_cores"=>"14","gpu"=>"SYNTHETIC TEST DATA",
        "detail"=>state=="OK" ? "" : "<native timeout>")
end
function fixture()
    rows=[sample_row("a&b","M1";ms=2,error=1e-4),sample_row("a&b","M2";ms=1,error=.1),
        sample_row("a&b","cuSOLVER_F32"),sample_row("a&b","cuSOLVER_F64";ms=12,error=0.0),
        sample_row("a&b","sTiles_F64";ms=4),sample_row("b","M1";state="TIMEOUT",ms=1),
        sample_row("b","cuSOLVER_F32";ms=16)]
    settings=Dict("selection"=>["a&b","b","unrecorded"],"methods"=>"all",
        "gpu_methods"=>["M1","M2","cuSOLVER_F32","cuSOLVER_F64"])
    rows,settings
end

@testset "Paired timings, accuracy and missing-case denominators" begin
    rows,settings=fixture()
    d=R.prepare(rows,settings)
    @test length(d.matrices)==3
    @test length(d.methods)==5
    @test R.speedup(d,"a&b","M1","cuSOLVER_F32")==4
    @test isnan(R.speedup(d,"b","M1","cuSOLVER_F32"))
    @test !R.qualifies(rows[6],"forward_error",1e-3)
    @test !R.qualifies(rows[2],"forward_error",1e-3)
    @test R.qualifies(rows[4],"forward_error",1e-3) # exact zero stays a pass
    summary,winners,problems=R.summaries(d)
    m1=only(filter(r -> r["method"]=="M1",summary))
    @test m1["selected_matrices"]==3
    @test m1["ok"]==1
    @test m1["accuracy_pass"]==1
    @test m1["not_recorded"]==1
    @test m1["cuSOLVER_F32_pairs"]==1
    @test m1["cuSOLVER_F32_geomean_speedup"]≈4
    @test m1["cuSOLVER_F32_accuracy_pass_pairs"]==1
    m2=only(filter(r -> r["method"]=="M2",summary))
    @test m2["cuSOLVER_F32_pairs"]==1
    @test m2["cuSOLVER_F32_accuracy_pass_pairs"]==0
    @test isnan(m2["cuSOLVER_F32_accuracy_pass_geomean_speedup"])
    @test winners[1]["fastest_observed_qualifying_method"]=="M1"
    @test winners[end]["fastest_observed_qualifying_method"]=="NONE"
    @test any(r -> r["status"]=="NOT_RECORDED",problems)
    @test any(r -> r["status"]=="TIMEOUT",problems)
    @test any(r -> r["detail"]=="Above exploratory accuracy cutoff",problems)
    rows[1]["median_ms"]="0"
    @test isnan(R.speedup(R.prepare(rows,settings),"a&b","M1","cuSOLVER_F32"))
    @test R.speedup(R.prepare(rows,settings;timing="min"),"a&b","M1","cuSOLVER_F32")==4
end

@testset "Input validation and escaped report text" begin
    rows,settings=fixture()
    @test_throws ErrorException R.prepare(vcat(rows,[first(rows)]),settings)
    mixed=deepcopy(rows); mixed[1]["source_hash"]="different"
    @test_throws ErrorException R.prepare(mixed,settings)
    @test_throws ErrorException R.prepare(rows,settings;metric="invented")
    @test_throws ErrorException R.prepare(rows,settings;tolerance=-1)
    @test_throws ErrorException R.prepare(rows,settings;timing="average")
    @test_throws ErrorException R.prepare(Dict{String,String}[],settings)
    d=R.prepare(rows,settings)
    html=R.html_report(d,NamedTuple[],R.summaries(d)...,"fixture-hash","<settings>")
    @test occursin("a&amp;b",html)
    @test !occursin("<native timeout>",html)
    @test occursin("&lt;native timeout&gt;",html)
    @test occursin("Partial snapshot",html)
    @test length(R.prepare(rows).matrices)==2 # Without settings, only observed names are known.
    @test occursin("fixture-hash",html)
    @test occursin("&lt;settings&gt;",html)
end

@testset "Stacked count and phase bars have disjoint intervals" begin
    segments=R.stack_segments([2 1 0; 0 1 2])
    first_bar=filter(s -> s.row==1,segments)
    @test [(s.lower,s.upper) for s in first_bar]==[(0.0,2.0),(2.0,3.0)]
    @test last(filter(s -> s.row==2,segments)).upper==3
    @test length(segments)==4
    phases=R.stack_segments([25.0 75.0; 0.0 100.0])
    @test all(i -> last(filter(s -> s.row==i,phases)).upper==100,1:2)
    @test_throws ErrorException R.stack_segments([1.0 -1.0])
    @test_throws ErrorException R.stack_segments([NaN 1.0])
end

# Inspect the real chart datasets even on systems without a usable GR library.
@testset "Figure datasets and incomplete baseline coverage" begin
    rows,settings=fixture()
    d=R.prepare(rows,settings)
    render_heatmap(d,title,subtitle,value;kwargs...) =
        (kind=:heatmap,data=[value(n,m) for n in d.matrices,m in d.methods])
    render_scatter(d,title,subtitle,points;kwargs...) = (kind=:scatter,data=points)
    renderer=(heatmap=render_heatmap,scatter=render_scatter,status=d -> nothing,stiles=d -> nothing)
    figs=R.figures(d;renderer=renderer)
    @test length(figs)==19
    @test length(unique(getproperty.(figs,:id)))==19
    forward=only(filter(f -> f.id=="02-forward-error",figs)).svg.data
    @test count(isfinite,forward)==6 # timeout and unrecorded cells are excluded
    @test minimum(filter(isfinite,vec(forward)))==0.0
    ratios=only(filter(f -> f.id=="07a-speedup-cuSOLVER_F32",figs)).svg.data
    @test ratios[1,findfirst(==("M1"),d.methods)]==4
    @test isnan(ratios[2,findfirst(==("M1"),d.methods)])
    tradeoff=only(filter(f -> f.id=="08a-tradeoff-cuSOLVER_F32",figs)).svg.data
    @test length(tradeoff)==2
    @test all(p -> p.matrix=="a&b",tradeoff)
    coverage=only(filter(f -> f.id=="14-accuracy-coverage",figs)).svg.data
    @test only(filter(p -> p.method=="M1" && p.x==1e-3,coverage)).y≈100/3
    failed=R.prepare([sample_row("a&b","M1";state="TIMEOUT")],settings)
    empty_figs=R.figures(failed;renderer=renderer)
    @test isempty(only(filter(f -> f.id=="08a-tradeoff-cuSOLVER_F32",empty_figs)).svg.data)
    @test all(isnan,only(filter(f -> f.id=="03-factor-residual",empty_figs)).svg.data)
end

# Optional end-to-end rendering check with Plots installed:
# julia --project=test/catalog test/catalog/report_tests.jl --render
if "--render" in ARGS
    @testset "Rendered report and vector/PNG exports (synthetic data)" begin
        rows,settings=fixture()
        output=abspath(get(ENV,"REPORT_TEST_DIR",joinpath(@__DIR__,"..","..","benchmark-results","report-smoke")))
        mkpath(output)
        input=joinpath(output,"results.csv")
        R.CatalogSweep.write_csv(input,rows;columns=sort(collect(keys(first(rows)))))
        open(io -> TOML.print(io,settings),joinpath(output,"settings.toml"),"w")
        generated=R.generate(input)
        html=read(joinpath(generated,"report.html"),String)
        @test count("<figure ",html)==19
        @test length(filter(f -> endswith(f,".svg"),readdir(joinpath(generated,"figures"))))==19
        @test length(filter(f -> endswith(f,".png"),readdir(joinpath(generated,"figures"))))==19
        @test length(R.CatalogSweep.read_csv(joinpath(generated,"method-summary.csv")))==5
        @test occursin("a&amp;b",html)
        ids=[m.captures[1] for m in eachmatch(r"\bid=\"([^\"]+)\"",html)]
        @test length(ids)==length(unique(ids))
        # Empty numerical plots must also render when all recorded cases failed.
        failed=[sample_row("a&b","M1";state="TIMEOUT")]
        R.CatalogSweep.write_csv(input,failed;columns=sort(collect(keys(first(failed)))))
        R.generate(input;outdir=joinpath(output,"all-failed-report"))
        @test isfile(joinpath(output,"all-failed-report","report.html"))
    end
end
