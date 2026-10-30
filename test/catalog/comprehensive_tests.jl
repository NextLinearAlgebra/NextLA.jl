using Test
include("comprehensive.jl")
const C=ComprehensiveCholesky
@testset "Historical numeric transcription and identity" begin
    @test parse(Float64,C.numeric_text("2.72×10⁻¹"))≈.272
    @test parse(Float64,C.numeric_text("1,986.099"))≈1986.099
    @test C.numeric_text("—")==""
    @test C.numeric_text("ERROR")==""
    @test_throws ErrorException C.numeric_text("unrecorded value")
    @test C.method_name("Mixed F16x6 F32")=="M6"
    @test C.canonical("inla_graph_animal2")=="animal2"
    @test C.canonical("inla_graph_animal1")=="inla_graph_animal1"
end
@testset "Hierarchy geometry follows actual partition policy" begin
    a=C.geometry(65536,6);b=C.geometry(65536,7)
    @test (a.leaves,a.offdiag,a.minleaf,a.maxleaf)==(64,63,1024,1024)
    @test (b.leaves,b.offdiag,b.minleaf,b.maxleaf)==(128,127,512,512)
    @test b.fp16_entries>a.fp16_entries
    @test 100*(b.fp16_entries-a.fp16_entries)/(65536*65537/2)<1
    for n in (1,7,3948,5126,17546,50126),k in (1,4,7)
        g=C.geometry(n,k)
        @test 1<=g.minleaf<=g.maxleaf<=n
        @test g.offdiag==g.leaves-1
        @test 0<=g.fp16_entries<=n*(n-1)÷2
    end
end
prepared=normpath(joinpath(@__DIR__,"..","..","benchmark-results","comprehensive-cholesky"))
if isfile(joinpath(prepared,"data","all-results.csv"))
    @testset "Available real experiment audit" begin
        rows=C.CatalogSweep.read_csv(joinpath(prepared,"data","all-results.csv"))
        primary=C.primary_rows(rows);inv=C.inventory(primary);d=C.lookup(primary)
        old=C.lookup(C.historical_rows(rows));scaling=C.scaling_rows(rows)
        @test length(rows)==445
        @test length(primary)==370
        @test length(scaling)==15
        @test length(C.historical_rows(rows))==170
        @test length(inv)==31
        @test count(r->r["kind"]=="real",inv)==26
        @test count(r->r["kind"]=="synthetic",inv)==5
        @test count(r->r["policy"]=="shifted",rows)==60
        @test all(r->!haskey(r,"time_ms") || isempty(r["time_ms"]),filter(r->r["policy"]=="shifted",rows))
        @test C.nval(d[("animal2","original","M6")],"time_ms")==357.482628
        @test C.nval(old[("nasasrb","original","M6")],"factor_residual")==.272
        @test d[("nasasrb","original","M6")]["status"]=="ERROR"
        @test C.nval(d[("nasasrb","original","M6")],"quantization_error")≈7.564124732457926e-5
        @test d[("crankseg_2","original","M6")]["status"]=="ERROR"
        @test d[("animal2","original","sTiles_F64")]["status"]=="TIMEOUT"
        @test !C.validtime(d[("animal2","original","sTiles_F64")])
        @test C.nval(d[("nasasrb","shifted","M6")],"factor_residual")==.00257
        @test C.nval(d[("dense_65536","seeded dense + nI","M6")],"time_ms")==413.606
        @test d[("dense_65536","seeded dense + nI","sTiles_F64")]["status"]=="SKIP"
        @test d[("nasasrb","original","M7")]["status"]=="ERROR"
        @test d[("bodyy6","original","M1")]["source_precision"]=="raw CSV"
        @test occursin("rounded",old[("bodyy6","original","M1")]["source_precision"])
        @test count(r->r["cohort"]=="Catalog / 14-core CPU" && r["status"]=="ERROR",rows)==29
        current=C.analysis_rows(primary)
        @test length(current)==260
        @test count(r->r["status"]=="OK",current)==226
        @test count(r->r["status"]=="ERROR",current)==33
        @test count(r->r["status"]=="TIMEOUT",current)==1
        for method in C.MIXED
            @test count(r->r["method"]==method && C.passes(r),current)==(method=="M1" ? 14 : 11)
        end
        best=[C.bestmixed(filter(r->r["matrix"]==name,current)) for name in unique(getindex.(current,"matrix"))]
        eligible=filter(!isnothing,best)
        compared=filter(r->C.validtime(d[(r["matrix"],"original","sTiles_F64")]),eligible)
        @test length(eligible)==14
        @test length(compared)==13
        @test count(r->C.nval(d[(r["matrix"],"original","sTiles_F64")],"time_ms")>C.nval(r,"time_ms"),compared)==4
        outcomes=C.outcome_rows(primary,inv)
        @test length(outcomes)==26
        @test length(unique(getindex.(outcomes,"matrix")))==26
        @test [count(r->r["outcome"]==g,outcomes) for g in C.OUTCOME_GROUPS]==[11,3,10,2]
        @test Set(r["matrix"] for r in outcomes if r["outcome"]==C.OUTCOME_GROUPS[2])==Set(["bodyy4","bodyy5","bodyy6"])
        @test all(r->r["accurate_methods"]=="M1",filter(r->r["outcome"]==C.OUTCOME_GROUPS[2],outcomes))
        @test all(r->r["application"]=="Bayesian Statistics",filter(r->r["outcome"]==C.OUTCOME_GROUPS[1],outcomes))
        @test Set(r["matrix"] for r in outcomes if r["outcome"]==C.OUTCOME_GROUPS[4])==Set(["gyro_m","inla_graph_stcov"])
        @test count(r->C.nval(r,"severe_forward_methods")>0,outcomes)==9
        @test count(r->r["outcome"]==C.OUTCOME_GROUPS[1] && C.nval(r,"quantization_max")<=1e-12,outcomes)==10
        @test all(r->C.nval(r,"dominant_fraction")<.336,filter(r->r["outcome"]==C.OUTCOME_GROUPS[3],outcomes))
        @test count(r->C.nval(r,"dominant_fraction")==1,outcomes)==10
        @test all(r->C.nval(r,"mixed_completed")==5,filter(r->C.nval(r,"dominant_fraction")==1,outcomes))
        @test all(r->C.passes(r),filter(r->r["kind"]=="synthetic" && r["method"] in C.MIXED,primary))
        exported=C.CatalogSweep.read_csv(joinpath(prepared,"data","matrix-outcomes.csv"))
        @test getindex.(exported,"outcome")==getindex.(outcomes,"outcome")
        specs=C.chart_specs(primary,inv;history=C.historical_rows(rows),scaling=scaling)
        @test length(specs)==20
        coverage=only(filter(s->s.id=="29-current-coverage",specs))
        @test [Int(p["y"]) for p in coverage.points if p["series"]=="Backward ≤ 1e-3"]==[25,26,25,21,19,16,15,25,26,25]
        @test [Int(p["y"]) for p in coverage.points if p["series"]=="Backward ≤ 1e-6"]==[25,26,24,20,18,13,12,25,26,25]
        backwardchart=only(filter(s->s.id=="35-current-backward-tradeoff",specs))
        @test length(backwardchart.points)==sum(r->r["method"] in C.MIXED && C.validtime(r) && C.okay(r,"backward_error"),current)
        @test any(p->startswith(p["label"],"nasasrb / M4") && p["x"]<1e-3 && p["y"]>1,backwardchart.points)
        @test C.nval(d[("nasasrb","original","M4")],"forward_error")>.8
        shiftchart=only(filter(s->s.id=="23-shift-factor",specs))
        @test any(p->p["series"]=="original" && p["label"]=="nasasrb" && p["y"]==.272,shiftchart.points)
        for r in scaling
            @test r["status"]=="OK"
            @test length(split(r["repetition_ms"],';'))==5
            @test C.nval(r,"efficiency")≈C.nval(r,"speedup")/C.nval(r,"stiles_cores")
            @test C.nval(r,"repeat_min")<=C.nval(r,"time_ms")<=C.nval(r,"repeat_max")
            @test r["matrix_sha256"]==d[(r["matrix"],"original","sTiles_F64")]["matrix_sha256"]
        end
        sem=filter(r->r["matrix"]=="inla_graph_sem_n5000",scaling)
        @test C.nval(sem[argmin(C.nval.(sem,"time_ms"))],"stiles_cores")==2
        @test C.nval(only(filter(r->C.nval(r,"stiles_cores")==14,sem)),"speedup")<1
        for folder in ("legacy-14","cores-1","cores-2","cores-4","cores-8","cores-14")
            source=joinpath(prepared,"..","extended-27277754-20261029-230602",folder,"results.csv")
            copy=joinpath(prepared,"data","extended",folder,"results.csv")
            @test read(source)==read(copy)
        end
        @test all(r->C.nval(r,"density_percent")+C.nval(r,"sparsity_percent")≈100,inv)
        html=read(joinpath(prepared,"report.html"),String)
        @test count("<figure ",html)==39
        @test count("<section class='matrix'>",html)==31
        @test count("<h2 id='",html)==15
        @test [m.captures[1] for m in eachmatch(r"<h2 id='([^']+)'",html)]==first.(C.REPORT_SECTIONS)
        @test first(findfirst("fig-29-current-coverage",html))<first(findfirst("id='crossover'",html))
        @test first(findfirst("fig-35-current-backward-tradeoff",html))<first(findfirst("id='crossover'",html))
        @test first.(C.REPORT_SECTIONS)[2]=="patterns"
        @test occursin("sh7Pgi has zero strictly dominant rows",html)
        @test !occursin("separate pending experiment",html)
        @test !occursin("extended sweep is prepared but",html)
        @test occursin("Historical precision",html)
        @test occursin("Overall conclusion",html)
    end
end
