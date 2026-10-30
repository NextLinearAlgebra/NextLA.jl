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
        inv=C.inventory(rows);d=C.lookup(rows)
        @test length(rows)==370
        @test length(inv)==31
        @test count(r->r["kind"]=="real",inv)==26
        @test count(r->r["kind"]=="synthetic",inv)==5
        @test count(r->r["policy"]=="shifted",rows)==60
        @test all(r->!haskey(r,"time_ms") || isempty(r["time_ms"]),filter(r->r["policy"]=="shifted",rows))
        @test C.nval(d[("animal2","original","M6")],"time_ms")==338.315
        @test C.nval(d[("nasasrb","original","M6")],"factor_residual")==.272
        @test C.nval(d[("nasasrb","shifted","M6")],"factor_residual")==.00257
        @test C.nval(d[("dense_65536","seeded dense + nI","M6")],"time_ms")==413.606
        @test d[("dense_65536","seeded dense + nI","sTiles_F64")]["status"]=="SKIP"
        @test d[("nasasrb","original","M7")]["status"]=="ERROR"
        @test occursin("rounded",d[("bodyy6","original","M1")]["source_precision"])
        @test count(r->r["cohort"]=="Catalog / 14-core CPU" && r["status"]=="ERROR",rows)==29
        @test length(C.chart_specs(rows,inv))==13
        @test all(r->C.nval(r,"density_percent")+C.nval(r,"sparsity_percent")≈100,inv)
        html=read(joinpath(prepared,"report.html"),String)
        @test count("<figure ",html)==32
        @test count("<section class='matrix'>",html)==31
        @test count("<h2 id='",html)==13
        @test occursin("Historical precision",html)
        @test occursin("Overall conclusion",html)
    end
end
