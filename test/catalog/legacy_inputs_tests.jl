using Test
include("legacy_inputs.jl")
@testset "Legacy file identity checks" begin
    entries=CatalogSweep.read_csv(joinpath(@__DIR__,"legacy-matrices.csv"))
    @test length(entries)==6
    @test Set(getindex.(entries,"name"))==Set(["bodyy4","bodyy5","bodyy6","nasasrb","crankseg_2","inla_graph_animal2"])
    mktempdir() do dir
        path=joinpath(dir,"tiny.mtx")
        write(path,"%%MatrixMarket matrix coordinate real symmetric\n% header\n2 2 2\n1 1 1\n2 2 1\n")
        @test header_ok(path,2)
        @test !header_ok(path,3)
        write(path,"<html>download failed</html>\n")
        @test !header_ok(path,2)
    end
end
