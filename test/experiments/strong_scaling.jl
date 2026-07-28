using Test
using NextLA
using KernelAbstractions

include(joinpath(@__DIR__, "../../experiments/strong_scaling.jl"))
using .StrongScalingExperiment

@testset "configuration-driven GEMM experiments" begin
    run = RunConfig((Float64,), 1, 1, 0, 29, KernelAbstractions.CPU())

    strong = StrongScalingConfig([8, 12], 4, (2, 2), run)
    results = strong_scaling(strong)
    @test length(results) == 2
    @test [(r.m, r.k, r.n) for r in results] == [(8, 8, 8), (12, 12, 12)]
    @test all(r -> r.experiment == :strong_scaling, results)
    @test all(r -> r.dtype === Float64, results)
    @test all(r -> r.timing.tlr_dense_ms > 0, results)

    rank_results = rank_sweep(RankSweepConfig(8, 4, [1, 2], run))
    @test length(rank_results) == 2
    @test [r.rank_A for r in rank_results] == [1, 2]

    tile_results = tile_size_sweep(TileSizeSweepConfig(8, [4, 8], 1, run))
    @test length(tile_results) == 2
    @test [r.tile_size for r in tile_results] == [4, 8]

    shape = MatrixShapeSweepConfig(
        8, 4, 1, [(1, 1, 1), (2, 1, 1)], run)
    shape_results = matrix_shape_sweep(shape)
    @test length(shape_results) == 2
    @test [(r.m, r.k, r.n) for r in shape_results] == [(8, 8, 8), (12, 8, 8)]
end
