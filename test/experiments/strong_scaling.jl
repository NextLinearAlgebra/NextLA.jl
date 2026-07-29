using Test
using NextLA
using KernelAbstractions

include(joinpath(@__DIR__, "../../experiments/strong_scaling.jl"))
include(joinpath(@__DIR__, "../../experiments/rank_sweep.jl"))
include(joinpath(@__DIR__, "../../experiments/tile_size_sweep.jl"))
include(joinpath(@__DIR__, "../../experiments/matrix_shape_sweep.jl"))
using .StrongScalingExperiment
using .DenseGemmCommon
using .RankSweepExperiment
using .TileSizeSweepExperiment
using .MatrixShapeSweepExperiment

@testset "configuration-driven GEMM experiments" begin
    run = RunConfig((Float64,), 1, 1, 0, 29, KernelAbstractions.CPU())

    cases = [MatrixCase(:padded, :padded, :constant, 2, 2)]
    strong = StrongScalingConfig([8, 12], 4, (2, 2), cases, run)
    results = strong_scaling(strong)
    @test length(results) == 2
    @test [(r.m, r.k, r.n) for r in results] == [(8, 8, 8), (12, 12, 12)]
    @test all(r -> r.experiment == :strong_scaling, results)
    @test all(r -> r.dtype === Float64, results)
    @test all(r -> r.timing.tlr_dense_ms > 0, results)

    rank_results = rank_sweep(RankSweepConfig(8, 4, [1, 2], cases, run))
    @test length(rank_results) == 2
    @test [r.rank_A for r in rank_results] == [1, 2]

    tile_results = tile_size_sweep(TileSizeSweepConfig(8, [4, 8], 1 / 2, cases, run))
    @test length(tile_results) == 2
    @test [r.tile_size for r in tile_results] == [4, 8]

    shape = MatrixShapeSweepConfig(
        8, 4, 1, [(1, 1, 1), (2, 1, 1)], cases, run)
    shape_results = matrix_shape_sweep(shape)
    @test length(shape_results) == 2
    @test [(r.m, r.k, r.n) for r in shape_results] == [(8, 8, 8), (12, 8, 8)]
end
