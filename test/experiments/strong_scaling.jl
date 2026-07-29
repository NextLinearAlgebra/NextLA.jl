using Test
using NextLA
using KernelAbstractions

include(joinpath(@__DIR__, "../../experiments/common.jl"))
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
    run = RunConfig((Float64,), 1, 1, 0, 29, KernelAbstractions.CPU();
                    show_progress=false)

    cases = [MatrixCase(:padded, :padded, :constant, 2, 2)]
    strong = StrongScalingConfig([8], 4, (2, 2), cases, run)
    results = strong_scaling(strong)
    @test length(results) == 1
    @test (only(results).m, only(results).k, only(results).n) == (8, 8, 8)
    @test all(r -> r.experiment == :strong_scaling, results)
    @test all(r -> r.dtype === Float64, results)
    @test all(r -> r.timing.tlr_dense_ms > 0, results)

    Acount, Bcount = DenseGemmCommon.generate_ftlr_operands(
        8, 8, 8, 4, (1, 2), Float64;
        backend=KernelAbstractions.CPU(), format=:padded)
    row_bytes = DenseGemmCommon._row_run_workspace_bytes(Acount, Bcount, 1)
    @test row_bytes == cld(NextLA.gemm_maximum_workspace_bytes(Acount, Bcount), 2)
    # Row-major N/N executes FoldRight. Its per-(i,k,j) cost is
    # 2*(bk*rA*rB + bn*rA*rB + bm*rA*bn) = 64 for this geometry.
    @test DenseGemmCommon._tlr_tlr_executed_flops(Acount, Bcount, row_bytes) == 512

    rank_results = rank_sweep(RankSweepConfig(8, 4, [1], cases, run))
    @test only(rank_results).rank_A == 1

    tile_results = tile_size_sweep(TileSizeSweepConfig(8, [4], 1 / 2, cases, run))
    @test only(tile_results).tile_size == 4

    shape = MatrixShapeSweepConfig(8, 4, 1, [(2, 1, 1)], cases, run)
    shape_results = matrix_shape_sweep(shape)
    @test [(r.m, r.k, r.n) for r in shape_results] == [(12, 8, 8)]
end
