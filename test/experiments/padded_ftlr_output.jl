using Test
using KernelAbstractions

isdefined(Main, :DenseGemmCommon) ||
    include(joinpath(@__DIR__, "../../experiments/common.jl"))
include(joinpath(@__DIR__, "../../experiments/padded_ftlr_output.jl"))
using .PaddedFTLROutputExperiment

@testset "Padded-FTLR-output experiments" begin
    run = PaddedFTLROutputRunConfig(
        (Float64,), 1, 1, 0, 71, KernelAbstractions.CPU();
        block=2, tol=1e-8, rel=true, show_progress=false)

    strong = PaddedFTLROutputStrongScalingConfig([8], 4, (1, 1), 2, run)
    strong_results = padded_ftlr_output_strong_scaling(strong)
    @test length(strong_results) == 1
    result = only(strong_results)
    @test result.experiment == :padded_ftlr_output_strong_scaling
    @test result.timing.tlr_tlr_ms > 0
    @test result.timing.dense_compress_ms > 0
    @test result.timing.dense_dense_ms > 0
    @test isfinite(result.timing.tlr_tlr_rel_fro_error)
    @test isfinite(result.timing.dense_compress_rel_fro_error)

    overlap = PaddedFTLROutputOverlapConfig(8, 4, (1, 1), 2, [1], run)
    overlap_results = padded_ftlr_output_overlap_sweep(overlap)
    @test only(overlap_results).shared_rank == 1
end
