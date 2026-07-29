using Test
using KernelAbstractions

isdefined(Main, :DenseGemmCommon) ||
    include(joinpath(@__DIR__, "../../experiments/common.jl"))
include(joinpath(@__DIR__, "../../experiments/tlr_output.jl"))
using .TLROutputExperiment

@testset "TLR-output experiments" begin
    run = TLROutputRunConfig(
        (Float64,), 1, 1, 0, 71, KernelAbstractions.CPU();
        block=2, tol=1e-8, rel=true, show_progress=false)

    strong = TLROutputStrongScalingConfig([8], 4, (1, 1), 2, run)
    strong_results = tlr_output_strong_scaling(strong)
    @test length(strong_results) == 1
    result = only(strong_results)
    @test result.experiment == :tlr_output_strong_scaling
    @test result.timing.tlr_tlr_ms > 0
    @test result.timing.dense_compress_ms > 0
    @test result.timing.dense_dense_ms > 0
    @test isfinite(result.timing.tlr_tlr_rel_fro_error)
    @test isfinite(result.timing.dense_compress_rel_fro_error)

    overlap = TLROutputOverlapConfig(8, 4, (1, 1), 2, [1], run)
    overlap_results = tlr_output_overlap_sweep(overlap)
    @test only(overlap_results).shared_rank == 1
end
