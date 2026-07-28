using Test
using KernelAbstractions

include(joinpath(@__DIR__, "../../experiments/tlr_output.jl"))
using .TLROutputExperiment

@testset "TLR-output experiments" begin
    run = TLROutputRunConfig(
        (Float64,), 1, 1, 0, 71, KernelAbstractions.CPU();
        block=2, tol=1e-8, rel=true)

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

    overlap = TLROutputOverlapConfig(8, 4, (1, 1), 2, [0, 1], run)
    overlap_results = tlr_output_overlap_sweep(overlap)
    @test length(overlap_results) == 2
    @test [r.shared_rank for r in overlap_results] == [0, 1]
end
