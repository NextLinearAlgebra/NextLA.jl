"""Run PaddedFTLR × PaddedFTLR → PaddedFTLR benchmarks."""

using CUDA

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "padded_ftlr_output.jl"))
include(joinpath(@__DIR__, "strong_scaling.jl"))
include(joinpath(@__DIR__, "overlap_sweep.jl"))

using .DenseGemmCommon
using .PaddedFTLROutputExperiment
using .PaddedFTLROutputStrongScaling
using .PaddedFTLROutputOverlapSweep

const PRECISIONS = (PrecisionConfig(:fp32_full, Float32, GEMMCompute{Float32}()),)
const NWARMUP = 1
const NREPS = 3
const ROWS_PER_RUN = 4
const SEED = 20260728
const BLOCK = 32
const TOL = 0.0
const RELATIVE_TOLERANCE = false
const SHOW_PROGRESS = true
const OUTPUT_DIR = joinpath(@__DIR__, "results")

CUDA.functional() || error("the benchmark requires a functional CUDA device")
const BACKEND = CUDA.CUDABackend()
const RUN = PaddedFTLROutputRunConfig(
    PRECISIONS, ROWS_PER_RUN, NREPS, NWARMUP, SEED, BACKEND;
    block=BLOCK, tol=TOL, rel=RELATIVE_TOLERANCE,
    show_progress=SHOW_PROGRESS)

function main()
    mkpath(OUTPUT_DIR)
    write_padded_ftlr_output_csv(
        joinpath(OUTPUT_DIR, "strong_scaling.csv"),
        PaddedFTLROutputStrongScaling.run(RUN))
    write_padded_ftlr_output_csv(
        joinpath(OUTPUT_DIR, "overlap_sweep.csv"),
        PaddedFTLROutputOverlapSweep.run(RUN))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
