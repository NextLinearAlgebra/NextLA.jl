"""Orchestrate the padded/PackedFTLR-storage benchmark campaign.

This campaign uses the padded storage format; the dense-output campaign uses
the same sweep implementations with additional compressed-storage cases.
"""

using KernelAbstractions

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "strong_scaling.jl"))
include(joinpath(@__DIR__, "..", "tile_size_sweep.jl"))
include(joinpath(@__DIR__, "strong_scaling.jl"))
include(joinpath(@__DIR__, "tile_size_sweep.jl"))

using .DenseGemmCommon
using .StrongScalingExperiment
using .TileSizeSweepExperiment
using .PackedStrongScaling
using .PackedTileSizeSweep

# ── Common campaign configuration ────────────────────────────────────────────
const PRECISIONS = (
    PrecisionConfig(:fp16_full, Float16, GEMMCompute{Float16}()),
    PrecisionConfig(:fp32_full, Float32, GEMMCompute{Float32}()),
)
const NWARMUP = 1
const NREPS = 3
const WORKSPACE_FACTOR = 2
const SEED = 20260728
# A dense Float32 reference at 32768 is several GiB; enable only for a
# deliberately small validation campaign.
const CHECK_RESULTS = false
const SHOW_PROGRESS = true
const OUTPUT_DIR = joinpath(@__DIR__, "results")

const BACKEND = let
    try
        @eval import CUDA
        CUDA.functional() ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    catch
        KernelAbstractions.CPU()
    end
end

const RUN = RunConfig(PRECISIONS, WORKSPACE_FACTOR, NREPS, NWARMUP, SEED, BACKEND;
                      check_results=CHECK_RESULTS, show_progress=SHOW_PROGRESS)

function main()
    mkpath(OUTPUT_DIR)
    write_dense_csv(joinpath(OUTPUT_DIR, "strong_scaling.csv"), PackedStrongScaling.run(RUN))
    write_dense_csv(joinpath(OUTPUT_DIR, "tile_size_sweep.csv"), PackedTileSizeSweep.run(RUN))
    nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
