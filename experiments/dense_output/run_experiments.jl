"""Run dense-output benchmarks with padded and packed FTLR operands."""

using CUDA

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "strong_scaling.jl"))
include(joinpath(@__DIR__, "..", "rank_sweep.jl"))
include(joinpath(@__DIR__, "..", "tile_size_sweep.jl"))
include(joinpath(@__DIR__, "..", "matrix_shape_sweep.jl"))
include(joinpath(@__DIR__, "strong_scaling.jl"))
include(joinpath(@__DIR__, "rank_sweep.jl"))
include(joinpath(@__DIR__, "tile_size_sweep.jl"))
include(joinpath(@__DIR__, "matrix_shape_sweep.jl"))

using .DenseGemmCommon
using .StrongScalingExperiment
using .RankSweepExperiment
using .TileSizeSweepExperiment
using .MatrixShapeSweepExperiment
using .DenseOutputStrongScaling
using .DenseOutputRankSweep
using .DenseOutputTileSizeSweep
using .DenseOutputMatrixShapeSweep

# ── Common campaign configuration ────────────────────────────────────────────
const PRECISIONS = (
    PrecisionConfig(:fp16_fp32, Float16, GEMMCompute{Float32}()),
    PrecisionConfig(:bf16_fp32, Core.BFloat16, GEMMCompute{Float32}()),
    PrecisionConfig(:fp32_tf32, Float32, TF32()),
)
const NWARMUP = 1
const NREPS = 3
const ROWS_PER_RUN = 4
const SEED = 20260728
const CHECK_RESULTS = true
const SHOW_PROGRESS = true
const OUTPUT_DIR = joinpath(@__DIR__, "results")

CUDA.functional() || error("the benchmark requires a functional CUDA device")
CUDA.capability(CUDA.device()) >= v"8.0" ||
    error("the BF16 campaign requires an NVIDIA SM80 or newer device")
const BACKEND = CUDA.CUDABackend()

const RUN = RunConfig(PRECISIONS, ROWS_PER_RUN, NREPS, NWARMUP, SEED, BACKEND;
                      check_results=CHECK_RESULTS, show_progress=SHOW_PROGRESS)

function main()
    mkpath(OUTPUT_DIR)
    write_dense_csv(joinpath(OUTPUT_DIR, "strong_scaling.csv"),
                    DenseOutputStrongScaling.run(RUN))
    write_dense_csv(joinpath(OUTPUT_DIR, "rank_sweep.csv"),
                    DenseOutputRankSweep.run(RUN))
    write_dense_csv(joinpath(OUTPUT_DIR, "tile_size_sweep.csv"),
                    DenseOutputTileSizeSweep.run(RUN))
    write_dense_csv(joinpath(OUTPUT_DIR, "matrix_shape_sweep.csv"),
                    DenseOutputMatrixShapeSweep.run(RUN))
    nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
