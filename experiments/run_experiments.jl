"""Top-level configuration and launcher for the benchmark experiments.

Edit the configuration block below, then run this file directly.  Large
operands are owned and released by the experiment functions; this script keeps
only the compact result vectors and writes each experiment to its own CSV.
"""

using KernelAbstractions

include(joinpath(@__DIR__, "strong_scaling.jl"))
include(joinpath(@__DIR__, "tlr_output.jl"))

using .StrongScalingExperiment
using .TLROutputExperiment

const BACKEND = let
    try
        @eval import CUDA
        CUDA.functional() ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    catch
        KernelAbstractions.CPU()
    end
end

const OUTPUT_DIR = joinpath(@__DIR__, "results")
const DTYPES = (Float32, Float64)

# ── Configuration block ──────────────────────────────────────────────────────

const RUN = RunConfig(DTYPES, 1, 10, 2, 20260728, BACKEND)

# The same sweep definitions can be reused for the exact-rank format.  Change
# only these fields to compare padded storage with variable-rank storage.
const COMPRESSED_RUN = RunConfig(
    DTYPES, 1, 10, 2, 20260728, BACKEND;
    format=:compressed, rank_distribution=:uniform, min_rank=8, max_rank=32,
)
const COMPRESSED_CONSTANT_RUN = RunConfig(
    DTYPES, 1, 10, 2, 20260728, BACKEND;
    format=:compressed, rank_distribution=:constant,
)
const COMPRESSED_SKEWED_RUN = RunConfig(
    DTYPES, 1, 10, 2, 20260728, BACKEND;
    format=:compressed, rank_distribution=:skewed, min_rank=8, max_rank=32,
)

const STRONG_SCALING = StrongScalingConfig(
    [4096, 8192, 16384, 32768],
    512,
    (32, 32),
    RUN,
)

const RANK_SWEEP = RankSweepConfig(
    16384,
    512,
    [8, 16, 32, 64, 128, 256],
    RUN,
)

const TILE_SIZE_SWEEP = TileSizeSweepConfig(
    16384,
    [128, 256, 512, 1024, 2048],
    32,
    RUN,
)

const MATRIX_SHAPE_SWEEP = MatrixShapeSweepConfig(
    16384,
    512,
    32,
    [(1, 1, 1), (4, 1, 1), (1, 1, 4), (1, 4, 1), (1, 0.25, 1)],
    RUN,
)

const COMPRESSED_STRONG_SCALING = StrongScalingConfig(
    [4096, 8192, 16384, 32768], 512, (32, 32), COMPRESSED_RUN)
const COMPRESSED_CONSTANT_STRONG_SCALING = StrongScalingConfig(
    [4096, 8192, 16384, 32768], 512, (32, 32), COMPRESSED_CONSTANT_RUN)
const COMPRESSED_SKEWED_STRONG_SCALING = StrongScalingConfig(
    [4096, 8192, 16384, 32768], 512, (32, 32), COMPRESSED_SKEWED_RUN)
const COMPRESSED_RANK_SWEEP = RankSweepConfig(
    16384, 512, [8, 16, 32, 64, 128, 256],
    RunConfig(DTYPES, 1, 10, 2, 20260728, BACKEND;
              format=:compressed, rank_distribution=:constant))
const COMPRESSED_TILE_SIZE_SWEEP = TileSizeSweepConfig(
    16384, [128, 256, 512, 1024, 2048], 32,
    RunConfig(DTYPES, 1, 10, 2, 20260728, BACKEND;
              format=:compressed, rank_distribution=:constant))
const COMPRESSED_MATRIX_SHAPE_SWEEP = MatrixShapeSweepConfig(
    16384, 512, 32, [(1, 1, 1), (4, 1, 1), (1, 4, 1), (1, 1, 4), (1, 0.25, 1)],
    RunConfig(DTYPES, 1, 10, 2, 20260728, BACKEND;
              format=:compressed, rank_distribution=:constant))

const OUTPUT_RUN = TLROutputRunConfig(
    DTYPES, 1, 10, 2, 20260728, BACKEND;
    block=32, tol=1e-6, rel=true,
)

const TLR_OUTPUT_STRONG_SCALING = TLROutputStrongScalingConfig(
    [4096, 8192, 16384],
    512,
    (32, 32),
    64,
    OUTPUT_RUN,
)

const TLR_OUTPUT_OVERLAP = TLROutputOverlapConfig(
    16384,
    512,
    (32, 32),
    64,
    [0, 8, 16, 24, 32],
    OUTPUT_RUN,
)

# ── CSV output ───────────────────────────────────────────────────────────────

function _csv_value(x)
    s = string(x)
    return occursin(',', s) || occursin('"', s) ?
        "\"" * replace(s, '"' => "\"\"") * "\"" : s
end

function _write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(_csv_value.(row), ','))
        end
    end
    return path
end

function _write_dense_experiment(path, results)
    header = [
        "experiment", "format", "rank_distribution", "min_rank", "max_rank",
        "dtype", "m", "k", "n", "tile_size", "rank_A", "rank_B",
        "tlr_dense_ms", "dense_tlr_ms", "tlr_tlr_ms", "dense_dense_ms",
    ]
    rows = ([
        r.experiment, r.format, r.rank_distribution, r.min_rank, r.max_rank,
        r.dtype, r.m, r.k, r.n, r.tile_size, r.rank_A, r.rank_B,
        r.timing.tlr_dense_ms, r.timing.dense_tlr_ms,
        r.timing.tlr_tlr_ms, r.timing.dense_dense_ms,
    ] for r in results)
    return _write_csv(path, header, rows)
end

function _write_tlr_output_experiment(path, results)
    header = [
        "experiment", "dtype", "m", "k", "n", "tile_size", "rank_A", "rank_B",
        "output_rank", "shared_rank", "tlr_tlr_ms", "dense_compress_ms",
        "dense_dense_ms", "tlr_tlr_rel_fro_error", "dense_compress_rel_fro_error",
    ]
    rows = ([
        r.experiment, r.dtype, r.m, r.k, r.n, r.tile_size, r.rank_A, r.rank_B,
        r.output_rank, r.shared_rank, r.timing.tlr_tlr_ms,
        r.timing.dense_compress_ms, r.timing.dense_dense_ms,
        r.timing.tlr_tlr_rel_fro_error, r.timing.dense_compress_rel_fro_error,
    ] for r in results)
    return _write_csv(path, header, rows)
end

function main()
    mkpath(OUTPUT_DIR)

    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "strong_scaling.csv"),
        strong_scaling(STRONG_SCALING))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "rank_sweep.csv"),
        rank_sweep(RANK_SWEEP))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "tile_size_sweep.csv"),
        tile_size_sweep(TILE_SIZE_SWEEP))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "matrix_shape_sweep.csv"),
        matrix_shape_sweep(MATRIX_SHAPE_SWEEP))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "compressed_strong_scaling.csv"),
        strong_scaling(COMPRESSED_STRONG_SCALING))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "compressed_constant_strong_scaling.csv"),
        strong_scaling(COMPRESSED_CONSTANT_STRONG_SCALING))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "compressed_skewed_strong_scaling.csv"),
        strong_scaling(COMPRESSED_SKEWED_STRONG_SCALING))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "compressed_rank_sweep.csv"),
        rank_sweep(COMPRESSED_RANK_SWEEP))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "compressed_tile_size_sweep.csv"),
        tile_size_sweep(COMPRESSED_TILE_SIZE_SWEEP))
    _write_dense_experiment(
        joinpath(OUTPUT_DIR, "compressed_matrix_shape_sweep.csv"),
        matrix_shape_sweep(COMPRESSED_MATRIX_SHAPE_SWEEP))
    _write_tlr_output_experiment(
        joinpath(OUTPUT_DIR, "tlr_output_strong_scaling.csv"),
        tlr_output_strong_scaling(TLR_OUTPUT_STRONG_SCALING))
    _write_tlr_output_experiment(
        joinpath(OUTPUT_DIR, "tlr_output_overlap_sweep.csv"),
        tlr_output_overlap_sweep(TLR_OUTPUT_OVERLAP))

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
