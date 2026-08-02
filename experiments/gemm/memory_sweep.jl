#!/usr/bin/env julia

"""Fixed-size speedup versus measured operand-plus-workspace storage ratio."""

include(joinpath(@__DIR__, "common.jl"))
using .GemmExperimentCommon

const N = parse(Int, get(ENV, "NEXTLA_MEMORY_N", "16384"))
const TILE_DIVISORS = parse_int_list("NEXTLA_MEMORY_TILE_DIVISORS", "16,8")
const DISTRIBUTIONS = parse_symbol_list(
    "NEXTLA_MEMORY_DISTRIBUTIONS", "uniform,skewed")
const LAYOUTS = parse_symbol_list(
    "NEXTLA_MEMORY_LAYOUTS",
    "compressed_dense,dense_compressed,compressed_compressed")
const RANK_BANDS = parse_string_list(
    "NEXTLA_MEMORY_RANK_BANDS", "16:8")
const WORKSPACE_LEVELS = parse_int_list(
    "NEXTLA_MEMORY_WORKSPACE_LEVELS", "1,2,4,8,16")
const PRECISIONS = selected_precisions(
    "NEXTLA_MEMORY_PRECISION", "fp16")
const WARMUP = parse(Int, get(ENV, "NEXTLA_MEMORY_WARMUP", "1"))
const REPS = parse(Int, get(ENV, "NEXTLA_MEMORY_REPS", "5"))
const ANALYSIS_REPS = parse(Int, get(ENV, "NEXTLA_MEMORY_ANALYSIS_REPS", "3"))
const SEED = parse(Int, get(ENV, "NEXTLA_MEMORY_SEED", "20260802"))
const FILL_MODE = Symbol(get(ENV, "NEXTLA_MEMORY_FILL", "random"))
const EXECUTION_POLICY = Symbol(get(
    ENV, "NEXTLA_MEMORY_EXECUTION_POLICY", "q8"))
const CASE_FILTER = Regex(get(ENV, "NEXTLA_MEMORY_FILTER", ".*"))

function validate_configuration()
    N > 0 || throw(ArgumentError("N must be positive"))
    length(PRECISIONS) == 1 || throw(ArgumentError(
        "NEXTLA_MEMORY_PRECISION must select exactly one precision"))
    all(d -> d in (:uniform, :skewed), DISTRIBUTIONS) ||
        throw(ArgumentError("distributions must be uniform or skewed"))
    all(l -> l in (:compressed_dense, :dense_compressed, :compressed_compressed),
        LAYOUTS) || throw(ArgumentError("unknown operand layout"))
    EXECUTION_POLICY in (:exact, :q8, :q16, :pow2) ||
        throw(ArgumentError("execution policy must be exact, q8, q16, or pow2"))
    FILL_MODE in (:random, :constant, :zeros) ||
        throw(ArgumentError("fill mode must be random, constant, or zeros"))
    all(>(0), WORKSPACE_LEVELS) ||
        throw(ArgumentError("workspace levels must be positive"))
    for divisor in TILE_DIVISORS
        divisor > 0 && N % divisor == 0 || throw(ArgumentError(
            "N=$N is not divisible by tile divisor $divisor"))
        for spec in RANK_BANDS
            rank_band(spec, N ÷ divisor)
        end
    end
    return nothing
end

workspace_levels(divisor) =
    sort!(unique(min(level, divisor) for level in WORKSPACE_LEVELS))

function list_cases()
    precision = only(PRECISIONS)
    count = 1
    println(baseline_case_id(N, precision))
    for divisor in TILE_DIVISORS
        b = N ÷ divisor
        for spec in RANK_BANDS
            band = rank_band(spec, b)
            for distribution in DISTRIBUTIONS, layout in LAYOUTS,
                workspace_level in workspace_levels(divisor)
                id = compressed_case_id(
                    N, divisor, distribution, band.lo, band.hi, precision,
                    layout, EXECUTION_POLICY, workspace_level)
                occursin(CASE_FILTER, id) || continue
                println(id)
                count += 1
            end
        end
    end
    println("$count CSV rows (including one dense baseline)")
    return nothing
end

function run_memory_sweep()
    validate_configuration()
    "--list" in ARGS && return list_cases()
    validate_cuda_precisions(PRECISIONS)
    precision = only(PRECISIONS)
    run = fresh_csv("memory_sweep", "NEXTLA_MEMORY_OUTPUT")
    println("Memory sweep on $(gpu_name())")
    println("Output: $(run.path)")
    try
        dense = benchmark_dense(
            N, precision; warmup=WARMUP, repetitions=REPS,
            seed=SEED, fill_mode=FILL_MODE)
        base = baseline_row(
            run, N, precision, dense; warmup=WARMUP, repetitions=REPS,
            seed=SEED, fill_mode=FILL_MODE)
        write_csv_row(run, base)
        print_case(base)

        for divisor in TILE_DIVISORS
            b = N ÷ divisor
            for spec in RANK_BANDS
                band = rank_band(spec, b)
                for distribution in DISTRIBUTIONS, layout in LAYOUTS
                    previous_ratio = -Inf
                    for workspace_level in workspace_levels(divisor)
                        id = compressed_case_id(
                            N, divisor, distribution, band.lo, band.hi,
                            precision, layout, EXECUTION_POLICY, workspace_level)
                        occursin(CASE_FILTER, id) || continue
                        measured = benchmark_compressed_case(
                            N, b, distribution, band.lo, band.hi, precision,
                            layout; warmup=WARMUP, repetitions=REPS,
                            analysis_repetitions=ANALYSIS_REPS, seed=SEED,
                            fill_mode=FILL_MODE,
                            execution_rank_policy=EXECUTION_POLICY,
                            rows_per_run=workspace_level,
                            mixed_stripes=workspace_level)
                        measured.memory_ratio >= previous_ratio || error(
                            "workspace sweep is not monotone for $id")
                        previous_ratio = measured.memory_ratio
                        row = compressed_row(
                            run, N, divisor, b, distribution, band.name,
                            band.lo, band.hi, precision, layout,
                            EXECUTION_POLICY, SEED, FILL_MODE, WARMUP, REPS,
                            ANALYSIS_REPS, measured, dense)
                        write_csv_row(run, row)
                        print_case(row)
                    end
                end
            end
        end
    finally
        close_csv(run)
    end
    println("Completed: $(run.path)")
    return run.path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_memory_sweep()
end
