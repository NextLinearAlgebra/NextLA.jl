#!/usr/bin/env julia

"""Size and precision sweep for all dense-output compressed GEMM layouts."""

include(joinpath(@__DIR__, "common.jl"))
using .GemmExperimentCommon

const SIZES = parse_int_list(
    "NEXTLA_PRECISION_SIZES", "4096,8192,16384,32768,65536")
const TILE_DIVISORS = parse_int_list("NEXTLA_PRECISION_TILE_DIVISORS", "16,8")
const DISTRIBUTIONS = parse_symbol_list(
    "NEXTLA_PRECISION_DISTRIBUTIONS", "uniform,skewed")
const LAYOUTS = parse_symbol_list(
    "NEXTLA_PRECISION_LAYOUTS",
    "compressed_dense,dense_compressed,compressed_compressed")
const PRECISIONS = selected_precisions(
    "NEXTLA_PRECISION_PRECISIONS", "bf16,fp16,fp32,tf32")
const MIN_RANK_DIVISOR = parse(Int, get(
    ENV, "NEXTLA_PRECISION_MIN_RANK_DIVISOR", "16"))
const MAX_RANK_DIVISOR = parse(Int, get(
    ENV, "NEXTLA_PRECISION_MAX_RANK_DIVISOR", "8"))
const WARMUP = parse(Int, get(ENV, "NEXTLA_PRECISION_WARMUP", "1"))
const REPS = parse(Int, get(ENV, "NEXTLA_PRECISION_REPS", "3"))
const ANALYSIS_REPS = parse(Int, get(
    ENV, "NEXTLA_PRECISION_ANALYSIS_REPS", "3"))
const RUNS = parse(Int, get(ENV, "NEXTLA_PRECISION_RUNS", "1"))
const MIXED_STRIPES = parse(Int, get(
    ENV, "NEXTLA_PRECISION_MIXED_STRIPES", "1"))
const SEED = parse(Int, get(ENV, "NEXTLA_PRECISION_SEED", "20260802"))
const FILL_MODE = Symbol(get(ENV, "NEXTLA_PRECISION_FILL", "random"))
const EXECUTION_POLICY = Symbol(get(
    ENV, "NEXTLA_PRECISION_EXECUTION_POLICY", "q8"))
const CASE_FILTER = Regex(get(ENV, "NEXTLA_PRECISION_FILTER", ".*"))

function validate_configuration()
    all(>(0), SIZES) || throw(ArgumentError("matrix sizes must be positive"))
    all(>(0), TILE_DIVISORS) || throw(ArgumentError("tile divisors must be positive"))
    all(d -> d in (:uniform, :skewed), DISTRIBUTIONS) ||
        throw(ArgumentError("distributions must be uniform or skewed"))
    all(l -> l in (:compressed_dense, :dense_compressed, :compressed_compressed),
        LAYOUTS) || throw(ArgumentError("unknown operand layout"))
    EXECUTION_POLICY in (:exact, :q8, :q16, :pow2) ||
        throw(ArgumentError("execution policy must be exact, q8, q16, or pow2"))
    FILL_MODE in (:random, :constant, :zeros) ||
        throw(ArgumentError("fill mode must be random, constant, or zeros"))
    RUNS > 0 || throw(ArgumentError("run count must be positive"))
    MIXED_STRIPES > 0 || throw(ArgumentError("mixed stripes must be positive"))
    for N in SIZES, divisor in TILE_DIVISORS
        N % divisor == 0 || throw(ArgumentError(
            "N=$N is not divisible by tile divisor $divisor"))
        rank_interval(N ÷ divisor, MIN_RANK_DIVISOR, MAX_RANK_DIVISOR)
    end
    return nothing
end

function cases()
    result = NamedTuple[]
    for N in SIZES, precision in PRECISIONS
        push!(result, (kind=:baseline, id=baseline_case_id(N, precision),
                       N, precision))
        for divisor in TILE_DIVISORS
            b = N ÷ divisor
            lo, hi = rank_interval(b, MIN_RANK_DIVISOR, MAX_RANK_DIVISOR)
            for distribution in DISTRIBUTIONS, layout in LAYOUTS
                id = compressed_case_id(
                    N, divisor, distribution, lo, hi, precision, layout,
                    EXECUTION_POLICY,
                    layout === :compressed_compressed ?
                        RUNS : MIXED_STRIPES)
                occursin(CASE_FILTER, id) || continue
                push!(result, (kind=:compressed, id, N, precision, divisor, b,
                               distribution, layout, lo, hi))
            end
        end
    end
    return result
end

function list_cases()
    selected = cases()
    for case in selected
        println(case.id)
    end
    println("$(length(selected)) CSV rows (including dense baselines)")
    return nothing
end

function run_precision_sweep()
    validate_configuration()
    "--list" in ARGS && return list_cases()
    validate_cuda_precisions(PRECISIONS)

    run = fresh_csv("precision_sweep", "NEXTLA_PRECISION_OUTPUT")
    println("Precision sweep on $(gpu_name())")
    println("Output: $(run.path)")
    try
        for N in SIZES, precision in PRECISIONS
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
                lo, hi = rank_interval(b, MIN_RANK_DIVISOR, MAX_RANK_DIVISOR)
                band_name = "b$(MIN_RANK_DIVISOR)_b$(MAX_RANK_DIVISOR)"
                for distribution in DISTRIBUTIONS, layout in LAYOUTS
                    id = compressed_case_id(
                        N, divisor, distribution, lo, hi, precision, layout,
                        EXECUTION_POLICY,
                        layout === :compressed_compressed ?
                            RUNS : MIXED_STRIPES)
                    occursin(CASE_FILTER, id) || continue
                    measured = benchmark_compressed_case(
                        N, b, distribution, lo, hi, precision, layout;
                        warmup=WARMUP, repetitions=REPS,
                        analysis_repetitions=ANALYSIS_REPS, seed=SEED,
                        fill_mode=FILL_MODE,
                        execution_rank_policy=EXECUTION_POLICY,
                        runs=RUNS, mixed_stripes=MIXED_STRIPES)
                    row = compressed_row(
                        run, N, divisor, b, distribution, band_name, lo, hi,
                        precision, layout, EXECUTION_POLICY, SEED, FILL_MODE,
                        WARMUP, REPS, ANALYSIS_REPS, measured, dense)
                    write_csv_row(run, row)
                    print_case(row)
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
    run_precision_sweep()
end
