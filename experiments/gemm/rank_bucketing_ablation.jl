#!/usr/bin/env julia

"""Compact bar-plot ablation for execution-rank bucketing policies."""

include(joinpath(@__DIR__, "common.jl"))
using .GemmExperimentCommon

const N = parse(Int, get(ENV, "NEXTLA_ABLATION_N", "16384"))
const TILE_DIVISOR = parse(Int, get(ENV, "NEXTLA_ABLATION_TILE_DIVISOR", "16"))
const DISTRIBUTIONS = parse_symbol_list(
    "NEXTLA_ABLATION_DISTRIBUTIONS", "uniform,skewed")
const POLICIES = parse_symbol_list(
    "NEXTLA_ABLATION_POLICIES", "exact,q8,q16,pow2")
const BAND_SPEC = get(ENV, "NEXTLA_ABLATION_RANK_BAND", "32:8")
const PRECISIONS = selected_precisions("NEXTLA_ABLATION_PRECISION", "fp16")
const WARMUP = parse(Int, get(ENV, "NEXTLA_ABLATION_WARMUP", "1"))
const REPS = parse(Int, get(ENV, "NEXTLA_ABLATION_REPS", "10"))
const ANALYSIS_REPS = parse(Int, get(ENV, "NEXTLA_ABLATION_ANALYSIS_REPS", "3"))
const RUNS = parse(Int, get(ENV, "NEXTLA_ABLATION_RUNS", "1"))
const SEED = parse(Int, get(ENV, "NEXTLA_ABLATION_SEED", "20260802"))
const FILL_MODE = Symbol(get(ENV, "NEXTLA_ABLATION_FILL", "random"))

function validate_configuration()
    N > 0 && TILE_DIVISOR > 0 && N % TILE_DIVISOR == 0 ||
        throw(ArgumentError("N must be positive and divisible by the tile divisor"))
    length(PRECISIONS) == 1 || throw(ArgumentError(
        "NEXTLA_ABLATION_PRECISION must select exactly one precision"))
    all(d -> d in (:uniform, :skewed), DISTRIBUTIONS) ||
        throw(ArgumentError("distributions must be uniform or skewed"))
    all(p -> p in (:exact, :q8, :q16, :pow2), POLICIES) ||
        throw(ArgumentError("policies must be exact, q8, q16, or pow2"))
    FILL_MODE in (:random, :constant, :zeros) ||
        throw(ArgumentError("fill mode must be random, constant, or zeros"))
    RUNS > 0 || throw(ArgumentError("run count must be positive"))
    rank_band(BAND_SPEC, N ÷ TILE_DIVISOR)
    return nothing
end

function list_cases()
    precision = only(PRECISIONS)
    b = N ÷ TILE_DIVISOR
    band = rank_band(BAND_SPEC, b)
    println(baseline_case_id(N, precision))
    for distribution in DISTRIBUTIONS, policy in POLICIES
        println(compressed_case_id(
            N, TILE_DIVISOR, distribution, band.lo, band.hi, precision,
            :compressed_compressed, policy, RUNS))
    end
    println("$(1 + length(DISTRIBUTIONS) * length(POLICIES)) CSV rows " *
            "(including one dense baseline)")
    return nothing
end

function run_ablation()
    validate_configuration()
    "--list" in ARGS && return list_cases()
    validate_cuda_precisions(PRECISIONS)
    precision = only(PRECISIONS)
    b = N ÷ TILE_DIVISOR
    band = rank_band(BAND_SPEC, b)
    run = fresh_csv("rank_bucketing_ablation", "NEXTLA_ABLATION_OUTPUT")
    println("Rank-bucketing ablation on $(gpu_name())")
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

        for distribution in DISTRIBUTIONS, policy in POLICIES
            measured = benchmark_compressed_case(
                N, b, distribution, band.lo, band.hi, precision,
                :compressed_compressed; warmup=WARMUP, repetitions=REPS,
                analysis_repetitions=ANALYSIS_REPS, seed=SEED,
                fill_mode=FILL_MODE, execution_rank_policy=policy,
                runs=RUNS, mixed_stripes=1)
            row = compressed_row(
                run, N, TILE_DIVISOR, b, distribution, band.name,
                band.lo, band.hi, precision, :compressed_compressed, policy,
                SEED, FILL_MODE, WARMUP, REPS, ANALYSIS_REPS, measured, dense)
            write_csv_row(run, row)
            print_case(row)
        end
    finally
        close_csv(run)
    end
    println("Completed: $(run.path)")
    return run.path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ablation()
end
