#!/usr/bin/env julia

"""Tune numerical workspace independently for every compressed GEMM case."""

include(joinpath(@__DIR__, "common.jl"))
using .GemmExperimentCommon

const SIZES = parse_int_list(
    "NEXTLA_TUNING_SIZES", "4096,8192,16384,32768,65536")
const TILE_DIVISORS = parse_int_list("NEXTLA_TUNING_TILE_DIVISORS", "16,8")
const DISTRIBUTIONS = parse_symbol_list(
    "NEXTLA_TUNING_DISTRIBUTIONS", "uniform,skewed")
const LAYOUTS = parse_symbol_list(
    "NEXTLA_TUNING_LAYOUTS",
    "compressed_dense,dense_compressed,compressed_compressed")
const PRECISIONS = selected_precisions(
    "NEXTLA_TUNING_PRECISIONS", "bf16,fp16,fp32,tf32")
const RANK_BANDS = let value = strip(get(ENV, "NEXTLA_TUNING_RANK_BANDS", "32:16"))
    isempty(value) ? String[] : strip.(split(value, ','))
end
const FIXED_RANKS = let value = strip(get(ENV, "NEXTLA_TUNING_FIXED_RANKS", ""))
    isempty(value) ? Int[] : parse.(Int, split(value, ','))
end
const RUN_SPEC = strip(get(
    ENV, "NEXTLA_TUNING_WORKSPACE_LEVELS", "1,2,4,8,16,32,64"))
const MIXED_STRIPE_SPEC = strip(get(
    ENV, "NEXTLA_TUNING_MIXED_STRIPES", "all"))
const WARMUP = parse(Int, get(ENV, "NEXTLA_TUNING_WARMUP", "1"))
const REPS = parse(Int, get(ENV, "NEXTLA_TUNING_REPS", "5"))
const ANALYSIS_REPS = parse(Int, get(ENV, "NEXTLA_TUNING_ANALYSIS_REPS", "1"))
const SEED = parse(Int, get(ENV, "NEXTLA_TUNING_SEED", "20260802"))
const FILL_MODE = Symbol(get(ENV, "NEXTLA_TUNING_FILL", "random"))
const EXECUTION_POLICY = Symbol(get(
    ENV, "NEXTLA_TUNING_EXECUTION_POLICY", "q8"))
const CASE_FILTER = Regex(get(ENV, "NEXTLA_TUNING_FILTER", ".*"))

function parse_workspace_levels(spec::AbstractString; all_levels)
    lowercase(spec) == "all" && return collect(all_levels)
    levels = Int[]
    for token in split(spec, ',')
        pieces = split(strip(token), ':')
        if length(pieces) == 1
            push!(levels, parse(Int, only(pieces)))
        elseif length(pieces) == 2
            first_level, last_level = parse.(Int, pieces)
            first_level <= last_level || throw(ArgumentError(
                "workspace range '$token' must be increasing"))
            append!(levels, first_level:last_level)
        else
            throw(ArgumentError(
                "workspace token '$token' must be an integer or FIRST:LAST"))
        end
    end
    all(>(0), levels) || throw(ArgumentError(
        "workspace levels must be positive"))
    return sort!(unique(levels))
end

function workspace_levels(q::Int, layout::Symbol)
    if layout === :compressed_compressed
        # This is a target run count: larger values request less workspace and
        # remain meaningful above q because the scheduler column-blocks.
        return parse_workspace_levels(RUN_SPEC; all_levels=1:64)
    end
    # Mixed GEMM still uses physical tile stripes. Values above q describe the
    # same full-width allocation and would only duplicate a measurement.
    levels = parse_workspace_levels(MIXED_STRIPE_SPEC; all_levels=1:q)
    return sort!(unique(min(level, q) for level in levels))
end

function rank_cases(b::Int)
    cases = NamedTuple[]
    seen = Set{Tuple{Symbol,Int,Int}}()
    for spec in RANK_BANDS
        band = rank_band(spec, b)
        for distribution in DISTRIBUTIONS
            key = (distribution, band.lo, band.hi)
            key in seen && continue
            push!(seen, key)
            push!(cases, (distribution, name=band.name,
                          lo=band.lo, hi=band.hi))
        end
    end
    for rank in FIXED_RANKS
        rank <= b || continue
        key = (:constant, rank, rank)
        key in seen && continue
        push!(seen, key)
        push!(cases, (distribution=:constant, name="r$(rank)",
                      lo=rank, hi=rank))
    end
    return cases
end

function validate_configuration()
    all(>(0), SIZES) || throw(ArgumentError("matrix sizes must be positive"))
    all(>(0), TILE_DIVISORS) || throw(ArgumentError(
        "tile divisors must be positive"))
    all(>(0), FIXED_RANKS) || throw(ArgumentError(
        "fixed ranks must be positive"))
    isempty(RANK_BANDS) && isempty(FIXED_RANKS) && throw(ArgumentError(
        "at least one rank band or fixed rank is required"))
    all(distribution -> distribution in (:uniform, :skewed), DISTRIBUTIONS) ||
        throw(ArgumentError("distributions must be uniform or skewed"))
    all(layout -> layout in (
            :compressed_dense, :dense_compressed, :compressed_compressed),
        LAYOUTS) || throw(ArgumentError("unknown operand layout"))
    EXECUTION_POLICY in (:exact, :q8, :q16, :pow2) || throw(ArgumentError(
        "execution policy must be exact, q8, q16, or pow2"))
    FILL_MODE in (:random, :constant, :zeros) || throw(ArgumentError(
        "fill mode must be random, constant, or zeros"))
    WARMUP >= 0 || throw(ArgumentError("warmup count must be nonnegative"))
    REPS > 0 || throw(ArgumentError("repetition count must be positive"))
    ANALYSIS_REPS > 0 || throw(ArgumentError(
        "analysis repetition count must be positive"))
    for N in SIZES, divisor in TILE_DIVISORS
        N % divisor == 0 || throw(ArgumentError(
            "N=$N is not divisible by tile divisor $divisor"))
        for spec in RANK_BANDS
            rank_band(spec, N ÷ divisor)
        end
        workspace_levels(divisor, :compressed_compressed)
        workspace_levels(divisor, :compressed_dense)
    end
    return nothing
end

function selected_levels(N, divisor, rank_case, precision, layout)
    result = Int[]
    for level in workspace_levels(divisor, layout)
        id = compressed_case_id(
            N, divisor, rank_case.distribution, rank_case.lo, rank_case.hi,
            precision, layout, EXECUTION_POLICY, level)
        occursin(CASE_FILTER, id) && push!(result, level)
    end
    return result
end

function list_cases()
    count = 0
    for N in SIZES, precision in PRECISIONS
        println(baseline_case_id(N, precision))
        count += 1
        for divisor in TILE_DIVISORS
            b = N ÷ divisor
            for rank_case in rank_cases(b), layout in LAYOUTS
                for level in selected_levels(
                    N, divisor, rank_case, precision, layout)
                    println(compressed_case_id(
                        N, divisor, rank_case.distribution,
                        rank_case.lo, rank_case.hi, precision, layout,
                        EXECUTION_POLICY, level))
                    count += 1
                end
            end
        end
    end
    println("$count CSV rows (including dense baselines)")
    return nothing
end

function run_workspace_tuning()
    validate_configuration()
    "--list" in ARGS && return list_cases()
    validate_cuda_precisions(PRECISIONS)
    run = fresh_csv("workspace_tuning", "NEXTLA_TUNING_OUTPUT")
    println("Workspace tuning on $(gpu_name())")
    println("Output: $(run.path)")
    try
        for N in SIZES, precision in PRECISIONS
            dense = benchmark_dense(
                N, precision; warmup=WARMUP, repetitions=REPS,
                seed=SEED, fill_mode=FILL_MODE)
            base = baseline_row(
                run, N, precision, dense; warmup=WARMUP,
                repetitions=REPS, seed=SEED, fill_mode=FILL_MODE)
            write_csv_row(run, base)
            print_case(base)

            for divisor in TILE_DIVISORS
                b = N ÷ divisor
                for rank_case in rank_cases(b), layout in LAYOUTS
                    levels = selected_levels(
                        N, divisor, rank_case, precision, layout)
                    isempty(levels) && continue
                    measured = benchmark_compressed_workspace_sweep(
                        N, b, rank_case.distribution, rank_case.lo,
                        rank_case.hi, precision, layout, levels;
                        warmup=WARMUP, repetitions=REPS,
                        analysis_repetitions=ANALYSIS_REPS, seed=SEED,
                        fill_mode=FILL_MODE,
                        execution_rank_policy=EXECUTION_POLICY)
                    previous_ratio = layout === :compressed_compressed ? Inf : -Inf
                    for result in measured
                        monotone = layout === :compressed_compressed ?
                            result.memory_ratio <= previous_ratio :
                            result.memory_ratio >= previous_ratio
                        monotone || error(
                            "workspace sweep is not monotone for N=$N, " *
                            "q=$divisor, layout=$layout")
                        previous_ratio = result.memory_ratio
                        row = compressed_row(
                            run, N, divisor, b, rank_case.distribution,
                            rank_case.name, rank_case.lo, rank_case.hi,
                            precision, layout, EXECUTION_POLICY, SEED,
                            FILL_MODE, WARMUP, REPS, ANALYSIS_REPS, result,
                            dense)
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
    run_workspace_tuning()
end
