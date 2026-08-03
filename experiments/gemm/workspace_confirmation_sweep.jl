#!/usr/bin/env julia

"""Independently remeasure the workspace winners selected from a tuning CSV."""

include(joinpath(@__DIR__, "common.jl"))
using .GemmExperimentCommon

const WARMUP = parse(Int, get(ENV, "NEXTLA_CONFIRM_WARMUP", "3"))
const REPS = parse(Int, get(ENV, "NEXTLA_CONFIRM_REPS", "10"))
const ANALYSIS_REPS = parse(Int, get(ENV, "NEXTLA_CONFIRM_ANALYSIS_REPS", "3"))
const CASE_FILTER = Regex(get(ENV, "NEXTLA_CONFIRM_FILTER", ".*"))

function parse_csv_record(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if character == '"'
            following = nextind(line, index)
            if quoted && following <= lastindex(line) && line[following] == '"'
                write(buffer, '"')
                index = following
            else
                quoted = !quoted
            end
        elseif character == ',' && !quoted
            push!(fields, String(take!(buffer)))
        else
            write(buffer, character)
        end
        index = nextind(line, index)
    end
    quoted && throw(ArgumentError("unterminated quoted CSV field"))
    push!(fields, String(take!(buffer)))
    return fields
end

function read_winner_rows(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("winner CSV is empty: $path"))
    header = parse_csv_record(first(lines))
    required = Set([
        "experiment", "record_kind", "case_id", "N", "precision",
        "operand_layout", "tile_divisor", "tile_size", "distribution",
        "rank_band", "min_rank", "max_rank", "execution_rank_policy",
        "seed", "factor_fill", "workspace_parameter",
    ])
    missing = setdiff(required, Set(header))
    isempty(missing) || throw(ArgumentError(
        "winner CSV is missing columns $(join(sort!(collect(missing)), ", "))"))
    rows = Dict{String,String}[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        line_number = offset + 1
        isempty(line) && continue
        values = parse_csv_record(line)
        length(values) == length(header) || throw(ArgumentError(
            "CSV row $line_number has $(length(values)) fields; " *
            "expected $(length(header))"))
        row = Dict(zip(header, values))
        row["record_kind"] == "compressed" || continue
        row["experiment"] == "workspace_winners" || throw(ArgumentError(
            "expected workspace_winners input, got $(row["experiment"])"))
        occursin(CASE_FILTER, row["case_id"]) && push!(rows, row)
    end
    isempty(rows) && throw(ArgumentError(
        "winner CSV contains no selected compressed cases"))
    return rows
end

row_int(row, name) = parse(Int, row[name])

function winner_key(row)
    return (
        N=row_int(row, "N"), precision=row["precision"],
        layout=row["operand_layout"], divisor=row_int(row, "tile_divisor"),
        band=row["rank_band"], distribution=row["distribution"],
    )
end

function validate_configuration(rows)
    WARMUP >= 0 || throw(ArgumentError("warmup count must be nonnegative"))
    REPS > 0 || throw(ArgumentError("repetition count must be positive"))
    ANALYSIS_REPS > 0 || throw(ArgumentError(
        "analysis repetition count must be positive"))
    unknown = setdiff(
        Set(row["precision"] for row in rows), Set(keys(PRECISION_TABLE)))
    isempty(unknown) || throw(ArgumentError(
        "unknown precisions in winner CSV: $(join(sort!(collect(unknown)), ", "))"))
    for row in rows
        layout = Symbol(row["operand_layout"])
        layout in (:compressed_dense, :dense_compressed, :compressed_compressed) ||
            throw(ArgumentError("unknown operand layout $layout"))
        distribution = Symbol(row["distribution"])
        distribution in (:uniform, :skewed, :constant) || throw(ArgumentError(
            "unknown rank distribution $distribution"))
        row_int(row, "workspace_parameter") > 0 || throw(ArgumentError(
            "workspace parameter must be positive"))
    end
    return nothing
end

function list_cases(rows)
    for row in sort(rows; by=winner_key)
        println(row["case_id"])
    end
    println("$(length(rows)) selected cases")
    return nothing
end

function run_confirmation(path::AbstractString)
    rows = read_winner_rows(path)
    validate_configuration(rows)
    "--list" in ARGS && return list_cases(rows)
    precision_names = sort!(unique(row["precision"] for row in rows))
    validate_cuda_precisions([PRECISION_TABLE[name] for name in precision_names])
    run = fresh_csv("workspace_confirmation", "NEXTLA_CONFIRM_OUTPUT")
    println("Workspace confirmation on $(gpu_name())")
    println("Input: $(abspath(path))")
    println("Output: $(run.path)")
    try
        groups = sort!(unique(
            (N=row_int(row, "N"), precision=row["precision"])
            for row in rows))
        for group in groups
            selected = sort!(
                [row for row in rows if
                    row_int(row, "N") == group.N &&
                    row["precision"] == group.precision];
                by=winner_key)
            seeds = unique(row_int(row, "seed") for row in selected)
            fills = unique(Symbol(row["factor_fill"]) for row in selected)
            length(seeds) == 1 || throw(ArgumentError(
                "confirmation cases for N=$(group.N), $(group.precision) " *
                "must share one seed"))
            length(fills) == 1 || throw(ArgumentError(
                "confirmation cases for N=$(group.N), $(group.precision) " *
                "must share one fill mode"))
            seed, fill_mode = only(seeds), only(fills)
            precision = PRECISION_TABLE[group.precision]
            dense = benchmark_dense(
                group.N, precision; warmup=WARMUP, repetitions=REPS,
                seed, fill_mode)
            base = baseline_row(
                run, group.N, precision, dense; warmup=WARMUP,
                repetitions=REPS, seed, fill_mode)
            write_csv_row(run, base)
            print_case(base)

            for source in selected
                divisor = row_int(source, "tile_divisor")
                b = row_int(source, "tile_size")
                b == group.N ÷ divisor || throw(ArgumentError(
                    "inconsistent tile size in $(source["case_id"])"))
                level = row_int(source, "workspace_parameter")
                distribution = Symbol(source["distribution"])
                layout = Symbol(source["operand_layout"])
                policy = Symbol(source["execution_rank_policy"])
                lo, hi = row_int(source, "min_rank"), row_int(source, "max_rank")
                measured = benchmark_compressed_case(
                    group.N, b, distribution, lo, hi, precision, layout;
                    warmup=WARMUP, repetitions=REPS,
                    analysis_repetitions=ANALYSIS_REPS, seed, fill_mode,
                    execution_rank_policy=policy, runs=level,
                    mixed_stripes=level)
                measured.workspace_parameter == level || error(
                    "workspace parameter changed while confirming " *
                    source["case_id"])
                row = compressed_row(
                    run, group.N, divisor, b, distribution,
                    source["rank_band"], lo, hi, precision, layout, policy,
                    seed, fill_mode, WARMUP, REPS, ANALYSIS_REPS, measured,
                    dense)
                write_csv_row(run, row)
                print_case(row)
            end
        end
    finally
        close_csv(run)
    end
    println("Completed: $(run.path)")
    return run.path
end

function input_path()
    paths = [argument for argument in ARGS if argument != "--list"]
    length(paths) == 1 || throw(ArgumentError(
        "usage: workspace_confirmation_sweep.jl WINNERS.csv [--list]"))
    isfile(only(paths)) || throw(ArgumentError(
        "winner CSV does not exist: $(only(paths))"))
    return only(paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_confirmation(input_path())
end
