"""Shared configuration for the GEMM benchmark suite.

The benchmark implementations deliberately do not read `ENV` or `ARGS`.
Change the defaults below for a site-wide setup, or pass the same options to
`run_gemm_benchmark.sh`/`run.jl` for a one-off or job-array run.
"""
module GemmBenchmarksConfig

export GemmBenchmarkConfig, load_config, benchmark_from_args, output_path,
       print_help

const CONFIG_DIR = @__DIR__
const DEFAULT_OUTPUT_DIR = joinpath(CONFIG_DIR, "results")

# Site/user defaults. These are the only lines normally needed when preparing
# a batch environment; command-line options and NEXTLA_GEMM_* variables take
# precedence over them.
const DEFAULT_BACKEND = "auto"
const DEFAULT_REPS = "10"
const DEFAULT_WARMUP = "2"
const DEFAULT_SEED = "20260728"
const DEFAULT_SHARD_COUNT = "1"
const DEFAULT_SHARD_INDEX = "1"
const DEFAULT_CASE_REGEX = ".*"
const DEFAULT_PRECISIONS = "float16,float32,float64"
const DEFAULT_WORKSPACE_FRACTIONS = "0.125,0.25,0.5,0.75,1.0"
const DEFAULT_MEMORY_SAFETY = "0.90"
const DEFAULT_TILE_SIZE = "32"

# Add or remove entries here to define the complete benchmark grid. Each case
# owns its dimensions, tile sizes, and maximum operand ranks.
const GEMM_CASES = [
    (m=8192, k=8192, n=8192, bm=1024, bk=1024, bn=1024,
     maxrank_A=64, maxrank_B=32),
]

struct GemmBenchmarkConfig
    benchmark::Symbol
    backend::Symbol
    reps::Int
    warmup::Int
    seed::Int
    output_dir::String
    shard_count::Int
    shard_index::Int
    case_regex::Regex
    precisions::Vector{Symbol}
    workspace_fractions::Vector{Float64}
    workspace_memory_safety::Float64
    cases::Vector{<:NamedTuple}
    tile_size::Int
end

function print_help(io::IO=stdout)
    println(io, "Usage: run_gemm_benchmark.sh [options]")
    println(io)
    println(io, "Benchmarks: dense, workspace, tlr-output (default: dense)")
    println(io, "Common options:")
    println(io, "  --benchmark NAME       benchmark to run")
    println(io, "  --backend auto|cuda|cpu backend (default: auto)")
    println(io, "  --reps N               timed repetitions (default: 10)")
    println(io, "  --warmup N             warmup repetitions (default: 2)")
    println(io, "  --seed N               random seed (default: 20260728)")
    println(io, "  --output-dir DIR       CSV directory (default: scripts/gemm_benchmarks/results)")
    println(io, "  --shard-count N        number of job-array shards (default: 1)")
    println(io, "  --shard-index N        one-based shard index (default: 1)")
    println(io, "  --case-regex REGEX     restrict case IDs (default: .*)")
    println(io, "  precisions, dimensions, tiles, and ranks are set in config.jl")
    println(io, "Workspace options:")
    println(io, "  --workspace-fractions LIST   e.g. 0.125,0.25,0.5,1.0")
    println(io, "  --memory-safety FRACTION     GPU safety fraction (default: 0.90)")
    println(io, "TLR-output options:")
    println(io)
    println(io, "The same common options apply to all three benchmarks. Each option may")
    println(io, "also be supplied through the NEXTLA_GEMM_* environment variables.")
end

_key(name) = replace(lowercase(name), '-' => '_')

function _options(args)
    options = Dict{String,String}()
    allowed = Set((
        "benchmark", "backend", "reps", "warmup", "seed", "output_dir",
        "shard_count", "shard_index", "case_regex", "workspace_fractions",
        "memory_safety",
    ))
    i = 1
    while i <= length(args)
        arg = String(args[i])
        arg == "--help" && (print_help(); exit(0))
        startswith(arg, "--") || error("unexpected argument `$arg`; use --help")
        body = arg[3:end]
        name, value = if occursin('=', body)
            split(body, '='; limit=2)
        else
            i += 1
            i <= length(args) || error("missing value for `$arg`")
            body, String(args[i])
        end
        name = _key(name)
        name in allowed || error("unknown option `--$name`; use --help")
        options[name] = value
        i += 1
    end
    return options
end

function _env(names, default)
    for name in names
        haskey(ENV, name) && return ENV[name]
    end
    return default
end

function _value(options, key, env_names, default)
    return get(options, key, _env(env_names, default))
end

function _list(::Type{T}, value, name) where {T}
    values = T[]
    for item in split(value, ',')
        item = strip(item)
        isempty(item) && error("$name contains an empty value")
        push!(values, parse(T, item))
    end
    isempty(values) && error("$name must not be empty")
    return values
end

function _precisions(value)
    result = Symbol[]
    for item in split(value, ',')
        normalized = lowercase(strip(item))
        normalized = normalized == "fp32" ? "float32" :
                     normalized == "fp64" ? "float64" : normalized
        normalized in ("float16", "float32", "float64") ||
            error("precisions must contain only float16, float32, and float64")
        precision = Symbol(normalized)
        precision in result || push!(result, precision)
    end
    isempty(result) && error("precisions must not be empty")
    return result
end

function _symbol(value, name, allowed)
    result = Symbol(lowercase(strip(value)))
    result in allowed || error("$name must be one of $(join(allowed, ", "))")
    return result
end

function load_config(args=ARGS; default_benchmark::Symbol=:dense)
    options = _options(args)
    benchmark = _symbol(
        _value(options, "benchmark", ("NEXTLA_GEMM_BENCHMARK",), string(default_benchmark)),
        "benchmark", (:dense, :workspace, Symbol("tlr-output")))
    backend = _symbol(
        _value(options, "backend", ("NEXTLA_GEMM_BACKEND",), DEFAULT_BACKEND),
        "backend", (:auto, :cuda, :cpu))

    reps = parse(Int, _value(options, "reps", ("NEXTLA_GEMM_REPS",), DEFAULT_REPS))
    warmup = parse(Int, _value(options, "warmup", ("NEXTLA_GEMM_WARMUP",), DEFAULT_WARMUP))
    seed = parse(Int, _value(options, "seed", ("NEXTLA_GEMM_SEED",), DEFAULT_SEED))
    shard_count = parse(Int, _value(options, "shard_count", ("NEXTLA_GEMM_SHARD_COUNT",), DEFAULT_SHARD_COUNT))
    shard_index = parse(Int, _value(options, "shard_index", ("NEXTLA_GEMM_SHARD_INDEX",), DEFAULT_SHARD_INDEX))
    case_regex = Regex(_value(options, "case_regex", ("NEXTLA_GEMM_CASE_REGEX",), DEFAULT_CASE_REGEX))
    precisions = _precisions(DEFAULT_PRECISIONS)
    output_dir = abspath(expanduser(_value(
        options, "output_dir", ("NEXTLA_GEMM_OUTPUT_DIR",), DEFAULT_OUTPUT_DIR)))

    workspace_fractions = _list(Float64, _value(
        options, "workspace_fractions", ("NEXTLA_GEMM_WORKSPACE_FRACTIONS",),
        DEFAULT_WORKSPACE_FRACTIONS), "workspace fractions")
    workspace_memory_safety = parse(Float64, _value(
        options, "memory_safety", ("NEXTLA_GEMM_MEMORY_SAFETY",), DEFAULT_MEMORY_SAFETY))
    tile_size = parse(Int, DEFAULT_TILE_SIZE)

    reps >= 1 || error("reps must be positive")
    warmup >= 0 || error("warmup must be nonnegative")
    shard_count >= 1 || error("shard-count must be positive")
    1 <= shard_index <= shard_count || error("shard-index must lie in 1:shard-count")
    all(0 < f <= 1 for f in workspace_fractions) ||
        error("workspace fractions must lie in (0, 1]")
    0 < workspace_memory_safety <= 1 ||
        error("memory-safety must lie in (0, 1]")
    isempty(GEMM_CASES) && error("GEMM_CASES must not be empty")
    for case in GEMM_CASES
        all(>(0), (case.m, case.k, case.n, case.bm, case.bk, case.bn,
                   case.maxrank_A, case.maxrank_B)) ||
            error("all case dimensions, tile sizes, and ranks must be positive")
        case.m % case.bm == 0 && case.k % case.bk == 0 && case.n % case.bn == 0 ||
            error("each case must be divisible by its bm, bk, and bn")
        max(case.maxrank_A, case.maxrank_B) <= min(case.bm, case.bk, case.bn) ||
            error("case maximum ranks must not exceed its tile sizes")
    end

    mkpath(output_dir)
    return GemmBenchmarkConfig(
        benchmark, backend, reps, warmup, seed, output_dir, shard_count,
        shard_index, case_regex, precisions, workspace_fractions,
        workspace_memory_safety, collect(GEMM_CASES), tile_size)
end

load_config(; default_benchmark::Symbol=:dense) =
    load_config(ARGS; default_benchmark=default_benchmark)

function benchmark_from_args(args=ARGS)
    options = _options(args)
    return _symbol(
        get(options, "benchmark", _env(("NEXTLA_GEMM_BENCHMARK",), "dense")),
        "benchmark", (:dense, :workspace, Symbol("tlr-output")))
end

function output_path(config::GemmBenchmarkConfig, benchmark::Symbol)
    filename = benchmark === :dense ? "tlr_gemm_benchmark.csv" :
               benchmark === :workspace ? "tlr_gemm_workspace_benchmark.csv" :
               benchmark === Symbol("tlr-output") ? "tlr_gemm_output_benchmark.csv" :
               error("unknown benchmark `$benchmark`")
    return joinpath(config.output_dir, filename)
end

end
