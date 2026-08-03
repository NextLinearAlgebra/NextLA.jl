# Poster-oriented CompressedFTLR × CompressedFTLR → dense benchmark.
#
# Symbolic analysis and numerical execution are timed separately. Matrix/factor
# generation and numerical workspace allocation are setup and excluded
# consistently from all methods.

using LinearAlgebra
using Printf
using Statistics
using KernelAbstractions
using CUDA
using NextLA

include(joinpath(@__DIR__, "compressed_dense_support.jl"))
using .DenseGemmCommon
const TLRM = NextLA.TLRmodule

parse_ints(name, default) = parse.(Int, split(get(ENV, name, default), ','))
const SIZES = parse_ints("NEXTLA_DENSE_SIZES", "4096,8192,16384,32768")
const TILE_DIVISORS = parse_ints("NEXTLA_DENSE_TILE_DIVISORS", "16,8,4")
const WARMUP = parse(Int, get(ENV, "NEXTLA_DENSE_WARMUP", "1"))
const REPS = parse(Int, get(ENV, "NEXTLA_DENSE_REPS", "3"))
const ANALYSIS_REPS = parse(Int, get(ENV, "NEXTLA_DENSE_ANALYSIS_REPS", "3"))
const RUNS = parse(Int, get(ENV, "NEXTLA_DENSE_RUNS", "1"))
const OUTPUT = get(ENV, "NEXTLA_DENSE_OUTPUT",
    joinpath(@__DIR__, "results", "compressed_dense_v2.csv"))
const CASE_FILTER = Regex(get(ENV, "NEXTLA_DENSE_FILTER", ".*"))

const PRECISIONS = (
    (name="fp16_fp32", T=Float16, compute=NextLA.GEMMCompute{Float32}()),
    (name="fp32_tf32", T=Float32, compute=NextLA.TF32()),
    (name="fp32", T=Float32, compute=NextLA.GEMMCompute{Float32}()),
)

function rank_profiles(b)
    r32 = max(1, b ÷ 32); r16 = max(1, b ÷ 16); r8 = max(1, b ÷ 8)
    return (
        (name="constant_b32", distribution=:constant, ranks=(r32, r32), lo=r32, hi=r32),
        (name="constant_b16", distribution=:constant, ranks=(r16, r16), lo=r16, hi=r16),
        (name="constant_b8", distribution=:constant, ranks=(r8, r8), lo=r8, hi=r8),
        (name="uniform_b32_b8", distribution=:uniform, ranks=(r8, r8), lo=r32, hi=r8),
        (name="skewed_b32_b8", distribution=:skewed, ranks=(r8, r8), lo=r32, hi=r8),
    )
end

function samples_ms(f, C, T; warmup=WARMUP, reps=REPS)
    for _ in 1:warmup
        fill!(C, zero(T)); f(); CUDA.synchronize()
    end
    values = Vector{Float64}(undef, reps)
    for repetition in 1:reps
        fill!(C, zero(T)); CUDA.synchronize()
        start = time_ns(); f(); CUDA.synchronize()
        values[repetition] = (time_ns() - start) / 1.0e6
    end
    return (median=median(values), minimum=minimum(values), values)
end

function dense_timing(N, T, compute)
    A = CUDA.randn(T, N, N); B = CUDA.randn(T, N, N); C = CUDA.zeros(T, N, N)
    timing = samples_ms(C, T) do
        NextLA.precision_gemm!('N', 'N', one(T), A, B, one(T), C, compute)
    end
    A = B = C = nothing
    GC.gc(true); CUDA.reclaim()
    return timing
end

function time_analysis(C, A, B, workspace, compute)
    # Compile and initialize CUDA library paths before measuring symbolic work.
    warm = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
    CUDA.synchronize(); close(warm)
    values = Vector{Float64}(undef, ANALYSIS_REPS)
    retained = nothing
    for repetition in 1:ANALYSIS_REPS
        retained === nothing || close(retained)
        CUDA.synchronize(); start = time_ns()
        retained = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
        CUDA.synchronize(); values[repetition] = (time_ns() - start) / 1.0e6
    end
    return retained, (median=median(values), minimum=minimum(values), values)
end

function crossover_calls(analysis_ms, analyzed_ms, alternative_ms)
    saving = alternative_ms - analyzed_ms
    saving <= 0 && return Inf
    return ceil(Int, analysis_ms / saving)
end

const COLUMNS = (
    "case_id", "N", "tile_size", "profile", "distribution", "min_rank", "max_rank",
    "precision", "runs", "workspace_bytes", "analysis_ms",
    "analysis_min_ms", "transient_median_ms", "transient_min_ms",
    "analyzed_median_ms", "analyzed_min_ms",
    "dense_median_ms", "dense_min_ms", "analyzed_speedup",
    "cold_analysis_plus_numeric_ms", "analysis_amortization_calls",
    "exact_flops", "executed_flops", "padding_waste_pct",
    "analyzed_executed_gflops", "dense_gflops",
)

case_id(N, b, profile, precision) =
    "N$(N)__b$(b)__$(profile.name)__$(precision.name)__runs$(RUNS)"

completed(path) = completed(path, COLUMNS)

function completed(path, columns)
    isfile(path) || return Set{String}()
    lines = readlines(path)
    isempty(lines) && return Set{String}()
    first(lines) == join(columns, ',') || error("CSV schema mismatch at $path")
    return Set(first(split(line, ',')) for line in Iterators.drop(lines, 1) if !isempty(line))
end

ensure_output(path) = ensure_output(path, COLUMNS)

function ensure_output(path, columns)
    mkpath(dirname(path))
    if !isfile(path) || filesize(path) == 0
        open(path, "w") do io
            println(io, join(columns, ','))
        end
    end
end

function append_row(path, row)
    open(path, "a") do io
        println(io, join(row, ',')); flush(io)
    end
end

function benchmark_case(N, b, profile, precision, dense)
    T = precision.T; compute = precision.compute
    A, B = DenseGemmCommon.generate_ftlr_operands(
        N, N, N, b, profile.ranks, T;
        seed=20260729, backend=CUDA.CUDABackend(), format=:compressed,
        rank_distribution=profile.distribution,
        min_rank=profile.lo, max_rank=profile.hi)
    C = CUDA.zeros(T, N, N)
    workspace_bytes = NextLA.gemm_workspace_bytes(A, B; runs=RUNS)
    workspace = NextLA.DenseGemmWorkspace(A, B; bytes=workspace_bytes)

    transient = samples_ms(C, T) do
        TLRM.gemm!(C, A, B; workspace, alpha=one(T), beta=one(T), compute)
    end

    analysis, analysis_timing = time_analysis(C, A, B, workspace, compute)
    analyzed = samples_ms(C, T) do
        TLRM.gemm!(C, A, B; workspace, alpha=one(T), beta=one(T), compute, analysis)
    end

    executed = DenseGemmCommon._tlr_tlr_executed_flops(A, B, workspace_bytes)
    exact = DenseGemmCommon._tlr_tlr_exact_flops(A, B, workspace_bytes)
    padding_waste = executed == 0 ? 0.0 : 100 * (executed - exact) / executed
    dense_flops = 2.0 * N^3
    cold = analysis_timing.median + analyzed.median
    crossover = crossover_calls(analysis_timing.median, analyzed.median, transient.median)
    result = (
        case_id(N, b, profile, precision), N, b, profile.name, profile.distribution,
        profile.lo, profile.hi, precision.name, RUNS, workspace_bytes,
        analysis_timing.median, analysis_timing.minimum,
        transient.median, transient.minimum, analyzed.median, analyzed.minimum,
        dense.median, dense.minimum, dense.median / analyzed.median,
        cold, crossover, exact, executed, padding_waste,
        executed / (analyzed.median * 1e6), dense_flops / (dense.median * 1e6),
    )
    @printf(
        "%-55s analysis=%8.3f ms transient=%8.3f ms numeric=%8.3f ms dense=%8.3f ms speedup=%5.2fx padding=%5.1f%%\n",
        first(result), analysis_timing.median, transient.median, analyzed.median,
        dense.median, dense.median / analyzed.median, padding_waste)
    close(analysis)
    A = B = C = workspace = nothing
    GC.gc(true); CUDA.reclaim()
    return result
end

function run_compressed_dense()
    CUDA.functional() || error("compressed dense-output benchmark requires CUDA")
    ensure_output(OUTPUT); done = completed(OUTPUT)
    @printf("Compressed dense-output benchmark: H/W=%d/%d runs=%d output=%s\n",
            WARMUP, REPS, RUNS, OUTPUT)
    dense_cache = Dict{Tuple{Int,String},NamedTuple}()
    for N in SIZES, divisor in TILE_DIVISORS
        divisor > 0 || throw(ArgumentError("tile divisors must be positive"))
        N % divisor == 0 ||
            throw(ArgumentError("matrix size $N is not divisible by tile divisor $divisor"))
        b = N ÷ divisor
        for precision in PRECISIONS, profile in rank_profiles(b)
            id = case_id(N, b, profile, precision)
            occursin(CASE_FILTER, id) || continue
            id in done && (@printf("skip %s\n", id); continue)
            dense = get!(dense_cache, (N, precision.name)) do
                dense_timing(N, precision.T, precision.compute)
            end
            row = benchmark_case(N, b, profile, precision, dense)
            append_row(OUTPUT, row); push!(done, id)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_compressed_dense()
end
