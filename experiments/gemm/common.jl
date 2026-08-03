"""Shared machinery for the publication-oriented dense-output GEMM sweeps."""
module GemmExperimentCommon

using Dates
using Printf
using Random
using Statistics
using CUDA
using NextLA

include(joinpath(@__DIR__, "..", "compressed_dense_support.jl"))
using .DenseGemmCommon

const TLRM = NextLA.TLRmodule

export RESULT_COLUMNS, PRECISION_TABLE,
       parse_int_list, parse_string_list, parse_symbol_list, parse_bool,
       selected_precisions, rank_interval, rank_grid, rank_band,
       fresh_csv, write_csv_row, close_csv,
       baseline_case_id, compressed_case_id,
       benchmark_dense, benchmark_compressed_case,
       benchmark_compressed_workspace_sweep, baseline_row, compressed_row,
       print_case, gpu_name, validate_cuda_precisions

const PRECISION_TABLE = Dict(
    "bf16" => (
        name="bf16", T=Core.BFloat16,
        compute=NextLA.GEMMCompute{Float32}(), compute_name="fp32_accumulate"),
    "fp16" => (
        name="fp16", T=Float16,
        compute=NextLA.GEMMCompute{Float32}(), compute_name="fp32_accumulate"),
    "fp32" => (
        name="fp32", T=Float32,
        compute=NextLA.GEMMCompute{Float32}(), compute_name="fp32"),
    "tf32" => (
        name="tf32", T=Float32,
        compute=NextLA.TF32(), compute_name="tf32"),
)

const RESULT_COLUMNS = (
    :run_id, :timestamp_utc, :experiment, :record_kind, :case_id,
    :baseline_case_id, :operand_layout, :N, :tile_divisor, :tile_size,
    :tile_grid, :distribution, :rank_band, :min_rank, :max_rank,
    :mean_rank_A, :mean_rank_B, :mean_execution_rank_A,
    :mean_execution_rank_B, :exact_rank_count_A, :exact_rank_count_B,
    :execution_rank_count_A, :execution_rank_count_B,
    :execution_rank_policy, :precision, :storage_type, :compute_mode,
    :seed, :factor_fill, :warmup, :repetitions, :analysis_repetitions,
    :workspace_policy, :workspace_parameter, :A_storage_kind,
    :B_storage_kind, :A_storage_bytes, :B_storage_bytes,
    :operand_storage_bytes, :workspace_bytes, :dense_reference_bytes,
    :memory_ratio, :analysis_median_ms, :analysis_min_ms,
    :numeric_median_ms, :numeric_min_ms, :dense_median_ms, :dense_min_ms,
    :speedup_median, :speedup_best, :exact_flops, :executed_flops,
    :padding_waste_pct, :executed_gflops_median, :dense_gflops_median,
    :has_fallback, :numeric_samples_ms, :dense_samples_ms, :gpu_name,
    :julia_version,
)

parse_int_list(name, default) = parse.(Int, split(get(ENV, name, default), ','))
parse_string_list(name, default) = strip.(split(get(ENV, name, default), ','))
parse_symbol_list(name, default) = Symbol.(parse_string_list(name, default))

function parse_bool(name, default::Bool)
    value = lowercase(strip(get(ENV, name, string(default))))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    throw(ArgumentError("$name must be true/false, yes/no, on/off, or 1/0"))
end

function selected_precisions(name, default)
    names = parse_string_list(name, default)
    unknown = setdiff(names, collect(keys(PRECISION_TABLE)))
    isempty(unknown) || throw(ArgumentError(
        "unknown precision(s) $(join(unknown, ',')); use bf16,fp16,fp32,tf32"))
    return [PRECISION_TABLE[value] for value in names]
end

function rank_interval(tile_size::Int, min_divisor::Int, max_divisor::Int)
    min_divisor > 0 && max_divisor > 0 ||
        throw(ArgumentError("rank divisors must be positive"))
    lo = max(1, tile_size ÷ min_divisor)
    hi = max(lo, tile_size ÷ max_divisor)
    hi <= tile_size || throw(ArgumentError("rank interval exceeds tile size"))
    return lo, hi
end

function rank_band(spec::AbstractString, tile_size::Int)
    pieces = split(strip(spec), ':')
    length(pieces) == 2 || throw(ArgumentError(
        "rank band '$spec' must be MIN_DIVISOR:MAX_DIVISOR, e.g. 16:8"))
    min_divisor, max_divisor = parse.(Int, pieces)
    lo, hi = rank_interval(tile_size, min_divisor, max_divisor)
    return (name="b$(min_divisor)_b$(max_divisor)", lo, hi)
end

"""Create the exact logical rank grid used by a synthetic operand."""
function rank_grid(qm::Int, qn::Int, lo::Int, hi::Int,
                   distribution::Symbol, seed::Int)
    1 <= lo <= hi || throw(ArgumentError("rank interval must be positive and ordered"))
    distribution in (:uniform, :skewed, :constant) || throw(ArgumentError(
        "rank distribution must be uniform, skewed, or constant"))
    if distribution === :constant
        lo == hi || throw(ArgumentError(
            "constant rank distribution requires equal minimum and maximum ranks"))
        return fill(lo, qm, qn)
    end
    rng = MersenneTwister(seed)
    samples = rand(rng, qm, qn)
    mapped = distribution === :uniform ? samples : samples .^ 2
    return min.(lo .+ floor.(Int, mapped .* (hi - lo + 1)), hi)
end

mutable struct CsvRun
    path::String
    io::IO
    run_id::String
    timestamp::String
    experiment::String
end

function _timestamp()
    utc = Dates.unix2datetime(time())
    return Dates.format(utc, dateformat"yyyymmddTHHMMSS.sss") * "Z"
end

function _default_output(stem)
    stamp = replace(_timestamp(), ':' => '-')
    return normpath(joinpath(@__DIR__, "..", "results", "gemm",
                             "$(stem)__$(stamp)__pid$(getpid()).csv"))
end

"""
Open a brand-new CSV. An explicitly requested path is rejected if it exists;
the timestamped default receives a numeric suffix in the unlikely event of a
collision. Existing experiment results are therefore never appended or replaced.
"""
function fresh_csv(experiment::AbstractString, output_env::AbstractString)
    requested = strip(get(ENV, output_env, ""))
    path = isempty(requested) ? _default_output(experiment) : abspath(requested)
    if isempty(requested)
        root, ext = splitext(path)
        suffix = 1
        while ispath(path)
            path = "$(root)__$(suffix)$(ext)"
            suffix += 1
        end
    elseif ispath(path)
        throw(ArgumentError("refusing to overwrite existing output: $path"))
    end
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, join(string.(RESULT_COLUMNS), ','))
    flush(io)
    stamp = _timestamp()
    return CsvRun(path, io, "$(experiment)__$(stamp)__pid$(getpid())", stamp,
                  String(experiment))
end

function _csv_value(value)
    value === nothing && return ""
    value === missing && return ""
    text = string(value)
    if occursin(r"[\",\r\n]", text)
        return "\"" * replace(text, '"' => "\"\"") * "\""
    end
    return text
end

function write_csv_row(run::CsvRun, row::NamedTuple)
    values = map(RESULT_COLUMNS) do column
        hasproperty(row, column) ? getproperty(row, column) : missing
    end
    println(run.io, join(_csv_value.(values), ','))
    flush(run.io)
    return nothing
end

close_csv(run::CsvRun) = close(run.io)

gpu_name() = CUDA.functional() ? string(CUDA.name(CUDA.device())) : "unavailable"

function validate_cuda_precisions(precisions)
    CUDA.functional() || error("experiment requires a functional CUDA device")
    capability = CUDA.capability(CUDA.device())
    if capability < v"8.0" && any(
        precision -> precision.T === Core.BFloat16 || precision.name == "tf32",
        precisions)
        throw(ArgumentError(
            "BF16 and TF32 require an NVIDIA SM80+ GPU; $(gpu_name()) is SM$capability. " *
            "Restrict the precision list to fp16,fp32 on this device."))
    end
    return nothing
end

function _samples_ms(f, C, ::Type{T}; warmup::Int, repetitions::Int) where {T}
    warmup >= 0 || throw(ArgumentError("warmup count must be nonnegative"))
    repetitions > 0 || throw(ArgumentError("repetition count must be positive"))
    for _ in 1:warmup
        fill!(C, zero(T))
        f()
        CUDA.synchronize()
    end
    samples = Vector{Float64}(undef, repetitions)
    for repetition in eachindex(samples)
        fill!(C, zero(T))
        CUDA.synchronize()
        start = time_ns()
        f()
        CUDA.synchronize()
        samples[repetition] = (time_ns() - start) / 1.0e6
    end
    return (median=median(samples), minimum=minimum(samples), samples)
end

function _analysis_timing(C, A, B, workspace, compute; repetitions::Int)
    repetitions > 0 || throw(ArgumentError("analysis repetitions must be positive"))
    warm = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
    CUDA.synchronize()
    close(warm)
    samples = Vector{Float64}(undef, repetitions)
    retained = nothing
    for repetition in eachindex(samples)
        retained === nothing || close(retained)
        CUDA.synchronize()
        start = time_ns()
        retained = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
        CUDA.synchronize()
        samples[repetition] = (time_ns() - start) / 1.0e6
    end
    return retained, (median=median(samples), minimum=minimum(samples), samples)
end

function _fill_compressed!(A, seed::Int, fill_mode::Symbol, N::Int)
    if fill_mode === :zeros
        return A
    elseif fill_mode === :constant
        value = eltype(A)(inv(sqrt(Float64(N))))
        fill!(A.outer.data, value)
        fill!(A.inner.data, value)
    elseif fill_mode === :random
        CUDA.seed!(seed)
        CUDA.rand!(A.outer.data)
        CUDA.rand!(A.inner.data)
    else
        throw(ArgumentError("factor fill must be random, constant, or zeros"))
    end
    CUDA.synchronize()
    return A
end

function _compressed_operand(N, b, ranks, ::Type{T}, policy, seed, fill_mode) where {T}
    A = TLRM.CompressedFTLRMatrix(
        CUDA.CUDABackend(), T, N, N, b, ranks;
        outer_order=TLRM.TileRowMajor, inner_order=TLRM.TileColMajor,
        execution_rank_policy=policy)
    return _fill_compressed!(A, seed, fill_mode, N)
end

function _dense_operand(N, ::Type{T}, seed, fill_mode) where {T}
    if fill_mode === :zeros
        return CUDA.zeros(T, N, N)
    elseif fill_mode === :constant
        return CUDA.fill(T(inv(sqrt(Float64(N)))), N, N)
    elseif fill_mode === :random
        CUDA.seed!(seed)
        return CUDA.rand(T, N, N)
    end
    throw(ArgumentError("factor fill must be random, constant, or zeros"))
end

@inline _compressed_storage_bytes(A) =
    (length(A.outer.data) + length(A.inner.data)) * sizeof(eltype(A))
@inline _dense_storage_bytes(N, ::Type{T}) where {T} = Int128(N) * N * sizeof(T)

function _rank_stats(A)
    exact = Int.(TLRM.ranks(A))
    executed = Int.(TLRM.execution_ranks(A))
    return (
        mean_exact=mean(exact), mean_executed=mean(executed),
        exact_count=length(unique(exact)), execution_count=length(unique(executed)),
    )
end

function _mixed_flops(A, N::Int; execution::Bool)
    qm, qk = TLRM.grid_size(A)
    total = 0.0
    for i in 1:qm, k in 1:qk
        tm, tn = TLRM.tile_size(A, i, k)
        rank = execution ? TLRM._compressed_ftlr_execution_rank(A, i, k) :
                           TLRM._compressed_ftlr_rank(A, i, k)
        total += 2.0 * rank * N * (tm + tn)
    end
    return total
end

function _workspace(A, B, layout::Symbol, tile_size::Int, runs::Int,
                    mixed_stripes::Int)
    if layout === :compressed_compressed
        # NOTE: the sweep parameter is RUN COUNT, not rows per run, and the two
        # run in opposite directions -- `runs=1` is the LARGEST workspace (one
        # fused unit) whereas the old `runs=1` was the smallest. Run
        # count is the meaningful knob: it sets the number of grouped-GEMM
        # submissions, and work units stop being whole output rows once the
        # budget falls below a full-width row.
        bytes = NextLA.gemm_workspace_bytes(A, B; runs)
        workspace = NextLA.DenseGemmWorkspace(A, B; bytes)
        return workspace, sizeof(workspace), "tlr_tlr_runs", runs
    end
    compressed = layout === :compressed_dense ? A : B
    stripe_extent = min(size(compressed, 1), mixed_stripes * tile_size)
    total_execution_rank = sum(Int, TLRM.execution_ranks(compressed))
    bytes = total_execution_rank * stripe_extent * sizeof(eltype(compressed))
    workspace = NextLA.DenseGemmWorkspace(compressed, bytes)
    return workspace, sizeof(workspace), "one_or_more_tile_stripes", mixed_stripes
end

function _cleanup_gpu!()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

baseline_case_id(N, precision) = "dense__N$(N)__$(precision.name)"

function compressed_case_id(N, divisor, distribution, lo, hi, precision,
                            layout, policy, workspace_parameter)
    return "$(layout)__N$(N)__q$(divisor)__$(distribution)__r$(lo)-$(hi)" *
           "__$(precision.name)__$(policy)__w$(workspace_parameter)"
end

"""Benchmark one dense baseline and release its three dense matrices."""
function benchmark_dense(N::Int, precision; warmup::Int, repetitions::Int,
                         seed::Int, fill_mode::Symbol)
    T, compute = precision.T, precision.compute
    A = _dense_operand(N, T, seed, fill_mode)
    B = _dense_operand(N, T, seed + 1, fill_mode)
    C = CUDA.zeros(T, N, N)
    timing = _samples_ms(C, T; warmup, repetitions) do
        NextLA.precision_gemm!(
            'N', 'N', one(T), A, B, zero(T), C, compute)
    end
    A = B = C = nothing
    _cleanup_gpu!()
    return timing
end

function _benchmark_prepared_compressed_case(C, A, B, N::Int, b::Int,
                                             precision, layout::Symbol;
                                             warmup::Int, repetitions::Int,
                                             analysis_repetitions::Int,
                                             runs::Int,
                                             mixed_stripes::Int)
    T, compute = precision.T, precision.compute
    workspace, workspace_bytes, workspace_policy, workspace_parameter =
        _workspace(A, B, layout, b, runs, mixed_stripes)

    analysis = nothing
    try
        analysis, analysis_timing = _analysis_timing(
            C, A, B, workspace, compute; repetitions=analysis_repetitions)
        numeric = _samples_ms(C, T; warmup, repetitions) do
            TLRM.gemm!(C, A, B; workspace, alpha=one(T), beta=zero(T),
                       compute, analysis)
        end

        exact_flops, executed_flops = if layout === :compressed_compressed
            (DenseGemmCommon._tlr_tlr_exact_flops(A, B, workspace_bytes),
             DenseGemmCommon._tlr_tlr_executed_flops(A, B, workspace_bytes))
        else
            compressed = layout === :compressed_dense ? A : B
            (_mixed_flops(compressed, N; execution=false),
             _mixed_flops(compressed, N; execution=true))
        end
        astats = layout === :dense_compressed ? nothing : _rank_stats(A)
        bstats = layout === :compressed_dense ? nothing : _rank_stats(B)
        dense_bytes = Int(_dense_storage_bytes(N, T))
        Abytes = layout === :dense_compressed ? dense_bytes : _compressed_storage_bytes(A)
        Bbytes = layout === :compressed_dense ? dense_bytes : _compressed_storage_bytes(B)
        padding = executed_flops == 0 ? 0.0 :
            100.0 * (executed_flops - exact_flops) / executed_flops
        result = (
            analysis=analysis_timing, numeric, workspace_bytes, workspace_policy,
            workspace_parameter, Abytes, Bbytes, dense_reference_bytes=2 * dense_bytes,
            memory_ratio=(Abytes + Bbytes + workspace_bytes) / (2.0 * dense_bytes),
            astats, bstats, exact_flops, executed_flops, padding,
            has_fallback=hasproperty(analysis, :has_fallback) ? analysis.has_fallback : false,
        )
        return result
    finally
        analysis === nothing || close(analysis)
        workspace = nothing
    end
end

function _compressed_operands(N::Int, b::Int, distribution::Symbol,
                              lo::Int, hi::Int, precision, layout::Symbol,
                              seed::Int, fill_mode::Symbol,
                              execution_rank_policy::Symbol)
    layout in (:compressed_dense, :dense_compressed, :compressed_compressed) ||
        throw(ArgumentError("unsupported operand layout $layout"))
    T = precision.T
    q = N ÷ b
    ranksA = rank_grid(q, q, lo, hi, distribution, seed)
    # Use the same logical rank map for the two one-compressed-operand cases,
    # making their left/right comparison controlled. The two-compressed case
    # keeps independent A and B maps.
    ranksB = layout === :dense_compressed ? ranksA :
        rank_grid(q, q, lo, hi, distribution, seed + 1)
    A = layout === :dense_compressed ?
        _dense_operand(N, T, seed + 2, fill_mode) :
        _compressed_operand(N, b, ranksA, T, execution_rank_policy,
                            seed + 2, fill_mode)
    B = layout === :compressed_dense ?
        _dense_operand(N, T, seed + 3, fill_mode) :
        _compressed_operand(N, b, ranksB, T, execution_rank_policy,
                            seed + 3, fill_mode)
    C = CUDA.zeros(T, N, N)
    return C, A, B
end

"""Benchmark one of compressed×dense, dense×compressed, or compressed×compressed."""
function benchmark_compressed_case(N::Int, b::Int, distribution::Symbol,
                                   lo::Int, hi::Int, precision, layout::Symbol;
                                   warmup::Int, repetitions::Int,
                                   analysis_repetitions::Int, seed::Int,
                                   fill_mode::Symbol,
                                   execution_rank_policy::Symbol,
                                   runs::Int, mixed_stripes::Int)
    C, A, B = _compressed_operands(N, b, distribution, lo, hi, precision,
                                   layout, seed, fill_mode,
                                   execution_rank_policy)
    try
        return _benchmark_prepared_compressed_case(C, A, B, N, b, precision,
                                                   layout; warmup, repetitions,
                                                   analysis_repetitions,
                                                   runs,
                                                   mixed_stripes)
    finally
        A = B = C = nothing
        _cleanup_gpu!()
    end
end

"""
Benchmark several workspace parameters against the same operand allocations.
This keeps workspace tuning controlled and avoids regenerating multi-gigabyte
operands for every candidate. Results are returned in the order of `levels`.
"""
function benchmark_compressed_workspace_sweep(N::Int, b::Int,
                                              distribution::Symbol, lo::Int,
                                              hi::Int, precision,
                                              layout::Symbol, levels;
                                              warmup::Int, repetitions::Int,
                                              analysis_repetitions::Int,
                                              seed::Int, fill_mode::Symbol,
                                              execution_rank_policy::Symbol)
    candidates = unique(Int.(collect(levels)))
    isempty(candidates) && throw(ArgumentError("workspace sweep is empty"))
    all(>(0), candidates) || throw(ArgumentError(
        "workspace parameters must be positive"))
    C, A, B = _compressed_operands(N, b, distribution, lo, hi, precision,
                                   layout, seed, fill_mode,
                                   execution_rank_policy)
    results = NamedTuple[]
    try
        for level in candidates
            measured = _benchmark_prepared_compressed_case(C, A, B, N, b,
                                                           precision, layout;
                                                           warmup, repetitions,
                                                           analysis_repetitions,
                                                           runs=level,
                                                           mixed_stripes=level)
            push!(results, measured)
            # Analyses and workspaces can be very large. Reclaim only dead
            # allocations between candidates; A, B, and C remain live.
            _cleanup_gpu!()
        end
    finally
        A = B = C = nothing
        _cleanup_gpu!()
    end
    return results
end

_sample_string(values) = join((@sprintf("%.9g", value) for value in values), ';')

function baseline_row(run::CsvRun, N, precision, timing;
                      warmup, repetitions, seed, fill_mode)
    dense_bytes = Int(_dense_storage_bytes(N, precision.T))
    dense_flops = 2.0 * N^3
    return (
        run_id=run.run_id, timestamp_utc=run.timestamp, experiment=run.experiment,
        record_kind="baseline", case_id=baseline_case_id(N, precision),
        baseline_case_id=baseline_case_id(N, precision), operand_layout="dense_dense",
        N, precision=precision.name, storage_type=string(precision.T),
        compute_mode=precision.compute_name, seed, factor_fill=fill_mode,
        warmup, repetitions, analysis_repetitions=0,
        workspace_policy="none", workspace_parameter=0,
        A_storage_kind="dense", B_storage_kind="dense",
        A_storage_bytes=dense_bytes, B_storage_bytes=dense_bytes,
        operand_storage_bytes=2 * dense_bytes, workspace_bytes=0,
        dense_reference_bytes=2 * dense_bytes, memory_ratio=1.0,
        numeric_median_ms=timing.median, numeric_min_ms=timing.minimum,
        dense_median_ms=timing.median, dense_min_ms=timing.minimum,
        speedup_median=1.0, speedup_best=1.0, exact_flops=dense_flops,
        executed_flops=dense_flops, padding_waste_pct=0.0,
        executed_gflops_median=dense_flops / (timing.median * 1.0e6),
        dense_gflops_median=dense_flops / (timing.median * 1.0e6),
        has_fallback=false, numeric_samples_ms=_sample_string(timing.samples),
        dense_samples_ms=_sample_string(timing.samples), gpu_name=gpu_name(),
        julia_version=VERSION,
    )
end

function compressed_row(run::CsvRun, N, divisor, b, distribution, band_name,
                        lo, hi, precision, layout, policy, seed, fill_mode,
                        warmup, repetitions, analysis_repetitions, result, dense)
    id = compressed_case_id(
        N, divisor, distribution, lo, hi, precision, layout, policy,
        result.workspace_parameter)
    astats, bstats = result.astats, result.bstats
    return (
        run_id=run.run_id, timestamp_utc=run.timestamp, experiment=run.experiment,
        record_kind="compressed", case_id=id,
        baseline_case_id=baseline_case_id(N, precision), operand_layout=layout,
        N, tile_divisor=divisor, tile_size=b, tile_grid=divisor,
        distribution, rank_band=band_name, min_rank=lo, max_rank=hi,
        mean_rank_A=astats === nothing ? missing : astats.mean_exact,
        mean_rank_B=bstats === nothing ? missing : bstats.mean_exact,
        mean_execution_rank_A=astats === nothing ? missing : astats.mean_executed,
        mean_execution_rank_B=bstats === nothing ? missing : bstats.mean_executed,
        exact_rank_count_A=astats === nothing ? missing : astats.exact_count,
        exact_rank_count_B=bstats === nothing ? missing : bstats.exact_count,
        execution_rank_count_A=astats === nothing ? missing : astats.execution_count,
        execution_rank_count_B=bstats === nothing ? missing : bstats.execution_count,
        execution_rank_policy=policy, precision=precision.name,
        storage_type=string(precision.T), compute_mode=precision.compute_name,
        seed, factor_fill=fill_mode, warmup, repetitions, analysis_repetitions,
        workspace_policy=result.workspace_policy,
        workspace_parameter=result.workspace_parameter,
        A_storage_kind=layout === :dense_compressed ? "dense" : "compressed_tlr",
        B_storage_kind=layout === :compressed_dense ? "dense" : "compressed_tlr",
        A_storage_bytes=result.Abytes, B_storage_bytes=result.Bbytes,
        operand_storage_bytes=result.Abytes + result.Bbytes,
        workspace_bytes=result.workspace_bytes,
        dense_reference_bytes=result.dense_reference_bytes,
        memory_ratio=result.memory_ratio,
        analysis_median_ms=result.analysis.median,
        analysis_min_ms=result.analysis.minimum,
        numeric_median_ms=result.numeric.median,
        numeric_min_ms=result.numeric.minimum,
        dense_median_ms=dense.median, dense_min_ms=dense.minimum,
        speedup_median=dense.median / result.numeric.median,
        speedup_best=dense.minimum / result.numeric.minimum,
        exact_flops=result.exact_flops, executed_flops=result.executed_flops,
        padding_waste_pct=result.padding,
        executed_gflops_median=result.executed_flops / (result.numeric.median * 1.0e6),
        dense_gflops_median=2.0 * N^3 / (dense.median * 1.0e6),
        has_fallback=result.has_fallback,
        numeric_samples_ms=_sample_string(result.numeric.samples),
        gpu_name=gpu_name(), julia_version=VERSION,
    )
end

function print_case(row)
    if row.record_kind == "baseline"
        @printf("%-76s %9.3f ms  %8.2f TF/s\n", row.case_id,
                row.numeric_median_ms, row.dense_gflops_median / 1.0e3)
    else
        @printf("%-76s %9.3f ms  %6.2fx  memory=%6.3f  workspace=%8.2f MiB\n",
                row.case_id, row.numeric_median_ms, row.speedup_median,
                row.memory_ratio, row.workspace_bytes / 2.0^20)
    end
    return nothing
end

end
