# Workspace-scaling study for the largest dense-output TLR GEMMs.

if !isdefined(@__MODULE__, :GemmBenchmarksConfig)
    include(joinpath(@__DIR__, "config.jl"))
end
using .GemmBenchmarksConfig

const CONFIG = load_config(; default_benchmark=:workspace)
const WORKSPACE_OUTPUT = output_path(CONFIG, :workspace)
const WORKSPACE_FRACTIONS = Tuple(CONFIG.workspace_fractions)
const MEMORY_SAFETY_FRACTION = CONFIG.workspace_memory_safety
const LARGE_SHAPES = Set((
    "square_4096",
    "square_8192",
    "rect_2048x4096",
    "rect_4096x8192",
))

all(0 < f <= 1 for f in WORKSPACE_FRACTIONS) ||
    error("workspace fractions must lie in (0,1]")
0 < MEMORY_SAFETY_FRACTION <= 1 ||
    error("memory-safety must lie in (0,1]")

const WORKSPACE_CSV_COLUMNS = (
    "workspace_case_id", "case_id", "shape", "m", "k", "n",
    "tile_ratio", "bm", "bk", "bn", "rank_ratio_A", "rank_ratio_B",
    "maxrank_A", "maxrank_B", "axis", "precision", "nreps",
    "requested_workspace_fraction", "actual_workspace_fraction",
    "minimum_workspace_bytes", "maximum_workspace_bytes", "workspace_bytes",
    "status", "reason", "available_gpu_bytes_before_workspace",
    "dense_ms", "tlr_ms", "speedup", "tlr_work_ratio_pct",
    "tlr_arithmetic_reduction", "tlr_rate_ratio_pct", "wasted_flops_pct",
    "dense_gflops", "tlr_executed_gflops", "tlr_dense_equiv_gflops",
)

@inline workspace_case_id(case, fraction) =
    "$(case_id(case))__ws$(replace(@sprintf("%.6f", fraction), "." => "p"))"

function workspace_completed_cases(path)
    isfile(path) || return Set{String}()
    ids = Set{String}()
    open(path, "r") do io
        eof(io) && return
        readline(io)
        for line in eachline(io)
            isempty(line) || push!(ids, first(split(line, ',')))
        end
    end
    return ids
end

function write_workspace_header_if_needed(path)
    expected = join(WORKSPACE_CSV_COLUMNS, ',')
    if isfile(path) && filesize(path) > 0
        actual = open(readline, path)
        actual == expected || error(
            "workspace benchmark CSV schema mismatch in $path; choose a new " *
            "output directory or remove the obsolete file")
    else
        open(path, "w") do io
            println(io, expected)
        end
    end
end

@inline function available_workspace_memory(backend)
    backend isa KernelAbstractions.CPU && return typemax(Int)
    return Int(CUDA.available_memory())
end

function append_workspace_row(path, values)
    open(path, "a") do io
        println(io, join(values, ','))
        flush(io)
    end
end

function is_memory_error(err)
    message = lowercase(sprint(showerror, err))
    return err isa OutOfMemoryError ||
           occursin("outofgpumemory", lowercase(string(typeof(err)))) ||
           occursin("out of memory", message)
end

function record_case_memory_skip!(case, done, reason)
    available = Int(CUDA.available_memory())
    for fraction in WORKSPACE_FRACTIONS
        workspace_case_id(case, fraction) in done && continue
        row = skipped_workspace_row(
            case, fraction, 0, 0, 0, "skipped", reason, available)
        append_workspace_row(WORKSPACE_OUTPUT, row)
        push!(done, workspace_case_id(case, fraction))
    end
    release_backend_memory!()
    return nothing
end

function skipped_workspace_row(case, fraction, minimum_bytes, maximum_bytes,
                               bytes, status, reason, available)
    actual_fraction = maximum_bytes == 0 ? 0.0 : bytes / maximum_bytes
    return (
        workspace_case_id(case, fraction), case_id(case), case.shape,
        case.m, case.k, case.n, "1/$(case.tile_den)",
        case.bm, case.bk, case.bn,
        "1/$(case.rank_den_A)", "1/$(case.rank_den_B)",
        case.maxrank_A, case.maxrank_B, case.axis, case.precision, NREPS,
        fraction, actual_fraction, minimum_bytes, maximum_bytes, bytes,
        status, reason, available,
        "", "", "", "", "", "", "", "", "", "",
    )
end

function measured_workspace_row(case, fraction, minimum_bytes, maximum_bytes,
                                bytes, available, dense, ms, model)
    nominal = dense_flops(case.m, case.k, case.n)
    executed_rate = model.padded / (ms * 1e6)
    equivalent_rate = nominal / (ms * 1e6)
    speedup = dense.ms / ms
    work_ratio = model.padded / nominal
    arithmetic_reduction = nominal / model.padded
    rate_ratio = executed_rate / dense.gflops
    return (
        workspace_case_id(case, fraction), case_id(case), case.shape,
        case.m, case.k, case.n, "1/$(case.tile_den)",
        case.bm, case.bk, case.bn,
        "1/$(case.rank_den_A)", "1/$(case.rank_den_B)",
        case.maxrank_A, case.maxrank_B, case.axis, case.precision, NREPS,
        fraction, bytes / maximum_bytes, minimum_bytes, maximum_bytes, bytes,
        "ok", "", available,
        dense.ms, ms, speedup, 100 * work_ratio, arithmetic_reduction,
        100 * rate_ratio, model.waste, dense.gflops, executed_rate,
        equivalent_rate,
    )
end

function benchmark_workspace_case!(backend, case, dense, done)
    T = case.T
    A, ranksA = make_tlr(
        backend, T, case.m, case.k, case.bm, case.bk,
        case.maxrank_A, case.orderA; seed=SEED + 11)
    B, ranksB = make_tlr(
        backend, T, case.k, case.n, case.bk, case.bn,
        case.maxrank_B, case.orderB; seed=SEED + 29)
    C = backend_zeros(backend, T, case.m, case.n)
    minimum_bytes = TLRM.gemm_minimum_workspace_bytes(A, B)
    maximum_bytes = TLRM.gemm_maximum_workspace_bytes(A, B)
    model = tlr_flop_model(case, ranksA, ranksB)

    for fraction in WORKSPACE_FRACTIONS
        wid = workspace_case_id(case, fraction)
        wid in done && continue
        bytes = fld(floor(Int, fraction * maximum_bytes), sizeof(T)) * sizeof(T)
        available = available_workspace_memory(backend)
        if bytes < minimum_bytes
            row = skipped_workspace_row(
                case, fraction, minimum_bytes, maximum_bytes, bytes,
                "skipped", "below_minimum_workspace", available)
            append_workspace_row(WORKSPACE_OUTPUT, row)
            push!(done, wid)
            continue
        end
        safe_available = floor(Int, MEMORY_SAFETY_FRACTION * available)
        if bytes > safe_available
            row = skipped_workspace_row(
                case, fraction, minimum_bytes, maximum_bytes, bytes,
                "skipped", "insufficient_gpu_memory", available)
            append_workspace_row(WORKSPACE_OUTPUT, row)
            push!(done, wid)
            continue
        end

        workspace = try
            NextLA.DenseGemmWorkspace(A, B; bytes)
        catch err
            is_memory_error(err) || rethrow()
            row = skipped_workspace_row(
                case, fraction, minimum_bytes, maximum_bytes, bytes,
                "skipped", "workspace_allocation_failed",
                available)
            append_workspace_row(WORKSPACE_OUTPUT, row)
            push!(done, wid)
            release_backend_memory!()
            continue
        end
        f = () -> TLRM.gemm!(
            C, A, B; alpha=one(T), beta=zero(T),
            compute=case.compute, workspace)
        ms = best_time_ms(f, backend)
        row = measured_workspace_row(
            case, fraction, minimum_bytes, maximum_bytes,
            sizeof(workspace), available, dense, ms, model)
        append_workspace_row(WORKSPACE_OUTPUT, row)
        push!(done, wid)
        @printf(
            "%-58s ws=%6.1f%%  TLR=%9.3f ms  speedup=%6.2fx  rate=%5.1f%%  waste=%5.1f%%\n",
            case_id(case), 100 * sizeof(workspace) / maximum_bytes,
            ms, dense.ms / ms,
            100 * (model.padded / (ms * 1e6)) / dense.gflops,
            model.waste,
        )
        workspace = nothing
        release_backend_memory!()
    end
    A = B = C = nothing
    release_backend_memory!()
    return nothing
end

function workspace_main()
    HAS_CUDA || error("the large workspace study requires a functional CUDA backend")
    CONFIG.backend in (:auto, :cuda) ||
        error("the workspace study requires --backend auto or --backend cuda")
    backend = CUDA.CUDABackend()
    write_workspace_header_if_needed(WORKSPACE_OUTPUT)
    done = workspace_completed_cases(WORKSPACE_OUTPUT)
    candidates = [
        case for (index, case) in enumerate(all_cases())
        if case.shape in LARGE_SHAPES &&
           mod1(index, SHARD_COUNT) == SHARD_INDEX &&
           occursin(CASE_REGEX, case_id(case))
    ]
    @printf(
        "NextLA dense-output TLR GEMM workspace study\ncases=%d fractions=%s reps=%d warmup=%d shard=%d/%d output=%s\n",
        length(candidates), join(WORKSPACE_FRACTIONS, '/'), NREPS, WARMUP,
        SHARD_INDEX, SHARD_COUNT, WORKSPACE_OUTPUT,
    )

    dense_cache = Dict{Tuple{Int,Int,Int,String},NamedTuple}()
    for case in candidates
        all(workspace_case_id(case, f) in done for f in WORKSPACE_FRACTIONS) &&
            continue
        dense_key = (case.m, case.k, case.n, case.precision)
        dense = try
            get!(dense_cache, dense_key) do
                benchmark_dense(backend, case)
            end
        catch err
            is_memory_error(err) || rethrow()
            record_case_memory_skip!(case, done, "dense_problem_does_not_fit")
            continue
        end
        try
            benchmark_workspace_case!(backend, case, dense, done)
        catch err
            is_memory_error(err) || rethrow()
            record_case_memory_skip!(case, done, "tlr_problem_does_not_fit")
        end
    end
    return nothing
end

const GEMM_BENCHMARK_LIBRARY_ONLY = true
include(joinpath(@__DIR__, "benchmark_gemm.jl"))
workspace_main()
