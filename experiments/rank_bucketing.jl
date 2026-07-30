# Execution-rank bucketing ablation for CompressedFTLR × CompressedFTLR → dense.
#
# All policies use the same exact rank maps and factor values.  Only the stored
# execution capacity changes: exact, rounded-to-8, rounded-to-16, or power of 2.
# The companion descriptor CSV distinguishes cuBLAS grouped members from the
# ordinary-GEMM alignment fallbacks used by the current lowering.

using Printf
using Statistics
using CUDA
using NextLA

include(joinpath(@__DIR__, "compressed_dense_support.jl"))
using .DenseGemmCommon
const TLRM = NextLA.TLRmodule

parse_ints(name, default) = parse.(Int, split(get(ENV, name, default), ','))
parse_symbols(name, default) = Symbol.(split(get(ENV, name, default), ','))

const SIZES = parse_ints("NEXTLA_BUCKET_SIZES", "4096,8192")
const TILE_DIVISOR = parse(Int, get(ENV, "NEXTLA_BUCKET_TILE_DIVISOR", "16"))
const MIN_RANK = parse(Int, get(ENV, "NEXTLA_BUCKET_MIN_RANK", "1"))
const MAX_RANK = parse(Int, get(ENV, "NEXTLA_BUCKET_MAX_RANK", "64"))
const POLICIES = parse_symbols("NEXTLA_BUCKET_POLICIES", "exact,q8,q16,pow2")
const PRECISION_NAMES = split(get(ENV, "NEXTLA_BUCKET_PRECISIONS", "fp16_fp32"), ',')
const ROWS_PER_RUN = parse(Int, get(ENV, "NEXTLA_BUCKET_ROWS", "4"))
const WARMUP = parse(Int, get(ENV, "NEXTLA_BUCKET_WARMUP", "1"))
const REPS = parse(Int, get(ENV, "NEXTLA_BUCKET_REPS", "10"))
const ANALYSIS_REPS = parse(Int, get(ENV, "NEXTLA_BUCKET_ANALYSIS_REPS", "3"))
const SEED = parse(Int, get(ENV, "NEXTLA_BUCKET_SEED", "20260730"))
const OUTPUT = get(ENV, "NEXTLA_BUCKET_OUTPUT", joinpath(@__DIR__, "results", "rank_bucketing.csv"))
const GROUP_OUTPUT = get(ENV, "NEXTLA_BUCKET_GROUP_OUTPUT", joinpath(@__DIR__, "results", "rank_bucketing_groups.csv"))

const PRECISIONS = Dict(
    "fp16_fp32" => (name="fp16_fp32", T=Float16, compute=NextLA.GEMMCompute{Float32}()),
    "fp32_tf32" => (name="fp32_tf32", T=Float32, compute=NextLA.TF32()),
    "fp32" => (name="fp32", T=Float32, compute=NextLA.GEMMCompute{Float32}()),
)

const SUMMARY_COLUMNS = (
    "case_id", "N", "tile_size", "tile_grid", "min_rank", "max_rank", "policy", "precision",
    "rows_per_run", "factor_bytes", "workspace_bytes", "padding_waste_pct",
    "analysis_median_ms", "analysis_min_ms", "numeric_median_ms", "numeric_min_ms",
    "exact_flops", "executed_flops", "executed_padding_waste_pct", "row_runs",
    "left_runs", "has_fallback",
    "stage1_distinct_shapes", "stage1_grouped_submissions", "stage1_groups", "stage1_grouped_tasks", "stage1_fallback_tasks",
    "stage2_distinct_shapes", "stage2_grouped_submissions", "stage2_groups", "stage2_grouped_tasks", "stage2_fallback_tasks",
    "stage3_distinct_shapes", "stage3_grouped_submissions", "stage3_groups", "stage3_grouped_tasks", "stage3_fallback_tasks",
)
const GROUP_COLUMNS = (
    "case_id", "stage", "run", "kind", "m", "n", "k", "members",
)

function ensure_output(path, columns)
    mkpath(dirname(path))
    if !isfile(path) || filesize(path) == 0
        open(path, "w") do io
            println(io, join(columns, ','))
        end
    else
        first(readlines(path, keep=false)) == join(columns, ',') ||
            error("CSV schema mismatch at $path")
    end
end

function completed(path)
    isfile(path) || return Set{String}()
    lines = readlines(path)
    return Set(first(split(line, ',')) for line in Iterators.drop(lines, 1) if !isempty(line))
end

function append_row(path, values)
    open(path, "a") do io
        println(io, join(values, ','))
        flush(io)
    end
end

function samples_ms(f, C, T; reps=REPS, warmup=WARMUP)
    for _ in 1:warmup
        fill!(C, zero(T)); f(); CUDA.synchronize()
    end
    values = Float64[]
    sizehint!(values, reps)
    for _ in 1:reps
        fill!(C, zero(T)); CUDA.synchronize()
        start = time_ns(); f(); CUDA.synchronize()
        push!(values, (time_ns() - start) / 1e6)
    end
    return (median=median(values), minimum=minimum(values))
end

function analysis_timing(C, A, B, workspace, compute)
    warm = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
    CUDA.synchronize(); close(warm)
    values = Float64[]
    sizehint!(values, ANALYSIS_REPS)
    for _ in 1:ANALYSIS_REPS
        CUDA.synchronize(); start = time_ns()
        analysis = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
        CUDA.synchronize(); push!(values, (time_ns() - start) / 1e6)
        close(analysis)
    end
    return (median=median(values), minimum=minimum(values))
end

@inline factor_bytes(A) = (length(A.outer.data) + length(A.inner.data)) * sizeof(eltype(A))

function _task_dims(task)
    return NextLA._gemm_dims(task.transA, task.transB, task.A, task.B, task.C)[1:3]
end

"""Return aggregate descriptor counts plus one row per actual submitted shape."""
function stage_descriptor_rows(stage, case_id, stage_name, run_index)
    stage === nothing && return ((shapes=Set{NTuple{3,Int}}(), submissions=0, groups=0,
                                 grouped_tasks=0, fallback_tasks=0), Tuple[])
    bundle = stage isa NextLA.PreparedGroupedGemmBundle ? stage : nothing
    prepared = bundle === nothing ? stage : bundle.grouped
    fallback = bundle === nothing ? NextLA.GroupedGemmTask[] : bundle.fallback
    shapes = Set{NTuple{3,Int}}()
    rows = Tuple[]
    submissions = groups = grouped_tasks = 0
    if prepared !== nothing
        submissions = 1
        for group in eachindex(prepared.group_size)
            shape = (Int(prepared.m[group]), Int(prepared.n[group]), Int(prepared.k[group]))
            push!(shapes, shape)
            members = Int(prepared.group_size[group])
            groups += 1; grouped_tasks += members
            push!(rows, (case_id, stage_name, run_index, "grouped", shape..., members))
        end
    end
    for task in fallback
        shape = Tuple(Int.(collect(_task_dims(task))))
        push!(shapes, shape)
        push!(rows, (case_id, stage_name, run_index, "fallback", shape..., 1))
    end
    return ((shapes=shapes, submissions=submissions, groups=groups,
             grouped_tasks=grouped_tasks, fallback_tasks=length(fallback)), rows)
end

function descriptor_summary(analysis, case_id)
    totals = Dict(name => (shapes=Set{NTuple{3,Int}}(), submissions=0, groups=0,
                            grouped_tasks=0, fallback_tasks=0) for name in ("stage1", "stage2", "stage3"))
    rows = Tuple[]
    for (run_index, run) in enumerate(analysis.runs)
        for (name, stage) in (("stage1", run.stage1), ("stage2", run.stage2), ("stage3", run.stage3))
            value, stage_rows = stage_descriptor_rows(stage, case_id, name, run_index)
            total = totals[name]
            union!(total.shapes, value.shapes)
            totals[name] = (shapes=total.shapes,
                submissions=total.submissions + value.submissions,
                groups=total.groups + value.groups,
                grouped_tasks=total.grouped_tasks + value.grouped_tasks,
                fallback_tasks=total.fallback_tasks + value.fallback_tasks)
            append!(rows, stage_rows)
        end
    end
    return totals, rows
end

function case_id(N, b, policy, precision)
    return "N$(N)__b$(b)__r$(MIN_RANK)-$(MAX_RANK)__$(policy)__$(precision.name)__rows$(ROWS_PER_RUN)"
end

function benchmark_case(N, b, policy, precision)
    T, compute = precision.T, precision.compute
    A, B = DenseGemmCommon.generate_ftlr_operands(
        N, N, N, b, (MAX_RANK, MAX_RANK), T;
        seed=SEED, backend=CUDA.CUDABackend(), format=:compressed,
        rank_distribution=:uniform, min_rank=MIN_RANK, max_rank=MAX_RANK,
        compressed_execution_rank_policy=policy)
    C = CUDA.zeros(T, N, N)
    workspace_bytes = DenseGemmCommon._row_run_workspace_bytes(A, B, ROWS_PER_RUN)
    workspace = NextLA.DenseGemmWorkspace(A, B; bytes=workspace_bytes)
    id = case_id(N, b, policy, precision)

    symbolic = analysis_timing(C, A, B, workspace, compute)
    analysis = NextLA.analyze_compressed_gemm(C, A, B; workspace, compute)
    numeric = samples_ms(C, T) do
        TLRM.gemm!(C, A, B; workspace, alpha=one(T), beta=zero(T), compute, analysis)
    end
    exact_flops = DenseGemmCommon._tlr_tlr_exact_flops(A, B, workspace_bytes)
    executed_flops = DenseGemmCommon._tlr_tlr_executed_flops(A, B, workspace_bytes)
    padding = executed_flops == 0 ? 0.0 : 100 * (executed_flops - exact_flops) / executed_flops
    descriptors, descriptor_rows = descriptor_summary(analysis, id)
    profile = analysis.plan.profile
    runs = length(analysis.runs)
    left_runs = count(run -> run.fold === :left, analysis.runs)
    summary = (
        id, N, b, TILE_DIVISOR, MIN_RANK, MAX_RANK, policy, precision.name,
        ROWS_PER_RUN, factor_bytes(A) + factor_bytes(B), workspace_bytes, padding,
        symbolic.median, symbolic.minimum, numeric.median, numeric.minimum,
        exact_flops, executed_flops, padding, runs, left_runs, analysis.has_fallback,
        length(descriptors["stage1"].shapes), descriptors["stage1"].submissions, descriptors["stage1"].groups, descriptors["stage1"].grouped_tasks, descriptors["stage1"].fallback_tasks,
        length(descriptors["stage2"].shapes), descriptors["stage2"].submissions, descriptors["stage2"].groups, descriptors["stage2"].grouped_tasks, descriptors["stage2"].fallback_tasks,
        length(descriptors["stage3"].shapes), descriptors["stage3"].submissions, descriptors["stage3"].groups, descriptors["stage3"].grouped_tasks, descriptors["stage3"].fallback_tasks,
    )
    @printf("%-58s numeric=%8.3f ms analysis=%8.3f ms padding=%5.1f%% groups=(%d,%d,%d) fallback=(%d,%d,%d)\n",
        id, numeric.median, symbolic.median, padding,
        descriptors["stage1"].groups, descriptors["stage2"].groups, descriptors["stage3"].groups,
        descriptors["stage1"].fallback_tasks, descriptors["stage2"].fallback_tasks, descriptors["stage3"].fallback_tasks)
    close(analysis)
    A = B = C = workspace = nothing
    GC.gc(true); CUDA.reclaim()
    return summary, descriptor_rows
end

function run_rank_bucketing()
    CUDA.functional() || error("rank-bucketing ablation requires CUDA")
    TILE_DIVISOR > 0 || throw(ArgumentError("NEXTLA_BUCKET_TILE_DIVISOR must be positive"))
    MIN_RANK >= 0 && MIN_RANK <= MAX_RANK || throw(ArgumentError("invalid rank interval"))
    all(p -> p in (:exact, :q8, :q16, :pow2), POLICIES) ||
        throw(ArgumentError("policies must be exact,q8,q16,pow2"))
    all(haskey(PRECISIONS, p) for p in PRECISION_NAMES) ||
        throw(ArgumentError("unknown precision in NEXTLA_BUCKET_PRECISIONS"))
    ensure_output(OUTPUT, SUMMARY_COLUMNS)
    ensure_output(GROUP_OUTPUT, GROUP_COLUMNS)
    done = completed(OUTPUT)
    @printf("Rank-bucketing ablation: H/W=%d/%d analysis=%d rows=%d output=%s\n",
        WARMUP, REPS, ANALYSIS_REPS, ROWS_PER_RUN, OUTPUT)
    for N in SIZES
        N % TILE_DIVISOR == 0 || throw(ArgumentError("N=$N is not divisible by tile-grid size $TILE_DIVISOR"))
        b = N ÷ TILE_DIVISOR
        MAX_RANK <= b || throw(ArgumentError("max rank $MAX_RANK exceeds tile size $b"))
        for precision_name in PRECISION_NAMES, policy in POLICIES
            precision = PRECISIONS[precision_name]
            id = case_id(N, b, policy, precision)
            id in done && (@printf("skip %s\n", id); continue)
            summary, rows = benchmark_case(N, b, policy, precision)
            append_row(OUTPUT, summary)
            for row in rows
                append_row(GROUP_OUTPUT, row)
            end
            push!(done, id)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_rank_bucketing()
end
