"""Benchmark dense × CompressedFTLR + dense → dense."""

include(joinpath(@__DIR__, "compressed_dense.jl"))

const DENSE_COMPRESSED_OUTPUT = get(
    ENV, "NEXTLA_DENSE_COMPRESSED_OUTPUT",
    joinpath(@__DIR__, "results", "dense_compressed.csv"))

const DENSE_COMPRESSED_COLUMNS = (
    "case_id", "N", "tile_size", "profile", "distribution", "min_rank", "max_rank",
    "precision", "workspace_bytes", "analysis_ms", "analysis_min_ms",
    "transient_median_ms", "transient_min_ms", "analyzed_median_ms", "analyzed_min_ms",
    "dense_median_ms", "speedup_vs_dense", "exact_flops", "executed_flops",
    "padding_waste_pct", "executed_gflops", "dense_gflops",
)

function dense_compressed_case_id(N, b, profile, precision)
    return "N$(N)__b$(b)__$(profile.name)__$(precision.name)"
end

function dense_compressed_flops(B, N; execution::Bool)
    _, qn = TLRM.grid_size(B)
    qk, _ = TLRM.grid_size(B)
    total = 0.0
    for k in 1:qk, j in 1:qn
        tk, nj = TLRM.tile_size(B, k, j)
        rank = execution ?
            TLRM._compressed_ftlr_execution_rank(B, k, j) :
            TLRM._compressed_ftlr_rank(B, k, j)
        total += 2.0 * rank * N * (tk + nj)
    end
    return total
end

function benchmark_dense_compressed_case(N, b, profile, precision, dense)
    T = precision.T
    compute = precision.compute
    Acompressed, B = DenseGemmCommon.generate_ftlr_operands(
        N, N, N, b, profile.ranks, T;
        seed=20260729, backend=CUDA.CUDABackend(), format=:compressed,
        rank_distribution=profile.distribution,
        min_rank=profile.lo, max_rank=profile.hi)
    A = CUDA.zeros(T, N, N)
    NextLA.uncompress!(A, Acompressed)
    CUDA.synchronize()
    Acompressed = nothing
    GC.gc(true)
    CUDA.reclaim()

    C = CUDA.zeros(T, N, N)
    maximum_rank = maximum(TLRM.execution_ranks(B))
    workspace_bytes = maximum_rank * N * sizeof(T)
    workspace = NextLA.DenseGemmWorkspace(B, workspace_bytes)
    transient = samples_ms(C, T) do
        TLRM.gemm!(C, A, B; workspace, alpha=one(T), beta=one(T), compute)
    end
    analysis, analysis_timing = time_analysis(C, A, B, workspace, compute)
    analyzed = samples_ms(C, T) do
        TLRM.gemm!(
            C, A, B; workspace, alpha=one(T), beta=one(T), compute, analysis)
    end

    exact = dense_compressed_flops(B, N; execution=false)
    executed = dense_compressed_flops(B, N; execution=true)
    padding_waste = executed == 0 ? 0.0 : 100 * (executed - exact) / executed
    dense_flops = 2.0 * N^3
    result = (
        dense_compressed_case_id(N, b, profile, precision),
        N, b, profile.name, profile.distribution, profile.lo, profile.hi,
        precision.name, workspace_bytes,
        analysis_timing.median, analysis_timing.minimum,
        transient.median, transient.minimum, analyzed.median, analyzed.minimum,
        dense.median, dense.median / analyzed.median, exact, executed, padding_waste,
        executed / (analyzed.median * 1e6), dense_flops / (dense.median * 1e6),
    )
    @printf(
        "%-55s analysis=%8.3f ms transient=%8.3f ms numeric=%8.3f ms dense=%8.3f ms speedup=%5.2fx padding=%5.1f%%\n",
        first(result), analysis_timing.median, transient.median, analyzed.median,
        dense.median, dense.median / analyzed.median, padding_waste)
    close(analysis)
    A = B = C = workspace = nothing
    GC.gc(true)
    CUDA.reclaim()
    return result
end

function run_dense_compressed()
    CUDA.functional() || error("dense × compressed benchmark requires CUDA")
    ensure_output(DENSE_COMPRESSED_OUTPUT, DENSE_COMPRESSED_COLUMNS)
    done = completed(DENSE_COMPRESSED_OUTPUT, DENSE_COMPRESSED_COLUMNS)
    @printf("Dense × compressed → dense benchmark: H/W=%d/%d output=%s\n",
            WARMUP, REPS, DENSE_COMPRESSED_OUTPUT)
    dense_cache = Dict{Tuple{Int,String},NamedTuple}()
    for N in SIZES, divisor in TILE_DIVISORS
        N % divisor == 0 || throw(ArgumentError(
            "matrix size $N is not divisible by tile divisor $divisor"))
        b = N ÷ divisor
        for precision in PRECISIONS, profile in rank_profiles(b)
            id = dense_compressed_case_id(N, b, profile, precision)
            occursin(CASE_FILTER, id) || continue
            id in done && (@printf("skip %s\n", id); continue)
            dense = get!(dense_cache, (N, precision.name)) do
                dense_timing(N, precision.T, precision.compute)
            end
            append_row(DENSE_COMPRESSED_OUTPUT,
                benchmark_dense_compressed_case(N, b, profile, precision, dense))
            push!(done, id)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dense_compressed()
end
