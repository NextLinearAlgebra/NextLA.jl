# Poster-oriented fixed-rank PaddedFTLR × PaddedFTLR → dense benchmark.
#
# A uses tile-row-major storage and B uses tile-column-major storage. Matrix
# construction and reusable numerical workspace allocation are excluded from
# timing. Unlike the compressed benchmark, this path has no symbolic analysis.

include(joinpath(@__DIR__, "compressed_dense.jl"))

const PADDED_DENSE_OUTPUT = get(
    ENV, "NEXTLA_PADDED_DENSE_OUTPUT",
    joinpath(@__DIR__, "results", "padded_dense_v2.csv"))

const PADDED_DENSE_COLUMNS = (
    "case_id", "N", "tile_size", "rank", "profile", "precision",
    "workspace_bytes", "padded_median_ms", "padded_min_ms",
    "dense_median_ms", "dense_min_ms", "speedup_vs_dense",
    "executed_flops", "padded_gflops", "dense_gflops",
)

function padded_rank_profiles(b)
    r32 = max(1, b ÷ 32)
    r16 = max(1, b ÷ 16)
    r8 = max(1, b ÷ 8)
    return (
        (name="constant_b32", rank=r32),
        (name="constant_b16", rank=r16),
        (name="constant_b8", rank=r8),
    )
end

padded_dense_case_id(N, b, profile, precision) =
    "N$(N)__b$(b)__$(profile.name)__$(precision.name)"

function padded_dense_executed_flops(N, b, rank)
    q = N ÷ b
    # TileRowMajor(A) × TileColMajor(B) selects FoldLeft:
    # Stage 1 V'W, Stage 2 U(V'W), and one rank-stacked terminal GEMM.
    return 2.0 * q^3 * (2 * b * rank^2 + b^2 * rank)
end

function benchmark_padded_dense_case(N, b, profile, precision, dense)
    T = precision.T
    compute = precision.compute
    rank = profile.rank
    A, B = DenseGemmCommon.generate_ftlr_operands(
        N, N, N, b, (rank, rank), T;
        seed=20260729,
        backend=CUDA.CUDABackend(),
        format=:padded,
        rank_distribution=:constant,
        min_rank=rank,
        max_rank=rank,
        padded_orders=(TLRM.TileRowMajor, TLRM.TileColMajor),
    )
    C = CUDA.zeros(T, N, N)
    workspace_bytes = TLRM.gemm_maximum_workspace_bytes(A, B)
    workspace = NextLA.DenseGemmWorkspace(A, B; bytes=workspace_bytes)

    padded = samples_ms(C, T) do
        TLRM.gemm!(
            C, A, B;
            workspace,
            alpha=one(T),
            beta=one(T),
            compute,
        )
    end

    executed = padded_dense_executed_flops(N, b, rank)
    dense_flops = 2.0 * N^3
    result = (
        padded_dense_case_id(N, b, profile, precision),
        N, b, rank, profile.name, precision.name, workspace_bytes,
        padded.median, padded.minimum, dense.median, dense.minimum,
        dense.median / padded.median, executed,
        executed / (padded.median * 1e6),
        dense_flops / (dense.median * 1e6),
    )
    @printf(
        "%-55s padded=%8.3f ms dense=%8.3f ms speedup=%5.2fx workspace=%9.3f MiB\n",
        first(result), padded.median, dense.median,
        dense.median / padded.median, workspace_bytes / 2.0^20)

    A = B = C = workspace = nothing
    GC.gc(true)
    CUDA.reclaim()
    return result
end

function run_padded_dense()
    CUDA.functional() || error("padded dense-output benchmark requires CUDA")
    ensure_output(PADDED_DENSE_OUTPUT, PADDED_DENSE_COLUMNS)
    done = completed(PADDED_DENSE_OUTPUT, PADDED_DENSE_COLUMNS)
    @printf(
        "Padded dense-output benchmark: H/W=%d/%d output=%s\n",
        WARMUP, REPS, PADDED_DENSE_OUTPUT)

    dense_cache = Dict{Tuple{Int,String},NamedTuple}()
    for N in SIZES, divisor in TILE_DIVISORS
        divisor > 0 || throw(ArgumentError("tile divisors must be positive"))
        N % divisor == 0 || throw(ArgumentError(
            "matrix size $N is not divisible by tile divisor $divisor"))
        b = N ÷ divisor
        for precision in PRECISIONS, profile in padded_rank_profiles(b)
            id = padded_dense_case_id(N, b, profile, precision)
            occursin(CASE_FILTER, id) || continue
            id in done && (@printf("skip %s\n", id); continue)
            dense = get!(dense_cache, (N, precision.name)) do
                dense_timing(N, precision.T, precision.compute)
            end
            append_row(PADDED_DENSE_OUTPUT,
                benchmark_padded_dense_case(N, b, profile, precision, dense))
            push!(done, id)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_padded_dense()
end
