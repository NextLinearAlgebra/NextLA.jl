"""Sweep the CompressedFTLR dense-result scheduler from one to eight rows/run."""

include(joinpath(@__DIR__, "compressed_dense.jl"))

const ROW_SWEEP_SIZES = parse_ints(
    "NEXTLA_ROWS_SIZES", "2048,4096,8192,16384,32768")
const ROW_SWEEP_VALUES = parse_ints("NEXTLA_ROWS_VALUES", "1,2,3,4,5,6,7,8")
const ROW_SWEEP_OUTPUT = get(
    ENV, "NEXTLA_ROWS_OUTPUT", joinpath(@__DIR__, "results", "rows_per_run.csv"))

const ROW_SWEEP_COLUMNS = (
    "case_id", "N", "tile_size", "rank", "precision", "rows_per_run",
    "workspace_bytes", "analysis_ms", "analysis_min_ms",
    "numeric_median_ms", "numeric_min_ms", "executed_flops", "executed_gflops",
)

row_sweep_case_id(N, b, rank, precision, rows) =
    "N$(N)__b$(b)__r$(rank)__$(precision.name)__rows$(rows)"

function run_rows_per_run()
    CUDA.functional() || error("rows-per-run benchmark requires CUDA")
    ensure_output(ROW_SWEEP_OUTPUT, ROW_SWEEP_COLUMNS)
    done = completed(ROW_SWEEP_OUTPUT, ROW_SWEEP_COLUMNS)
    @printf("Rows/run benchmark: H/W=%d/%d output=%s\n",
            WARMUP, REPS, ROW_SWEEP_OUTPUT)

    for N in ROW_SWEEP_SIZES, precision in PRECISIONS
        N % 8 == 0 || throw(ArgumentError("matrix size $N must be divisible by 8"))
        b = N ÷ 8
        rank = max(1, b ÷ 8)
        T = precision.T
        A, B = DenseGemmCommon.generate_ftlr_operands(
            N, N, N, b, (rank, rank), T;
            seed=20260729, backend=CUDA.CUDABackend(), format=:compressed,
            rank_distribution=:constant, min_rank=rank, max_rank=rank)
        C = CUDA.zeros(T, N, N)

        for rows in ROW_SWEEP_VALUES
            1 <= rows <= 8 ||
                throw(ArgumentError("rows-per-run values must lie in 1:8"))
            id = row_sweep_case_id(N, b, rank, precision, rows)
            occursin(CASE_FILTER, id) || continue
            id in done && (@printf("skip %s\n", id); continue)
            workspace_bytes =
                DenseGemmCommon._row_run_workspace_bytes(A, B, rows)
            workspace = NextLA.DenseGemmWorkspace(A, B; bytes=workspace_bytes)
            analysis, analysis_timing =
                time_analysis(C, A, B, workspace, precision.compute)
            timing = samples_ms(C, T) do
                TLRM.gemm!(
                    C, A, B; workspace, alpha=one(T), beta=zero(T),
                    compute=precision.compute, analysis)
            end
            executed =
                DenseGemmCommon._tlr_tlr_executed_flops(A, B, workspace_bytes)
            row = (
                id, N, b, rank, precision.name, rows, workspace_bytes,
                analysis_timing.median, analysis_timing.minimum,
                timing.median, timing.minimum, executed,
                executed / (timing.median * 1e6),
            )
            append_row(ROW_SWEEP_OUTPUT, row)
            push!(done, id)
            @printf("%-55s analysis=%8.3f ms numeric=%8.3f ms %10.2f GFLOP/s\n",
                    id, analysis_timing.median, timing.median, last(row))
            close(analysis)
            workspace = nothing
        end
        A = B = C = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_rows_per_run()
end
