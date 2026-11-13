using CUDA, LinearAlgebra, Printf, KernelAbstractions, Dates
include("benchmark.jl")   # run_manual_benchmark
include("flops.jl")       # calculate_gflops(flops, time_ns)

# Load the library pieces if the session hasn't already (a REPL that did `using NextLA` or included
# them by hand keeps its own definitions).
isdefined(Main, :FullMixedPrec) || include(joinpath(@__DIR__, "..", "src", "fullmixedprec.jl"))
isdefined(Main, :recgemm!)      || include(joinpath(@__DIR__, "..", "src", "recgemm.jl"))

# ---------------------------------------------------------------------------
# Tall-and-skinny benchmark for recgemm!  (companion to test/recgemm.jl, same configs / alpha=-1, beta=1)
#
#   family "tall"      C(m x n) = A(m x k) * B(k x n),  m >> k = n    (e.g. trailing-panel update in LU/QR)
#   family "shortfat"  C(m x m) = A(m x k) * B(k x m),  m >> k        (small inner dimension, large output)
#
# Differences from test/recgemm.jl worth knowing when comparing numbers:
#   * Timing covers recgemm! only. The FullMixedPrec(C) construction/conversion is done once outside the timed
#     region (test/recgemm.jl rebuilds it inside the closure, which adds an O(m*n) copy to every sample).
#     C therefore accumulates -A*B on every sample; that does not change the cost of a call.
#   * Inputs are generated on the GPU (CUDA.randn), so m ~ 10^6 doesn't need a giant host randn.
#   * Error is relative to an FP64 CUBLAS GEMM on the same (upcast) operands.
#
# Environment knobs:
#   TS_QUICK=1            small grid for a smoke test (a few seconds per scenario)
#   CUTLASS_PROFILER=...  path to cutlass_profiler (else `cutlass_profiler` on PATH, else CUTLASS is skipped)
#   CUTLASS_KERNELS=...   passed to --kernels
# ---------------------------------------------------------------------------
flops_gemm(m, n, k) = 2.0 * m * n * k

const QUICK = get(ENV, "TS_QUICK", "0") == "1"
# Every precision list needs min(n_cols(C)) >= 2^(length-1) to reach its base case (longest list below has 7 entries),
# hence k = n >= 64 for the tall family.
const KS_TALL = QUICK ? [64]             : [64, 256, 1024]
const MS_TALL = QUICK ? [4096, 16384]    : [4096, 16384, 65536, 262144, 1048576]
const KS_SF   = QUICK ? [64]             : [64, 256, 1024]
const MS_SF   = QUICK ? [1024, 4096]     : [1024, 4096, 16384, 32768]

problems() = vcat(
    [(family = "tall",     m = m, n = k, k = k) for k in KS_TALL for m in MS_TALL],
    [(family = "shortfat", m = m, n = m, k = k) for k in KS_SF   for m in MS_SF],
)

# Worst-case device bytes per problem (FP64 everywhere): A, B and their FP64 copies, plus C, FP64 ground truth,
# the FullMixedPrec copy, the reconstruction and its FP64 upcast.
est_bytes(p) = 8.0 * (2 * (p.m * p.k + p.k * p.n) + 5 * p.m * p.n)

# GPU-side normal(0, 0.001) in element type T (Float16 goes via Float32).
gpu_randn(::Type{T}, dims...) where {T} =
    T === Float64 ? CUDA.randn(Float64, dims...) .* 0.001 : T.(CUDA.randn(Float32, dims...) .* 0.001f0)

# ---------------------------------------------------------------------------
# CUTLASS via cutlass_profiler (same approach as test/recgemm.jl, generalised to m x n x k)
# ---------------------------------------------------------------------------
const CUTLASS_PROFILER = get(ENV, "CUTLASS_PROFILER", something(Sys.which("cutlass_profiler"), ""))
const CUTLASS_TYPES = ["CUTLASS F32" => "f32", "CUTLASS F64" => "f64"]

function cutlass_gemm_ms(profiler::AbstractString, m::Integer, n::Integer, k::Integer, T::AbstractString; alpha = -1.0, beta = 1.0)
    mktempdir() do dir
        out = joinpath(dir, "cutlass_report")
        big = flops_gemm(m, n, k) >= flops_gemm(16384, 16384, 16384)
        args = ["--operation=Gemm", "--m=$m", "--n=$n", "--k=$k",
                "--A=$T:column", "--B=$T:column", "--C=$T:column",
                "--accumulator-type=$T", "--alpha=$alpha", "--beta=$beta",
                "--warmup-iterations=$(big ? 1 : 5)", "--profiling-iterations=$(big ? 3 : 10)",
                "--verification-enabled=false", "--output=$out"]
        haskey(ENV, "CUTLASS_KERNELS") && push!(args, "--kernels=" * ENV["CUTLASS_KERNELS"])
        run(pipeline(Cmd([String(profiler); args]); stdout = devnull))

        csvs = filter(f -> endswith(f, ".csv"), readdir(dir; join = true))
        isempty(csvs) && error("cutlass_profiler wrote no CSV (no $T GEMM kernels built into the profiler?)")

        best_ms, best_op, header = Inf, "", String[]
        for f in csvs
            lines = readlines(f)
            isempty(lines) && continue
            header = String.(strip.(split(lines[1], ',')))
            i_rt = findfirst(==("Runtime"), header)
            i_st = findfirst(==("Status"), header)
            i_op = findfirst(==("Operation"), header)
            i_rt === nothing && continue
            for line in lines[2:end]
                cols = String.(strip.(split(line, ',')))
                length(cols) == length(header) || continue
                i_st !== nothing && lowercase(cols[i_st]) != "success" && continue
                rt = tryparse(Float64, cols[i_rt])
                (rt === nothing || rt <= 0) && continue
                if rt < best_ms
                    best_ms, best_op = rt, i_op === nothing ? "" : cols[i_op]
                end
            end
        end
        isfinite(best_ms) || error("no successful kernel in profiler CSV (columns: $(join(header, ", ")))")
        return best_ms, best_op
    end
end

# ---------------------------------------------------------------------------
# Reports: a markdown file and a tidy CSV, both written to pwd() with a timestamp.
# ---------------------------------------------------------------------------
function md_table(io, title, impls, ms, recs, field, fmt)
    println(io, "#### ", title, "\n")
    println(io, "| Implementation | ", join(ms, " | "), " |")
    println(io, "|:--|", repeat("--:|", length(ms)))
    for impl in impls
        cells = map(ms) do m
            i = findfirst(r -> r.impl == impl && r.m == m, recs)
            (i === nothing || isnan(getfield(recs[i], field))) ? "—" : fmt(getfield(recs[i], field))
        end
        println(io, "| ", impl, " | ", join(cells, " | "), " |")
    end
    println(io)
end

function write_reports(records, console_log; dir = pwd())
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    md_path  = joinpath(dir, "recgemm_tallskinny_results_$stamp.md")
    csv_path = joinpath(dir, "recgemm_tallskinny_results_$stamp.csv")

    open(csv_path, "w") do io
        println(io, "family,m,n,k,impl,runtime_ms,gflops,rel_err,note")
        for r in records
            println(io, join([r.family, r.m, r.n, r.k, "\"" * r.impl * "\"", r.runtime_ms, r.gflops, r.rel_err,
                              "\"" * r.note * "\""], ","))
        end
    end

    open(md_path, "w") do io
        println(io, "# recgemm! tall-and-skinny benchmark results\n")
        println(io, "- **Date:** ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "- **GPU:** ", CUDA.functional() ? CUDA.name(CUDA.device()) : "n/a")
        println(io, "- **Julia:** ", VERSION, " | **CUDA.jl:** ", pkgversion(CUDA))
        println(io, "- **FLOP model:** 2mnk per GEMM; GFLOP/s = 2mnk / best runtime (recgemm! only, setup excluded)")
        println(io, "- **CUTLASS profiler:** ", isempty(CUTLASS_PROFILER) ? "not found (skipped)" : CUTLASS_PROFILER, "\n")

        for fam in unique(r.family for r in records)
            println(io, "## ", fam == "tall" ? "Tall × small square: C(m×k) = A(m×k)·B(k×k)" :
                                               "Short-fat × tall: C(m×m) = A(m×k)·B(k×m)", "\n")
            for k in sort(unique(r.k for r in records if r.family == fam))
                recs = filter(r -> r.family == fam && r.k == k, records)
                ms = sort(unique(r.m for r in recs))
                impls = unique(r.impl for r in recs)
                println(io, "### k = ", k, "\n")
                md_table(io, "Throughput (GFLOP/s) vs m", impls, ms, recs, :gflops, x -> @sprintf("%.1f", x))
                md_table(io, "Runtime (ms, best of N) vs m", impls, ms, recs, :runtime_ms, x -> @sprintf("%.3f", x))
                md_table(io, "Relative error vs. FP64 CUBLAS", impls, ms, recs, :rel_err, x -> @sprintf("%.2e", x))
            end
        end

        notes = filter(r -> !isempty(r.note), records)
        if !isempty(notes)
            println(io, "### Notes\n")
            for r in notes
                println(io, "- ", r.family, " m=", r.m, " n=", r.n, " k=", r.k, ", ", r.impl, ": `", r.note, "`")
            end
            println(io)
        end

        println(io, "## Raw console output\n")
        println(io, "```text")
        print(io, console_log)
        println(io, "\n```")
    end
    return md_path, csv_path
end

function run_tallskinny_benchmark()
    test_scenarios = [
        "Pure F16"             => [Float16, Float16, Float16],
        "Pure F32"             => [Float32, Float32, Float32],
        "Pure F64"             => [Float64, Float64, Float64],
        "[F32, F32, F64]"      => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]" => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]" => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"      => [Float32, Float64, Float64],
        "[F16, F16, F32]"      => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F32]"      => [Float16, Float32, Float32],
        "[F16, F32]"           => [Float16, Float32],
        "[F16, F64]"           => [Float16, Float64],
        "[F16, F16, F16, F64]" => [Float16, Float16, Float16, Float64],
        "[F16, F16, F16, F16, F64]" => [Float16, Float16, Float16, Float16, Float64],
        "[F16, F16, F16, F16, F16, F16, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float64],
    ]

    records = NamedTuple[]
    log_io = IOBuffer()
    emit(s) = (print(s); print(log_io, s))
    record!(p, impl, runtime_ms, gflops; rel_err = NaN, note = "") =
        push!(records, (family = p.family, m = p.m, n = p.n, k = p.k, impl = impl,
                        runtime_ms = runtime_ms, gflops = gflops, rel_err = rel_err, note = note))

    emit("🚀 Starting recgemm! tall-and-skinny benchmark...$(QUICK ? " (TS_QUICK)" : "")\n")

    try
        for p in problems()
            emit("\n" * "-"^70 * "\n")
            emit("[$(p.family)] C($(p.m) x $(p.n)) = A($(p.m) x $(p.k)) * B($(p.k) x $(p.n))\n")

            GC.gc(); CUDA.reclaim()
            if est_bytes(p) > 0.9 * CUDA.available_memory()
                emit(@sprintf("  [skipped] needs ~%.1f GB, only %.1f GB free\n", est_bytes(p) / 2^30, CUDA.available_memory() / 2^30))
                continue
            end
            gemm_flops = flops_gemm(p.m, p.n, p.k)

            for (name, precisions) in test_scenarios
                T_out = precisions[end]
                alpha, beta = T_out(-1.0), T_out(1.0)

                d_A = gpu_randn(T_out, p.m, p.k)
                d_B = gpu_randn(T_out, p.k, p.n)
                d_C_orig = CUDA.zeros(T_out, p.m, p.n)

                # Ground truth in FP64
                d_C_truth = CUDA.zeros(Float64, p.m, p.n)
                CUBLAS.gemm!('N', 'N', -1.0, Float64.(d_A), Float64.(d_B), 1.0, d_C_truth)

                # Accuracy
                C_mixed = FullMixedPrec(copy(d_C_orig); precisions = precisions)
                recgemm!(alpha, d_A, d_B, beta, C_mixed)
                X = Float64.(reconstruct_matrix(C_mixed))
                X .-= d_C_truth
                relative_error = max(norm(X) / norm(d_C_truth), 1e-20)
                X = nothing

                # Performance: recgemm! only (setup outside the timed region)
                C_perf = FullMixedPrec(copy(d_C_orig); precisions = precisions)
                backend = KernelAbstractions.get_backend(d_A)
                time_ns = run_manual_benchmark(backend) do
                    CUDA.@sync recgemm!(alpha, d_A, d_B, beta, C_perf)
                end
                runtime_ms = time_ns / 1_000_000
                gflops = calculate_gflops(gemm_flops, time_ns)
                record!(p, name, runtime_ms, gflops; rel_err = relative_error)
                emit(@sprintf("  %-38s | Rel. Error: %9.2e | Runtime: %9.3f ms | %10.1f GFLOP/s\n",
                              name, relative_error, runtime_ms, gflops))

                d_A = d_B = d_C_orig = d_C_truth = C_mixed = C_perf = nothing
                GC.gc(); CUDA.reclaim()
            end

            emit("\n--- Benchmarking standard CUBLAS.gemm! ---\n")
            for (name, T_prec) in ("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)
                d_A = gpu_randn(T_prec, p.m, p.k)
                d_B = gpu_randn(T_prec, p.k, p.n)
                d_C = CUDA.zeros(T_prec, p.m, p.n)
                backend = KernelAbstractions.get_backend(d_A)
                time_ns = run_manual_benchmark(backend) do
                    CUBLAS.gemm!('N', 'N', T_prec(-1.0), d_A, d_B, T_prec(1.0), d_C)
                end
                runtime_ms = time_ns / 1_000_000
                gflops = calculate_gflops(gemm_flops, time_ns)
                record!(p, name, runtime_ms, gflops)
                emit(@sprintf("  %-38s | Runtime: %9.3f ms | %10.1f GFLOP/s\n", name, runtime_ms, gflops))
                d_A = d_B = d_C = nothing
                GC.gc(); CUDA.reclaim()
            end

            emit("\n--- Benchmarking CUTLASS (cutlass_profiler) ---\n")
            if isempty(CUTLASS_PROFILER)
                emit("  [skipped] cutlass_profiler not found; set ENV[\"CUTLASS_PROFILER\"] or put it on PATH\n")
            else
                GC.gc(); CUDA.reclaim()
                for (name, T_str) in CUTLASS_TYPES
                    try
                        runtime_ms, kernel = cutlass_gemm_ms(CUTLASS_PROFILER, p.m, p.n, p.k, T_str)
                        gflops = calculate_gflops(gemm_flops, runtime_ms * 1e6)
                        record!(p, name, runtime_ms, gflops; note = kernel)
                        emit(@sprintf("  %-38s | Runtime: %9.3f ms | %10.1f GFLOP/s | %s\n", name, runtime_ms, gflops, kernel))
                    catch e
                        emit(@sprintf("  %-38s | FAILED: %s\n", name, first(split(sprint(showerror, e), '\n'))))
                    end
                end
            end
        end
        emit("\n" * "="^60 * "\n✅ Benchmark complete.\n")
    finally
        md_path, csv_path = write_reports(records, String(take!(copy(log_io))))
        println("📝 Results written to $md_path and $csv_path")
    end
end

run_tallskinny_benchmark()
