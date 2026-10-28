using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions, Dates
include("benchmark.jl")
include("flops.jl")   # calculate_gflops(flops, time_ns)

# ---------------------------------------------------------------------------
# FLOP model: C <- alpha*A*B + beta*C, square n x n. Standard GEMM count (2mnk);
# the O(n^2) alpha/beta terms are ignored, matching CUBLAS/CUTLASS convention.
# ---------------------------------------------------------------------------
flops_gemm(m, n, k) = 2.0 * m * n * k

# ---------------------------------------------------------------------------
# CUTLASS via cutlass_profiler (https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
#   - Binary: $CUTLASS_PROFILER, else `cutlass_profiler` on PATH. If neither, CUTLASS is skipped.
#   - Optional: $CUTLASS_KERNELS restricts kernels (maps to --kernels=...), e.g. "sgemm" or "tensorop".
#   - Same problem as CUBLAS.gemm!: column-major, N/N, alpha=-1, beta=1. The profiler tries every
#     matching kernel; we report the fastest successful one (verification off, we only time).
# ---------------------------------------------------------------------------
const CUTLASS_PROFILER = get(ENV, "CUTLASS_PROFILER", something(Sys.which("cutlass_profiler"), ""))
const CUTLASS_TYPES = ["CUTLASS F32" => "f32", "CUTLASS F64" => "f64"]  # name => profiler element type (A/B/C/accum)

function cutlass_gemm_ms(profiler::AbstractString, n::Integer, T::AbstractString; alpha = -1.0, beta = 1.0)
    mktempdir() do dir
        out = joinpath(dir, "cutlass_report")
        big = n >= 16384
        args = ["--operation=Gemm", "--m=$n", "--n=$n", "--k=$n",
                "--A=$T:column", "--B=$T:column", "--C=$T:column",
                "--accumulator-type=$T", "--alpha=$alpha", "--beta=$beta",
                "--warmup-iterations=$(big ? 1 : 5)", "--profiling-iterations=$(big ? 3 : 10)",
                "--verification-enabled=false", "--output=$out"]
        haskey(ENV, "CUTLASS_KERNELS") && push!(args, "--kernels=" * ENV["CUTLASS_KERNELS"])
        run(pipeline(Cmd([String(profiler); args]); stdout = devnull))

        # The profiler appends the operation kind to --output (e.g. <out>.gemm.csv); don't assume the exact name.
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
# Markdown report (written next to wherever the script was launched from, i.e. pwd(),
# same place savefig() drops the plots). Timestamped so repeated runs never overwrite each other.
# ---------------------------------------------------------------------------
function md_table(io, title, impls, n_values, records, field, fmt)
    println(io, "### ", title, "\n")
    println(io, "| Implementation | ", join(n_values, " | "), " |")
    println(io, "|:--|", repeat("--:|", length(n_values)))
    for impl in impls
        cells = map(n_values) do n
            i = findfirst(r -> r.impl == impl && r.n == n, records)
            (i === nothing || isnan(getfield(records[i], field))) ? "—" : fmt(getfield(records[i], field))
        end
        println(io, "| ", impl, " | ", join(cells, " | "), " |")
    end
    println(io)
end

function write_results_markdown(records, n_values, console_log; dir = pwd())
    path = joinpath(dir, "recgemm_results_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".md")
    impls = unique(r.impl for r in records)
    open(path, "w") do io
        println(io, "# recgemm! benchmark results\n")
        println(io, "- **Date:** ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "- **GPU:** ", CUDA.functional() ? CUDA.name(CUDA.device()) : "n/a")
        println(io, "- **Julia:** ", VERSION, " | **CUDA.jl:** ", pkgversion(CUDA))
        println(io, "- **Sizes (n x n x n):** ", join(n_values, ", "))
        println(io, "- **FLOP model:** 2n³ per GEMM; GFLOP/s = 2n³ / best runtime")
        println(io, "- **CUTLASS profiler:** ", isempty(CUTLASS_PROFILER) ? "not found (skipped)" : CUTLASS_PROFILER, "\n")

        md_table(io, "Throughput (GFLOP/s)", impls, n_values, records, :gflops, x -> @sprintf("%.1f", x))
        md_table(io, "Runtime (ms, best of N)", impls, n_values, records, :runtime_ms, x -> @sprintf("%.3f", x))
        md_table(io, "Relative error vs. FP64 CUBLAS", impls, n_values, records, :rel_err, x -> @sprintf("%.2e", x))

        notes = filter(r -> !isempty(r.note), records)
        if !isempty(notes)
            println(io, "### Notes\n")
            for r in notes
                println(io, "- n=", r.n, ", ", r.impl, ": `", r.note, "`")
            end
            println(io)
        end

        println(io, "## Raw console output\n")
        println(io, "```text")
        print(io, console_log)
        println(io, "\n```")
    end
    return path
end

function run_recgemm_benchmark()
    # Define the matrix sizes to test
    n_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768] #16, 32, 256, 512, 1024, ... 65536

    # Define the different mixed-precision scenarios
    test_scenarios = Dict(
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

        # --- Compelling New Configurations ---

        # Smooth Gradients (F16 leaf -> F32 mid -> F64 root)
        # "[F16, F16, F32, F64]" => [Float16, Float16, Float32, Float64],
        # "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        # "[F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float32, Float64],
        # "[F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],

        # # Extreme Jumps (F16 direct to F64)
        "[F16, F64]"           => [Float16, Float64],
        "[F16, F16, F16, F64]" => [Float16, Float16, Float16, Float64],
        "[F16, F16, F16, F16, F64]" => [Float16, Float16, Float16, Float16, Float64],
        "[F16, F16, F16, F16, F16, F16, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float64]
    )

    # Simplified dictionaries to store results
    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))

    cublas_runtime_results = Dict(
        "CUBLAS F32" => Float64[],
        "CUBLAS F64" => Float64[]
    )

    # Flat record of every measurement (feeds the markdown report) + mirror of everything printed.
    records = NamedTuple[]
    log_io = IOBuffer()
    emit(s) = (print(s); print(log_io, s))
    record!(n, impl, runtime_ms, gflops; rel_err = NaN, note = "") =
        push!(records, (n = n, impl = impl, runtime_ms = runtime_ms, gflops = gflops, rel_err = rel_err, note = note))

    emit("🚀 Starting recgemm! Benchmark...\n")

    # try/finally: even if a large size OOMs or CUTLASS blows up, the results so far still land in the .md
    try
        for n in n_values
            emit("\n" * "-"^50 * "\n")
            emit("Benchmarking C(n x n)=$n, A(n x n)=$n, B(n x n)=$n\n")
            gemm_flops = flops_gemm(n, n, n)

            for (name, precisions) in test_scenarios
                T_out = precisions[end]
                alpha, beta = -1.0, 1.0

                A_cpu = randn(T_out, n, n) .* 0.001f0
                B_cpu = randn(T_out, n, n) .* 0.001f0

                # Make strictly diagonally dominant but safe from Float16 overflow (max 65504)
                A_cpu .+= Diagonal(fill(T_out(100.0), n))
                B_cpu .+= Diagonal(fill(T_out(100.0), n))

                d_A = CuArray(A_cpu)
                d_B = CuArray(B_cpu)
                d_C_orig = CuArray(zeros(T_out, n, n))

                # Ground truth calculation
                d_A_fp64 = CuArray{Float64}(d_A)
                d_B_fp64 = CuArray{Float64}(d_B)
                d_C_ground_truth = CuArray(zeros(Float64, n, n))
                CUBLAS.gemm!('N', 'N', Float64(alpha), d_A_fp64, d_B_fp64, Float64(beta), d_C_ground_truth)

                # --- Simplified Logic: Call the correct function based on the test case ---
                C_for_custom = copy(d_C_orig)
                C_custom_result = if name in ["Pure F16", "Pure F32", "Pure F64"]
                    # For pure precision tests, call the standard recursive function
                    alpha = T_out(alpha)
                    beta = T_out(beta)
                    # Fallback to direct gemm for pure arrays, or wrap in FullMixedPrec with same type
                    C_mixed = FullMixedPrec(C_for_custom; precisions=precisions)
                    recgemm!(alpha, d_A, d_B, beta, C_mixed)
                    reconstruct_matrix(C_mixed)
                else
                    # For mixed precision, use the FullMixedPrec structure
                    alpha = T_out(alpha)
                    beta = T_out(beta)
                    C_mixed = FullMixedPrec(C_for_custom; precisions=precisions)
                    recgemm!(alpha, d_A, d_B, beta, C_mixed)
                    reconstruct_matrix(C_mixed)
                end

                error_norm = norm(CuArray{Float64}(C_custom_result) - d_C_ground_truth)
                solution_norm = norm(d_C_ground_truth)
                relative_error = max(error_norm / solution_norm, 1e-20)

                push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

                # Performance test
                backend = KernelAbstractions.get_backend(d_A)
                time_ns = run_manual_benchmark(backend) do
                    alpha = T_out(alpha)
                    beta = T_out(beta)
                    C_perf_mixed = FullMixedPrec(copy(d_C_orig); precisions=precisions)
                    CUDA.@sync recgemm!(alpha, d_A, d_B, beta, C_perf_mixed)
                end
                runtime_ms = time_ns / 1_000_000
                gflops = calculate_gflops(gemm_flops, time_ns)
                push!(runtime_results[name], runtime_ms)
                record!(n, name, runtime_ms, gflops; rel_err = relative_error)

                emit(@sprintf("  %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms | %10.1f GFLOP/s\n",
                              name, relative_error, runtime_ms, gflops))
            end

            emit("\n--- Benchmarking standard CUBLAS.gemm! ---\n")

            for (name, T_prec) in sort!(collect(Dict("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)); by = first)
                alpha, beta = T_prec(-1.0), T_prec(1.0)

                A_cpu = randn(T_prec, n, n) .* 0.001f0
                B_cpu = randn(T_prec, n, n) .* 0.001f0

                # Make strictly diagonally dominant (consistent with above)
                A_cpu .+= Diagonal(fill(T_prec(100.0), n))
                B_cpu .+= Diagonal(fill(T_prec(100.0), n))

                d_A_cublas = CuArray(A_cpu)
                d_B_cublas = CuArray(B_cpu)
                d_C_cublas = CuArray(zeros(T_prec, n, n))

                backend = KernelAbstractions.get_backend(d_A_cublas)
                time_ns = run_manual_benchmark(backend) do
                    CUBLAS.gemm!('N', 'N', alpha, d_A_cublas, d_B_cublas, beta, d_C_cublas)
                end
                runtime_ms = time_ns / 1_000_000
                gflops = calculate_gflops(gemm_flops, time_ns)
                push!(cublas_runtime_results[name], runtime_ms)
                record!(n, name, runtime_ms, gflops)
                emit(@sprintf("  %-22s | Runtime: %8.3f ms | %10.1f GFLOP/s\n", name, runtime_ms, gflops))
            end

            emit("\n--- Benchmarking CUTLASS (cutlass_profiler) ---\n")

            if isempty(CUTLASS_PROFILER)
                emit("  [skipped] cutlass_profiler not found; set ENV[\"CUTLASS_PROFILER\"] or put it on PATH\n")
            else
                # The profiler is a separate process and needs its own 3 n^2 buffers: hand back Julia's cached GPU memory first.
                GC.gc(); CUDA.reclaim()
                for (name, T_str) in CUTLASS_TYPES
                    try
                        runtime_ms, kernel = cutlass_gemm_ms(CUTLASS_PROFILER, n, T_str)
                        gflops = calculate_gflops(gemm_flops, runtime_ms * 1e6)
                        record!(n, name, runtime_ms, gflops; note = kernel)
                        emit(@sprintf("  %-22s | Runtime: %8.3f ms | %10.1f GFLOP/s | %s\n", name, runtime_ms, gflops, kernel))
                    catch e
                        emit(@sprintf("  %-22s | FAILED: %s\n", name, first(split(sprint(showerror, e), '\n'))))
                    end
                end
            end
        end

        # --- Simplified Plotting Logic ---
        emit("\n" * "="^60 * "\n")
        emit("📊 Generating and saving plots...\n")

        acc_plot = plot(title="Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2)
        perf_plot = plot(title="Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10)

        for (name, acc_values) in accuracy_results
            if name != "Pure Float64"
                plot!(acc_plot, n_values, acc_values, label=name, marker=:auto)
            end
        end

        for (name, runtimes) in runtime_results
            plot!(perf_plot, n_values, runtimes, label=name, marker=:auto)
        end

        for (name, runtimes) in cublas_runtime_results
            plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=:dash, linewidth=2)
        end

        # CUTLASS runtimes (only sizes that succeeded)
        for (name, _) in CUTLASS_TYPES
            rs = filter(r -> r.impl == name, records)
            isempty(rs) || plot!(perf_plot, [r.n for r in rs], [r.runtime_ms for r in rs],
                                 label=name, marker=:auto, linestyle=:dot, linewidth=2)
        end

        savefig(acc_plot, "recgemm_accuracy.png")
        savefig(perf_plot, "recgemm_performance.png")

        emit("✅ Benchmark complete. Plots saved to disk.\n")
        emit("="^60 * "\n")
    finally
        md_path = write_results_markdown(records, n_values, String(take!(copy(log_io))))
        println("📝 Results written to $md_path")
    end
end

run_recgemm_benchmark()
