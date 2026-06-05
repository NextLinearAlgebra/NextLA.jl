using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions
include("benchmark.jl")

function run_recgemm_benchmark()
    # Define the matrix sizes to test
    n_values = [2048, 4096, 8192, 16384, 32768, 65536] #16, 32, 256, 512, 1024, 

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
        "[F16, F16, F32, F64]" => [Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],

        # Extreme Jumps (F16 direct to F64)
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

    println("🚀 Starting recgemm! Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking C(n x n)=$n, A(n x n)=$n, B(n x n)=$n")

        for (name, precisions) in test_scenarios
            T_out = precisions[end]
            alpha, beta = -1.0, 1.0

            d_A = CuArray(randn(T_out, n, n) .* 0.1f0)
            d_B = CuArray(randn(T_out, n, n) .* 0.1f0)
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
            push!(runtime_results[name], runtime_ms)

            @printf("  %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
        end

        println("\n--- Benchmarking standard CUBLAS.gemm! ---")

        for (name, T_prec) in Dict("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)
            alpha, beta = T_prec(-1.0), T_prec(1.0)
            d_A_cublas = CuArray(randn(T_prec, n, n))
            d_B_cublas = CuArray(randn(T_prec, n, n))
            d_C_cublas = CuArray(zeros(T_prec, n, n))

            backend = KernelAbstractions.get_backend(d_A_cublas)
            time_ns = run_manual_benchmark(backend) do
                CUBLAS.gemm!('N', 'N', alpha, d_A_cublas, d_B_cublas, beta, d_C_cublas)
            end
            runtime_ms = time_ns / 1_000_000
            push!(cublas_runtime_results[name], runtime_ms)
            @printf("  %-22s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
    end

    # --- Simplified Plotting Logic ---
    println("\n" * "="^60)
    println("📊 Generating and saving plots...")

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

    savefig(acc_plot, "recgemm_accuracy.png")
    savefig(perf_plot, "recgemm_performance.png")

    println("✅ Benchmark complete. Plots saved to disk.")
    println("="^60)
end

run_recgemm_benchmark()
