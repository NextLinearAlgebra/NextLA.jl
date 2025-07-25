using Test
using CUDA
using LinearAlgebra
using Printf
using Plots
using KernelAbstractions

include("benchmark.jl")

function run_potrf_component_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    # block_size = 

    test_scenarios = Dict(
        "A both trsm and trmm"   => potrf_recursive_A!,
        "B only syrk"  => potrf_recursive_B!,
        "C only trsm"  => potrf_recursive_C!,
        "D neither syrk or trsm" => potrf_recursive_D!,
    )

    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))
    cusolver_runtime_results = Float64[]

    for n in n_values
        block_size = 4096
        @printf("\n--- Matrix Size n = %d ---\n", n)
        
        A_cpu = randn(Float64, n, n)
        A_spd_fp64_pristine = CuArray(A_cpu * A_cpu' + n * I)

        A_for_truth = copy(A_spd_fp64_pristine)
        CUSOLVER.potrf!('L', A_for_truth)
        L_truth = tril(A_for_truth) 

        for (name, potrf_func!) in test_scenarios
            
            backend = KernelAbstractions.get_backend(A_spd_fp64_pristine)
            time_ns = run_manual_benchmark(backend) do
                A_to_factor_inner = copy(A_spd_fp64_pristine)
                potrf_func!(A_to_factor_inner, block_size)
            end
            runtime_ms = time_ns / 1_000_000
            push!(runtime_results[name], runtime_ms)

            A_final_result = copy(A_spd_fp64_pristine)
            potrf_func!(A_final_result, block_size)
            
            L_result = tril(A_final_result) 
            error_norm = norm(L_result - L_truth)
            solution_norm = norm(L_truth)

            relative_error = max(error_norm / solution_norm, 1e-20)
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

            @printf("   %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
        end

        backend = KernelAbstractions.get_backend(A_spd_fp64_pristine)
        time_ns = run_manual_benchmark(backend) do
            A_to_factor_cusolver = copy(A_spd_fp64_pristine)
            CUSOLVER.potrf!('L', A_to_factor_cusolver)
        end
        runtime_ms = time_ns / 1_000_000
        push!(cusolver_runtime_results, runtime_ms)
        @printf("   %-22s | Runtime: %8.3f ms\n", "CUSOLVER F64", runtime_ms)
    end

    println("\n" * "="^60)
    println("📊 Generating and saving plots...")

    acc_plot = plot(title="Cholesky Component Accuracy (Float64)",
                      xlabel="Matrix Size (n)",
                      ylabel="-log10(Relative Error)",
                      legend=:outertopright,
                      xaxis=:log2)
    for (name, acc_values) in accuracy_results
        plot!(acc_plot, n_values, acc_values, label=name, marker=:auto)
    end
    savefig(acc_plot, "potrf_component_accuracy.png")

    perf_plot = plot(title="Cholesky Component Performance (Float64)",
                       xlabel="Matrix Size (n)",
                       ylabel="Runtime (ms)",
                       legend=:outertopright,
                       xaxis=:log2,
                       yaxis=:log10) # Log scale for y-axis is best for performance plots
    for (name, runtimes) in runtime_results
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto)
    end

    plot!(perf_plot, n_values, cusolver_runtime_results, label="CUSOLVER F64", marker=:auto, linestyle=:dash, linewidth=2)
    savefig(perf_plot, "potrf_component_performance.png")

    println("✅ Benchmark complete. Plots saved to 'potrf_component_accuracy.png' and 'potrf_component_performance.png'.")
    println("="^60)
end

run_potrf_component_benchmark()