using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER
using KernelAbstractions
using Plots
using LinearAlgebra

function run_cholesky_benchmarks()
    sizes = [512, 1024, 2048, 4096, 8192]
    block_size = 256 
    
    # --- CHANGE 1: Add scenarios for the CUSOLVER library baseline ---
    # I've also renamed your existing scenarios for clarity on the plot.
    test_scenarios = Dict(
        "Recursive (Float64)"      => Float64,
        "Recursive (Float32)"      => Float32,
        "CUSOLVER Library (F64)" => Float64,
        "CUSOLVER Library (F32)" => Float32
    )

    # Data structures to store the results
    all_runtimes = Dict(name => Float64[] for name in keys(test_scenarios))
    all_errors = Dict(name => Float64[] for name in keys(test_scenarios))

    println("======================================================")
    println("  Starting Cholesky Benchmark...")
    println("======================================================")

    for n in sizes
        println("\n--- Testing Matrix Size: $n x $n ---")

        for (name, T) in test_scenarios
            # --- Setup ---
            A_cpu = randn(T, n, n)
            A_gpu_orig = CuArray(A_cpu * A_cpu' + n * I)
            
            # --- Accuracy Test ---
            A_to_factor = copy(A_gpu_orig)

            # --- CHANGE 2: Use an if/else block to call the correct function ---
            if startswith(name, "Recursive")
                potrf_recursive!(A_to_factor, block_size)
            else # CUSOLVER Library
                CUSOLVER.potrf!('L', A_to_factor)
            end
            
            L = tril(A_to_factor)
            A_reconstructed = L * L'
            
            residual = norm(A_gpu_orig - A_reconstructed)
            rel_error = residual / norm(A_gpu_orig)
            push!(all_errors[name], rel_error)
            
            # --- Performance Test ---
            A_to_factor_perf = copy(A_gpu_orig) 
            backend = get_backend(A_to_factor_perf)

            runtime_ns = run_manual_benchmark(backend) do
                copyto!(A_to_factor_perf, A_gpu_orig)
                
                # --- CHANGE 3: Also call the correct function inside the benchmark ---
                if startswith(name, "Recursive")
                    potrf_recursive!(A_to_factor_perf, block_size)
                else # CUSOLVER Library
                    CUSOLVER.potrf!('L', A_to_factor_perf)
                end
            end
            
            runtime_ms = runtime_ns / 1_000_000
            push!(all_runtimes[name], runtime_ms)
            
            println("  Scenario: $name \t| Relative Error: $rel_error \t| Runtime: $runtime_ms ms")
        end
    end
    
    return sizes, all_runtimes, all_errors
end

function plot_results(sizes, runtimes, errors)
    runtime_plot = plot(
        title="Recursive Cholesky Performance",
        xlabel="Matrix Size (n)",
        ylabel="Runtime (ms)",
        xaxis=:log2, yaxis=:log10,
        legend=:topleft,
        fontfamily="Computer Modern",
        dpi=300
    )
    for (name, data) in runtimes
        plot!(runtime_plot, sizes, data, label=name, marker=:circle, lw=2)
    end
    savefig(runtime_plot, "cholesky_runtime_benchmark.png")
    println("\nSaved runtime plot to cholesky_runtime_benchmark.png")

    accuracy_plot = plot(
        title="Recursive Cholesky Accuracy",
        xlabel="Matrix Size (n)",
        ylabel="Relative Error",
        xaxis=:log2, yaxis=:log10,
        legend=:topleft,
        fontfamily="Computer Modern",
        dpi=300
    )
    for (name, data) in errors
        plot!(accuracy_plot, sizes, data, label=name, marker=:circle, lw=2)
    end
    savefig(accuracy_plot, "cholesky_accuracy_benchmark.png")
    println("Saved accuracy plot to cholesky_accuracy_benchmark.png")
end

sizes, runtimes, errors = run_cholesky_benchmarks()
plot_results(sizes, runtimes, errors)