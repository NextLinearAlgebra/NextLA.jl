using Test
using CUDA
using LinearAlgebra
using Printf
using Plots
using KernelAbstractions

include("benchmark.jl")

function run_block_size_benchmark()
    # --- Configuration ---
    n = 32768  # Set a fixed large matrix size
    block_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384] # Range of block sizes to test

    test_scenarios = Dict(
        "A (Nested TRSM & SYRK)" => potrf_recursive_A!
    )

    # --- Initialization ---
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))

    println("🚀 Starting Cholesky Block Size Benchmark...")
    println("Fixed Matrix Size (n x n) = $n x $n")

    # Create the pristine source matrix once
    A_cpu = randn(Float64, n, n)
    A_spd_pristine = CuArray(A_cpu * A_cpu' + n * I)
    backend = KernelAbstractions.get_backend(A_spd_pristine)

    # --- Main Benchmark Loop ---
    for block_size in block_sizes
        @printf("\n--- Testing Block Size = %d ---\n", block_size)

        for (name, potrf_func!) in test_scenarios
            # Benchmark the function with the current block_size
            time_ns = run_manual_benchmark(backend) do
                A_to_factor = copy(A_spd_pristine) # Reset matrix for each run
                potrf_func!(A_to_factor, block_size)
            end
            
            runtime_ms = time_ns / 1_000_000
            push!(runtime_results[name], runtime_ms)
            @printf("   %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
    end

    # --- CUSOLVER Baseline (measured once, as it's constant) ---
    println("\n--- Measuring CUSOLVER Baseline ---")
    time_ns = run_manual_benchmark(backend) do
        A_to_factor_cusolver = copy(A_spd_pristine)
        CUSOLVER.potrf!('L', A_to_factor_cusolver)
    end
    cusolver_runtime_ms = time_ns / 1_000_000
    @printf("   %-25s | Runtime: %8.3f ms\n", "CUSOLVER F64", cusolver_runtime_ms)

    # --- Plotting Results ---
    println("\n" * "="^60)
    println("📊 Generating and saving plot...")

    perf_plot = plot(
        title="Cholesky Performance vs. Block Size (n=$n)",
        xlabel="Block Size",
        ylabel="Runtime (ms)",
        legend=:outertopright,
        xaxis=:log2, # Log scale is great for power-of-2 block sizes
        yaxis=:identity, # Linear scale to easily see the minimum
        size=(800, 600)
    )

    # Plot results for each of your functions
    for (name, runtimes) in runtime_results
        plot!(perf_plot, block_sizes, runtimes, label=name, marker=:auto, linewidth=2)
    end

    # Plot the CUSOLVER baseline as a horizontal line for comparison
    hline!(
        perf_plot, 
        [cusolver_runtime_ms], 
        label="CUSOLVER F64", 
        linestyle=:dash, 
        linewidth=2,
        color=:black
    )

    savefig(perf_plot, "potrf_blocksize_performance.png")

    println("✅ Benchmark complete. Plot saved to 'potrf_blocksize_performance.png'.")
    println("="^60)
end

# Run the new benchmark function
run_block_size_benchmark()