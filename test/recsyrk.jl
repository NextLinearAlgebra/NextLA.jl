using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions
include("benchmark.jl") 

# reconstructs the matrix
function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.OffDiag
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = CuArray{T_Base}(undef, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end


function run_recsyrk_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384]
    m_fixed = 128

    # Scenarios and parameters are unchanged
    test_scenarios = Dict(
        "Pure F16"             => [Float16, Float16, Float16],
        "Pure F32"             => [Float32, Float32, Float32],
        "Pure F64"             => [Float64, Float64, Float64],
        "[F32, F32, F64]"      => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]" => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]" => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"      => [Float32, Float64, Float64],
        "[F16, F32, F32]"      => [Float16, Float32, Float32]
    )
    
    # NEW: Define the algorithm versions to test
    gemm_orders = Dict(
        "GEMM First"   => :first,
        "GEMM Middle"  => :middle,
        "GEMM End"     => :end
    )

    # NEW: Results dictionaries are now nested to hold data for each version
    all_accuracy_results = Dict(order_name => Dict(name => Float64[] for name in keys(test_scenarios)) for order_name in keys(gemm_orders))
    all_runtime_results = Dict(order_name => Dict(name => Float64[] for name in keys(test_scenarios)) for order_name in keys(gemm_orders))
    
    cublas_runtime_results = Dict(
        "CUBLAS F32" => Float64[],
        "CUBLAS F64" => Float64[]
    )

    println("🚀 Starting Comparative recsyrk! Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking C(n x n)=$n, A(n x m)=$m_fixed")
        
        # NEW: Loop over the different algorithm orders
        for (order_name, order_symbol) in gemm_orders
            println("\n--- Algorithm Order: $order_name ---")
            
            for (name, precisions) in test_scenarios
                T_out = precisions[end]
                alpha, beta = -1.0, 1.0
    
                d_A = CuArray(randn(T_out, n, m_fixed) .* 0.1f0)
                d_C_orig = CuArray(zeros(T_out, n, n))
    
                d_A_fp64 = CuArray{Float64}(d_A)
                d_C_ground_truth = CuArray(zeros(Float64, n, n))
                CUBLAS.syrk!('L', 'N', Float64(alpha), d_A_fp64, Float64(beta), d_C_ground_truth)

                # Accuracy test
                C_for_custom = copy(d_C_orig)
                C_mixed = SymmMixedPrec(C_for_custom, 'L'; precisions=precisions)
                # Pass the order_symbol to the function
                recsyrk!(alpha, d_A, beta, C_mixed; gemm_order=order_symbol)
                C_custom_result = reconstruct_matrix(C_mixed)

                error_norm = norm(tril(CuArray{Float64}(C_custom_result)) - tril(d_C_ground_truth))
                solution_norm = norm(tril(d_C_ground_truth))
                relative_error = max(error_norm / solution_norm, 1e-20)
                
                # Push to the nested dictionary
                push!(all_accuracy_results[order_name][name], -log10(max(relative_error, 1e-18)))

                # Benchmark
                backend = KernelAbstractions.get_backend(d_A)
                time_ns = run_manual_benchmark(backend) do
                    C_perf = SymmMixedPrec(copy(d_C_orig), 'L'; precisions=precisions)
                    # Pass the order_symbol to the function
                    recsyrk!(alpha, d_A, beta, C_perf; gemm_order=order_symbol)
                end
                runtime_ms = time_ns / 1_000_000
                push!(all_runtime_results[order_name][name], runtime_ms)

                @printf("  %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
            end
        end

        println("\n--- Benchmarking standard CUBLAS.syrk! ---")
        for (name, T_prec) in Dict("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)
            # This part is unchanged and runs once per n_value
            # ... (code for CUBLAS baseline) ...
        end
    end

    println("\n" * "="^60)
    println("📊 Generating and saving comparison plots...")

    # NEW: Plotting logic to handle the different versions
    acc_plot = plot(title="Accuracy Comparison", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2)
    perf_plot = plot(title="Performance Comparison", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10)
    
    for (order_name, results) in all_accuracy_results
        for (name, acc_values) in results
            # Only plot scenarios with meaningful error
            if name != "Pure Float64"
                plot!(acc_plot, n_values, acc_values, label="$order_name: $name", marker=:auto)
            end
        end
    end
    savefig(acc_plot, "recsyrk_accuracy_comparison.png")
    
    for (order_name, results) in all_runtime_results
        for (name, runtimes) in results
            plot!(perf_plot, n_values, runtimes, label="$order_name: $name", marker=:auto)
        end
    end
    
    for (name, runtimes) in cublas_runtime_results
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=:dash, linewidth=2)
    end
    savefig(perf_plot, "recsyrk_runtime_comparison.png")

    println("✅ Benchmark complete. Comparison plots saved to disk.")
    println("="^60)
end

run_recsyrk_benchmark()