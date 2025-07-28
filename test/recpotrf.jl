using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl") 


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


function run_cholesky_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # 1. Restructure scenarios for easier iteration
    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
    )
    mixed_scenarios_base = Dict(
        "[F32, F64, F64, F64]" => [Float32, Float64, Float64, Float64],
        "[F32, F32, F64]" => [Float32, Float32, Float64],
        "[F32, F64, F64]" => [Float32, Float64, Float64],
        "[F16, F32, F32]" => [Float16, Float32, Float32],
        "[F16, F32, F64]" => [Float16, Float32, Float64],
        "[F32, F64]"      => [Float32, Float64],
        "[F16, F64]"      => [Float16, Float64],
    )
    # Recreate the full scenario list to initialize result dictionaries correctly
    all_scenarios = copy(pure_scenarios)
    for (name, prec) in mixed_scenarios_base
        all_scenarios[name] = prec
        all_scenarios[name * " (No T)"] = prec
    end

    accuracy_results = Dict(name => Float64[] for name in keys(all_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(all_scenarios))
    cusolver_runtime_results = Dict(
        "CUSOLVER F32" => Float64[],
        "CUSOLVER F64" => Float64[]
    )

    println("🚀 Starting Cholesky Benchmark...")

    for n in n_values
        println("\n" * "="^70)
        println("Benchmarking Matrix Size (n x n) = $n x $n")
        
        A_cpu = randn(Float64, n, n)
        A_spd_fp64 = CuArray(A_cpu * A_cpu' + n * I)
        
        A_ground_truth = copy(A_spd_fp64)
        CUSOLVER.potrf!('L', A_ground_truth)
        L_truth = tril(A_ground_truth)
        
        backend = KernelAbstractions.get_backend(A_spd_fp64)

        # 2. Handle pure scenarios first
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[1] 
            A_pure_input = T_prec.(A_spd_fp64)
            potrf_recursive!(A_pure_input, 4096) 
            L_result = tril(A_pure_input)

            A_perf_template = T_prec.(A_spd_fp64) 
            time_ns = run_manual_benchmark(backend) do
                A_to_factor = copy(A_perf_template) 
                potrf_recursive!(A_to_factor, 4096)
            end
            runtime_ms = time_ns / 1_000_000
            
            error_norm = norm(L_result - L_truth)
            relative_error = max(error_norm / norm(L_truth), 1e-20)
            
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))
            push!(runtime_results[name], runtime_ms)
            @printf("    %-25s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
        end

        # 3. Loop through base mixed scenarios and run both versions
        println("\n--- Mixed Precision (Transpose vs. No Transpose) ---")
        @printf("    %-25s | T Runtime | No trans Runtime | Speedup\n", "Scenario")
        println("    " * "-"^65)

        for (name, precisions) in mixed_scenarios_base
            # --- Run Transpose Version ---
            A_mixed_T = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
            potrf_recursive!(A_mixed_T)
            error_T = max(norm(tril(reconstruct_matrix(A_mixed_T)) - L_truth) / norm(L_truth), 1e-20)
            time_ns_T = run_manual_benchmark(backend) do
                A_perf_mixed = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
                potrf_recursive!(A_perf_mixed)
            end
            runtime_T = time_ns_T / 1_000_000

            # --- Run No-Transpose Version ---
            A_mixed_noT = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
            potrf_recursive_T!(A_mixed_noT)
            error_no_T = max(norm(tril(reconstruct_matrix(A_mixed_noT)) - L_truth) / norm(L_truth), 1e-20)
            time_ns_noT = run_manual_benchmark(backend) do
                A_perf_mixed = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
                potrf_recursive_T!(A_perf_mixed)
            end
            runtime_no_T = time_ns_noT / 1_000_000

            # --- Store results for plotting ---
            push!(runtime_results[name], runtime_T)
            push!(runtime_results[name * " (No T)"], runtime_no_T)
            push!(accuracy_results[name], -log10(max(error_T, 1e-18)))
            push!(accuracy_results[name * " (No T)"], -log10(max(error_no_T, 1e-18)))

            # --- Print side-by-side comparison ---
            speedup = runtime_T > 0 && runtime_no_T > 0 ? runtime_T / runtime_no_T : 0.0
            @printf("    %-25s | %9.3fms | %12.3fms | %7.2fx\n", name, runtime_T, runtime_no_T, speedup)
        end
        
        # --- CUSOLVER benchmarks ---
        println("\n--- Standard CUSOLVER.potrf! ---")
        for (name, T_prec) in Dict("CUSOLVER F32" => Float32, "CUSOLVER F64" => Float64)
            A_spd_base = CuArray(T_prec.(A_cpu * A_cpu' + n * I))
            time_ns = run_manual_benchmark(backend) do
                CUSOLVER.potrf!('L', copy(A_spd_base))
            end
            runtime_ms = time_ns / 1_000_000
            push!(cusolver_runtime_results[name], runtime_ms)
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
    end
    
    # ... Plotting code remains unchanged ...
    println("\n" * "="^70)
    println("📊 Generating and saving plots...")

    acc_plot = plot(title="Cholesky Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2)
    for (name, acc_values) in accuracy_results
        if name != "Pure F64" 
            plot!(acc_plot, n_values, acc_values, label=name, marker=:auto)
        end
    end
    savefig(acc_plot, "cholesky_accuracy.png")

    perf_plot = plot(title="Cholesky Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10)
    for (name, runtimes) in runtime_results
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto)
    end
    for (name, runtimes) in cusolver_runtime_results
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=:dash, linewidth=2)
    end
    savefig(perf_plot, "cholesky_performance.png")

    println("✅ Benchmark complete. Plots saved to disk.")
    println("="^70)
end

run_cholesky_benchmark()