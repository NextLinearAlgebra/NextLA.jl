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
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768] 
    
    test_scenarios = Dict(
        "Pure F32"             => [Float32],
        "Pure F64"             => [Float64],
        "[F32, F32, F64]"      => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]" => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]" => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"      => [Float32, Float64, Float64],
        "[F16, F32, F32]"      => [Float16, Float32, Float32]
    )
    
    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))
    cusolver_runtime_results = Dict(
        "CUSOLVER F32" => Float64[],
        "CUSOLVER F64" => Float64[]
    )

    println("🚀 Starting Cholesky Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking Matrix Size (n x n) = $n x $n")
        
        A_cpu = randn(Float64, n, n)
        A_spd_fp64 = CuArray(A_cpu * A_cpu' + n * I)
        
        A_ground_truth = copy(A_spd_fp64)
        CUSOLVER.potrf!('L', A_ground_truth)
        L_truth = tril(A_ground_truth)
        
        backend = KernelAbstractions.get_backend(A_spd_fp64)

        for (name, precisions) in test_scenarios
            local L_result, runtime_ms 

            if name == "Pure F32" || name == "Pure F64"
                T_prec = precisions[1] 
                
                A_pure_input = T_prec.(A_spd_fp64)
                potrf_recursive_D!(A_pure_input, 256) 
                L_result = tril(A_pure_input)

                A_perf_template = T_prec.(A_spd_fp64) 
                time_ns = run_manual_benchmark(backend) do
                    A_to_factor = copy(A_perf_template) 
                    potrf_recursive_D!(A_to_factor, 256)
                end
                runtime_ms = time_ns / 1_000_000
            else
                
                A_mixed_input = copy(A_spd_fp64)
                A_mixed = SymmMixedPrec(A_mixed_input, 'L'; precisions=precisions)
                potrf_recursive!(A_mixed)
                A_reconstructed = reconstruct_matrix(A_mixed)
                L_result = tril(A_reconstructed)

                A_perf_template = copy(A_spd_fp64)
                time_ns = run_manual_benchmark(backend) do
                    A_to_factor_input = copy(A_perf_template)
                    A_perf_mixed = SymmMixedPrec(A_to_factor_input, 'L'; precisions=precisions)
                    potrf_recursive!(A_perf_mixed) 
                end
                runtime_ms = time_ns / 1_000_000
            end

            error_norm = norm(L_result - L_truth)
            solution_norm = norm(L_truth)
            relative_error = max(error_norm / solution_norm, 1e-20)
            
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))
            push!(runtime_results[name], runtime_ms)

            @printf("   %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
        end

        
        println("\n--- Benchmarking standard CUSOLVER.potrf! ---")
        for (name, T_prec) in Dict("CUSOLVER F32" => Float32, "CUSOLVER F64" => Float64)
            A_cpu_base = randn(T_prec, n, n)
            A_spd_base = CuArray(A_cpu_base * A_cpu_base' + n * I)
            
            time_ns = run_manual_benchmark(backend) do
                A_to_factor = copy(A_spd_base)
                CUSOLVER.potrf!('L', A_to_factor)
            end
            runtime_ms = time_ns / 1_000_000
            push!(cusolver_runtime_results[name], runtime_ms)
            @printf("   %-22s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
    end
    
    println("\n" * "="^60)
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
    println("="^60)
end

run_cholesky_benchmark()