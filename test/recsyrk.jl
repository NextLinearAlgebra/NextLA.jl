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
    # Define the matrix sizes to test
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # Define the different mixed-precision scenarios
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
    
    # Simplified dictionaries to store results
    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))
    
    cublas_runtime_results = Dict(
        "CUBLAS F32" => Float64[],
        "CUBLAS F64" => Float64[]
    )

    println("🚀 Starting recsyrk! Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking C(n x n)=$n, A(n x n)=$n")
        
        for (name, precisions) in test_scenarios
            T_out = precisions[end]
            alpha, beta = -1.0, 1.0

            d_A = CuArray(randn(T_out, n, n) .* 0.1f0)
            d_C_orig = CuArray(zeros(T_out, n, n))

            # Ground truth calculation
            d_A_fp64 = CuArray{Float64}(d_A)
            d_C_ground_truth = CuArray(zeros(Float64, n, n))
            CUBLAS.syrk!('L', 'N', Float64(alpha), d_A_fp64, Float64(beta), d_C_ground_truth)

            # --- Simplified Logic: Call the correct function based on the test case ---
            C_for_custom = copy(d_C_orig)
            C_custom_result = if name in ["Pure F16", "Pure F32", "Pure F64"]
                # For pure precision tests, call the standard recursive function
                alpha = T_out(alpha)
                beta = T_out(beta)
                recsyrk!(alpha, d_A, beta, C_for_custom, 256)
                C_for_custom
            else
                # For mixed precision, use the SymmMixedPrec structure
                alpha = T_out(alpha)
                beta = T_out(beta)
                C_mixed = SymmMixedPrec(C_for_custom, 'L'; precisions=precisions)
                recsyrk!(alpha, d_A, beta, C_mixed)
                reconstruct_matrix(C_mixed)
            end

            error_norm = norm(tril(CuArray{Float64}(C_custom_result)) - tril(d_C_ground_truth))
            solution_norm = norm(tril(d_C_ground_truth))
            relative_error = max(error_norm / solution_norm, 1e-20)
            
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

            # Performance test
            backend = KernelAbstractions.get_backend(d_A)
            time_ns = run_manual_benchmark(backend) do
                alpha = T_out(alpha)
                beta = T_out(beta)
                if name in ["Pure F16", "Pure F32", "Pure F64"]
                    C_perf = copy(d_C_orig)
                    recsyrk!(alpha, d_A, beta, C_perf, 256)
                else
                    C_perf_mixed = SymmMixedPrec(copy(d_C_orig), 'L'; precisions=precisions)
                    recsyrk!(alpha, d_A, beta, C_perf_mixed)
                end
            end
            runtime_ms = time_ns / 1_000_000
            push!(runtime_results[name], runtime_ms)

            @printf("  %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
        end

        println("\n--- Benchmarking standard CUBLAS.syrk! ---")
        
        for (name, T_prec) in Dict("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)
            alpha, beta = T_prec(-1.0), T_prec(1.0)
            d_A_cublas = CuArray(randn(T_prec, n, n))
            d_C_cublas = CuArray(zeros(T_prec, n, n))
            
            backend = KernelAbstractions.get_backend(d_A_cublas)
            time_ns = run_manual_benchmark(backend) do
                CUBLAS.syrk!('L', 'N', alpha, d_A_cublas, beta, d_C_cublas)
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
    
    savefig(acc_plot, "recsyrk_accuracy.png")
    savefig(perf_plot, "recsyrk_performance.png")

    println("✅ Benchmark complete. Plots saved to disk.")
    println("="^60)
end

run_recsyrk_benchmark()