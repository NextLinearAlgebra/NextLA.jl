using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl") 


# recunstructs the matrix from the recursive SymmMixedPrec
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

#run benchmarks and generate plots
function run_recsyrk_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192] 
    m_fixed = 128 

    test_scenarios = Dict(
        "Pure Float32"      => [Float32, Float32, Float32],
        "Pure Float64"      => [Float64, Float64, Float64],
        "[F32, F32, F64]"   => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]" => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]" => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"   => [Float32, Float64, Float64]
    )

    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))
    
    println("🚀 Starting recsyrk! Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking C(n x n)=$n, A(n x m)=$m_fixed")
        for (name, precisions) in test_scenarios
            T_out = precisions[end]
            alpha, beta = -1.0, 1.0
    
            d_A = CuArray(randn(T_out, n, m_fixed))
            d_C_orig = CuArray(zeros(T_out, n, n))
    
            # we test accuracy against fp64
            d_A_fp64 = CuArray{Float64}(d_A)
            d_C_ground_truth = CuArray(zeros(Float64, n, n))
            CUBLAS.syrk!('L', 'N', Float64(alpha), d_A_fp64, Float64(beta), d_C_ground_truth)

            # recsyrk accuracy test
            C_for_custom = copy(d_C_orig)
            C_mixed = SymmMixedPrec(C_for_custom, 'L'; precisions=precisions)
            recsyrk!(alpha, d_A, beta, C_mixed)
            C_custom_result = reconstruct_matrix(C_mixed)

            error_norm = norm(tril(CuArray{Float64}(C_custom_result)) - tril(d_C_ground_truth))
            solution_norm = norm(tril(d_C_ground_truth))
            relative_error = max(error_norm / solution_norm, 1e-20)
            
            push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

            # benchmark
            backend = KernelAbstractions.get_backend(d_A)
            time_ns = run_manual_benchmark(backend) do
                C_perf = SymmMixedPrec(copy(d_C_orig), 'L'; precisions=precisions)
                recsyrk!(alpha, d_A, beta, C_perf)
            end
            runtime_ms = time_ns / 1_000_000
            push!(runtime_results[name], runtime_ms)

            @printf("  %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
        end
    end

    println("\n" * "="^60)
    println("📊 Generating and saving plots...")

    acc_plot = plot(title="Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error) [Higher is Better]", legend=:outertopright, xaxis=:log2)
    for name in keys(test_scenarios)
        if name != "Pure Float64"
            plot!(acc_plot, n_values, accuracy_results[name], label=name, marker=:auto)
        end
    end
    savefig(acc_plot, "3recsyrk_accuracy.png")

    perf_plot = plot(title="Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms) [Lower is Better]", legend=:outertopright, xaxis=:log2, yaxis=:log10)
    for name in keys(test_scenarios)
        plot!(perf_plot, n_values, runtime_results[name], label=name, marker=:auto)
    end
    savefig(perf_plot, "3recsyrk_runtime.png")

    println("✅ Benchmark complete. Plots saved to disk.")
    println("="^60)
end

run_recsyrk_benchmark()