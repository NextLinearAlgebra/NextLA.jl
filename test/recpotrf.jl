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

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
    )
    mixed_scenarios = Dict(
        "[F32, F64, F64, F64]" => [Float32, Float64, Float64, Float64],
        "[F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "[F32, F32, F64]" => [Float32, Float32, Float64],
        "[F32, F64, F64]" => [Float32, Float64, Float64],
        "[F16, F32, F32]" => [Float16, Float32, Float32],
        "[F16, F16, F32]" => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F64]" => [Float16, Float32, Float64],
        "[F32, F64]"      => [Float32, Float64],
        "[F16, F64]"      => [Float16, Float64],
        "[F16, F32]"      => [Float16, Float32],
    )
    
    all_scenarios = merge(pure_scenarios, mixed_scenarios)

    accuracy_results = Dict(name => Float64[] for name in keys(all_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(all_scenarios))
    cusolver_runtime_results = Dict(
        "CUSOLVER F32" => Float64[],
        "CUSOLVER F64" => Float64[]
    )

    println("🚀 Starting Cholesky Benchmark...")

    for n in n_values
        println("\n" * "="^80)
        println("Benchmarking Matrix Size (n x n) = $n x $n")
        
        A_cpu = randn(Float64, n, n)
        A_spd_cpu = A_cpu * A_cpu' + (n*100) * I

        local L_truth_cpu, L_truth_norm, backend
        let A_gpu = CuArray(A_spd_cpu)
            backend = get_backend(A_gpu)
            CUSOLVER.potrf!('L', A_gpu)
            L_truth_norm = norm(A_gpu)
            L_truth_cpu = collect(tril(A_gpu))
        end
        GC.gc(true); CUDA.reclaim()
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[1]
            try
                A_to_factor = CuArray{T_prec}(A_spd_cpu)
                time_ns = run_manual_benchmark(backend) do
                    copyto!(A_to_factor, CuArray{T_prec}(A_spd_cpu))
                    potrf_recursive!(A_to_factor, 4096)
                end
                runtime_ms = time_ns / 1_000_000
                
                L_result_cpu = collect(tril(A_to_factor))
                error_norm = norm(L_result_cpu - L_truth_cpu)
                relative_error = max(error_norm / L_truth_norm, 1e-20)
                
                push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))
                push!(runtime_results[name], runtime_ms)
                @printf("    %-28s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
            catch e
                if e isa CUDA.OutOfMemoryError
                    @printf("    %-28s | SKIPPED due to OutOfMemoryError\n", name)
                    push!(accuracy_results[name], NaN); push!(runtime_results[name], NaN)
                else rethrow(e) end
            end
            GC.gc(true); CUDA.reclaim()
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            try
                A_to_factor_mixed = SymmMixedPrec(CuArray(A_spd_cpu), 'L'; precisions=precisions)
                time_ns = run_manual_benchmark(backend) do
                    A_to_factor_mixed = SymmMixedPrec(CuArray(A_spd_cpu), 'L'; precisions=precisions)
                    potrf_recursive!(A_to_factor_mixed)
                end
                runtime_ms = time_ns / 1_000_000
                
                L_result_reconstructed = reconstruct_matrix(A_to_factor_mixed)
                L_result_cpu = collect(tril(L_result_reconstructed))
                error_norm = norm(L_result_cpu - L_truth_cpu)
                relative_error = max(error_norm / L_truth_norm, 1e-20)

                push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))
                push!(runtime_results[name], runtime_ms)
                @printf("    %-28s | Rel. Error: %9.2e | Runtime: %8.3f ms\n", name, relative_error, runtime_ms)
            catch e
                if e isa CUDA.OutOfMemoryError
                    @printf("    %-28s | SKIPPED due to OutOfMemoryError\n", name)
                    push!(accuracy_results[name], NaN); push!(runtime_results[name], NaN)
                else rethrow(e) end
            end
            GC.gc(true); CUDA.reclaim()
        end
        
        println("\n--- Standard CUSOLVER.potrf! ---")
        for (name, T_prec) in Dict("CUSOLVER F32" => Float32, "CUSOLVER F64" => Float64)
            try
                A_spd_base = CuArray{T_prec}(A_spd_cpu)
                time_ns = run_manual_benchmark(backend) do
                    copyto!(A_spd_base, CuArray{T_prec}(A_spd_cpu))
                    CUSOLVER.potrf!('L', A_spd_base)
                end
                runtime_ms = time_ns / 1_000_000
                push!(cusolver_runtime_results[name], runtime_ms)
                @printf("    %-28s | Runtime: %8.3f ms\n", name, runtime_ms)
            catch e
                if e isa CUDA.OutOfMemoryError
                    @printf("    %-28s | SKIPPED due to OutOfMemoryError\n", name)
                    push!(cusolver_runtime_results[name], NaN)
                else rethrow(e) end
            end
            GC.gc(true); CUDA.reclaim()
        end
    end
    
    println("\n" * "="^80)
    println("📊 Generating and saving plots...")

    acc_plot = plot(title="Cholesky Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2, size=(800,600), dpi=300)
    for (name, acc_values) in filter(p -> !any(isnan, p.second), accuracy_results)
        if name != "Pure F64"
            plot!(acc_plot, n_values, acc_values, label=name, marker=:auto)
        end
    end
    savefig(acc_plot, "cholesky_accuracy.png")

    perf_plot = plot(title="Cholesky Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10, size=(800,600), dpi=300)
    for (name, runtimes) in filter(p -> !any(isnan, p.second), runtime_results)
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto)
    end
    for (name, runtimes) in filter(p -> !any(isnan, p.second), cusolver_runtime_results)
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=:dash, linewidth=2)
    end
    savefig(perf_plot, "cholesky_performance.png")

    println("✅ Benchmark complete. Plots saved to disk.")
    println("="^80)
end

run_cholesky_benchmark()