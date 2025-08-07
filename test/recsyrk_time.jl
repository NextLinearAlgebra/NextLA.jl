using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl")

function get_syrk_runtime_pure(d_A, d_C_orig, T_prec, alpha, beta)
    backend = KernelAbstractions.get_backend(d_A)
    time_ns = run_manual_benchmark(backend) do
        C_perf = copy(d_C_orig)
        recsyrk!(T_prec(alpha), d_A, T_prec(beta), C_perf, 256)
    end
    return time_ns / 1_000_000
end

function get_syrk_runtime_mixed(d_A, d_C_orig, T_out, precisions, alpha, beta)
    backend = KernelAbstractions.get_backend(d_A)
    time_ns = run_manual_benchmark(backend) do
        C_perf_mixed = SymmMixedPrec(copy(d_C_orig), 'L'; precisions=precisions)
        recsyrk!(T_out(alpha), d_A, T_out(beta), C_perf_mixed)
    end
    return time_ns / 1_000_000
end

function get_syrk_runtime_cublas(n::Int, T_prec, alpha, beta)
    d_A_cublas = CuArray(randn(T_prec, n, n))
    d_C_cublas = CuArray(zeros(T_prec, n, n))
    
    backend = KernelAbstractions.get_backend(d_A_cublas)
    time_ns = run_manual_benchmark(backend) do
        CUBLAS.syrk!('L', 'N', T_prec(alpha), d_A_cublas, T_prec(beta), d_C_cublas)
    end
    return time_ns / 1_000_000
end



function run_recsyrk_performance_benchmark()
    n_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    pure_scenarios = Dict(
        "Pure F16" => [Float16, Float16, Float16],
        "Pure F32" => [Float32, Float32, Float32],
        "Pure F64" => [Float64, Float64, Float64],
    )
    mixed_scenarios = Dict(
        "[F32, F32, F64]"           => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]"      => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]"      => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"           => [Float32, Float64, Float64],
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F32]"           => [Float16, Float32, Float32]
    )

    runtime_results = Dict(name => Float64[] for name in union(keys(pure_scenarios), keys(mixed_scenarios)))
    cublas_runtime_results = Dict("CUBLAS F32" => Float64[], "CUBLAS F64" => Float64[])

    println("🚀 Starting recsyrk! Performance Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking Performance for C(n x n)=$n, A(n x n)=$n")
        alpha, beta = -1.0, 1.0
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[end]
            d_A = CuArray(randn(T_prec, n, n) .* 0.1f0)
            d_C_orig = CuArray(zeros(T_prec, n, n))
            runtime_ms = get_syrk_runtime_pure(d_A, d_C_orig, T_prec, alpha, beta)
            push!(runtime_results[name], runtime_ms)
            @printf("    %-28s | Runtime: %8.3f ms\n", name, runtime_ms)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            T_out = precisions[end]
            d_A = CuArray(randn(T_out, n, n) .* 0.1f0)
            d_C_orig = CuArray(zeros(T_out, n, n))
            
            runtime_ms = get_syrk_runtime_mixed(d_A, d_C_orig, T_out, precisions, alpha, beta)
            push!(runtime_results[name], runtime_ms)
            @printf("    %-28s | Runtime: %8.3f ms\n", name, runtime_ms)
        end

        println("\n--- Benchmarking standard CUBLAS.syrk! ---")
        for (name, T_prec) in Dict("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)
            alpha, beta = -1.0, 1.0
            runtime_ms = get_syrk_runtime_cublas(n, T_prec, alpha, beta)
            push!(cublas_runtime_results[name], runtime_ms)
            @printf("    %-28s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
        
        GC.gc(true); CUDA.reclaim() # Manual GC call after each matrix size
    end

    # --- Plotting Performance Results ---
    println("\n" * "="^60)
    println("📊 Generating and saving performance plot...")

    perf_plot = plot(title="Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10)
    
    all_results = merge(runtime_results, cublas_runtime_results)
    for name in sort(collect(keys(all_results)))
        runtimes = all_results[name]
        linestyle = startswith(name, "CUBLAS") ? :dash : :solid
        linewidth = startswith(name, "CUBLAS") ? 2 : 1
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=linestyle, linewidth=linewidth)
    end
    
    savefig(perf_plot, "recsyrk_performance.png")

    println("✅ Performance benchmark complete. Plot saved to recsyrk_performance.png")
    println("="^60)
end

run_recsyrk_performance_benchmark()