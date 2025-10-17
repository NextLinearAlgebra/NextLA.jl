using Test, AMDGPU, rocBLAS, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl")
include("flops.jl")

function benchmark_op(op, reset_op, backend)
    reset_op()
    op()
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:5
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

function get_syrk_runtime_pure(d_A, C_clean, n::Int, T_prec, alpha, beta)
    backend = KernelAbstractions.get_backend(d_A)
    C_perf = copy(C_clean)

    op = () -> recsyrk!(T_prec(alpha), d_A, T_prec(beta), C_perf, 256)
    reset_op = () -> copyto!(C_perf, C_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_syrk(T_prec, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function get_syrk_runtime_mixed(d_A, d_C_orig, n::Int, T_out, precisions, alpha, beta)
    backend = KernelAbstractions.get_backend(d_A)
    
    op = () -> begin
        C_perf_mixed = SymmMixedPrec(copy(d_C_orig), 'L'; precisions=precisions)
        recsyrk!(T_out(alpha), d_A, T_out(beta), C_perf_mixed)
    end
    
    reset_op = () -> ()

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_syrk(T_out, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function get_syrk_runtime_blas(n::Int, T_prec, alpha, beta)
    backend = AMDGPU.rocbackend()
    A_cpu = randn(T_prec, n, n)
    C_cpu = zeros(T_prec, n, n)

    d_A_blas = KernelAbstractions.allocate(backend, T_prec, n, n)
    copyto!(d_A_blas, A_cpu)
    d_C_blas = KernelAbstractions.allocate(backend, T_prec, n, n)
    copyto!(d_C_blas, C_cpu)
    d_C_clean = copy(d_C_blas) 
    
    op = () -> syrk!('L', 'N', T_prec(alpha), d_A_blas, T_prec(beta), d_C_blas)
    reset_op = () -> copyto!(d_C_blas, d_C_clean)
    
    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_syrk(T_prec, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end


function run_recsyrk_performance_benchmark()
    n_values = [4096, 8192, 16384, 32768, 65536]
    backend = AMDGPU.rocbackend()
    
    pure_scenarios = Dict(
        "Pure F16" => [Float16, Float16, Float16],
        "Pure F32" => [Float32, Float32, Float32],
        "Pure F64" => [Float64, Float64, Float64],
    )
    mixed_scenarios = Dict(
        "[F32, F32, F64]"          => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]"      => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]"      => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"          => [Float32, Float64, Float64],
        "[F16, F16, F32]"          => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "[F16, F32, F32]"          => [Float16, Float32, Float32]
    )

    runtime_results = Dict(name => Float64[] for name in union(keys(pure_scenarios), keys(mixed_scenarios)))
    blas_runtime_results = Dict("Vendor BLAS F32" => Float64[], "Vendor BLAS F64" => Float64[])

    println("🚀 Starting recsyrk! Performance Benchmark...")

    for n in n_values
        println("\n" * "-"^50)
        println("Benchmarking Performance for C(n x n)=$n, A(n x n)=$n")
        alpha, beta = -1.0, 1.0
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[end]
            A_cpu = randn(T_prec, n, n) .* 0.1f0
            C_cpu = zeros(T_prec, n, n)
            d_A = KernelAbstractions.allocate(backend, T_prec, n, n)
            copyto!(d_A, A_cpu)
            d_C_orig = KernelAbstractions.allocate(backend, T_prec, n, n)
            copyto!(d_C_orig, C_cpu)
            runtime_ms, gflops = get_syrk_runtime_pure(d_A, d_C_orig, n, T_prec, alpha, beta)
            push!(runtime_results[name], runtime_ms)
            @printf("        %-28s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            T_out = precisions[end]
            A_cpu = randn(T_out, n, n) .* 0.1f0
            C_cpu = zeros(T_out, n, n)
            d_A = KernelAbstractions.allocate(backend, T_out, n, n)
            copyto!(d_A, A_cpu)
            d_C_orig = KernelAbstractions.allocate(backend, T_out, n, n)
            copyto!(d_C_orig, C_cpu)
            
            runtime_ms, gflops = get_syrk_runtime_mixed(d_A, d_C_orig, n, T_out, precisions, alpha, beta)
            push!(runtime_results[name], runtime_ms)
            @printf("        %-28s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n--- Benchmarking standard Vendor BLAS syrk! ---")
        for (name, T_prec) in Dict("Vendor BLAS F32" => Float32, "Vendor BLAS F64" => Float64)
            alpha, beta = -1.0, 1.0
            runtime_ms, gflops = get_syrk_runtime_blas(n, T_prec, alpha, beta)
            push!(blas_runtime_results[name], runtime_ms)
            @printf("        %-28s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end
        
        GC.gc(true)
    end

    println("\n" * "="^60)
    println("📊 Generating and saving performance plot...")

    perf_plot = plot(title="Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10)
    
    all_results = merge(runtime_results, blas_runtime_results)
    for name in sort(collect(keys(all_results)))
        runtimes = all_results[name]
        linestyle = startswith(name, "Vendor BLAS") ? :dash : :solid
        linewidth = startswith(name, "Vendor BLAS") ? 2 : 1
        plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=linestyle, linewidth=linewidth)
    end
    
    savefig(perf_plot, "recsyrk_performance.png")

    println("✅ Performance benchmark complete. Plot saved to recsyrk_performance.png")
    println("="^60)
end

run_recsyrk_performance_benchmark()