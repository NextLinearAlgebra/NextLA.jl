using Test, AMDGPU, LinearAlgebra, Printf, Plots, KernelAbstractions

include("benchmark.jl")
include("flops.jl")

# 1. UPDATED: Adopted the cleaner benchmark_op from Script 2
function benchmark_op(op, reset_op, backend)
    # Warm-up
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

# 2. UPDATED: Accepts Master GPU arrays and casts them (GPU-to-GPU) like Script 2
function get_syrk_runtime_pure(d_A_master, d_C_master, n::Int, T_prec, alpha, beta)
    # Fast GPU-to-GPU cast/copy
    d_A = T_prec.(d_A_master)
    d_C_clean = T_prec.(d_C_master)
    
    backend = KernelAbstractions.get_backend(d_A)
    C_perf = copy(d_C_clean)

    op = () -> recsyrk!(T_prec(alpha), d_A, T_prec(beta), C_perf, 256)
    reset_op = () -> copyto!(C_perf, d_C_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_syrk(T_prec, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

# 3. UPDATED: Accepts Master GPU arrays
function get_syrk_runtime_mixed(d_A_master, d_C_master, n::Int, T_out, precisions, alpha, beta)
    # Fast GPU-to-GPU cast/copy
    d_A = T_out.(d_A_master)
    d_C_orig = T_out.(d_C_master) # This serves as the input C
    
    backend = KernelAbstractions.get_backend(d_A)
    
    op = () -> begin
        # Mixed precision wrapper creates its own copy/workspace usually, 
        # but we ensure d_C_orig is fresh if needed. 
        # Ideally SymmMixedPrec copies d_C_orig internally.
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

# 4. UPDATED: Removed CPU allocation, uses GPU master arrays
function get_syrk_runtime_blas(d_A_master, d_C_master, n::Int, T_prec, alpha, beta)
    d_A_blas = T_prec.(d_A_master)
    d_C_clean = T_prec.(d_C_master)
    d_C_blas = copy(d_C_clean)
    
    backend = AMDGPU.ROCBackend()
    
    op = () -> syrk!('L', 'N', T_prec(alpha), d_A_blas, T_prec(beta), d_C_blas)
    reset_op = () -> copyto!(d_C_blas, d_C_clean)
    
    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_syrk(T_prec, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function run_recsyrk_performance_benchmark()
    n_values = [4096, 8192, 16384, 20480, 24576, 28672, 32768, 65536]
    
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
        # 5. NEW: Generate data ONCE per N on the GPU (Optimization from Script 2)
        # We use Float64 as the master source, then cast down in the functions
        d_A_master = ROCArray{Float64}(undef, n, n)
        d_C_master = ROCArray{Float64}(undef, n, n)
        
        AMDGPU.randn!(d_A_master)
        d_A_master .*= 0.1 # Apply scaling on GPU
        fill!(d_C_master, 0.0) # Zero initialization on GPU
        
        AMDGPU.synchronize()

        println("\n" * "-"^50)
        println("Benchmarking Performance for C(n x n)=$n, A(n x n)=$n")
        alpha, beta = -1.0, 1.0
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[end]
            # Pass master arrays instead of creating new ones
            runtime_ms, gflops = get_syrk_runtime_pure(d_A_master, d_C_master, n, T_prec, alpha, beta)
            push!(runtime_results[name], runtime_ms)
            @printf("        %-28s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            T_out = precisions[end]
            # Pass master arrays
            runtime_ms, gflops = get_syrk_runtime_mixed(d_A_master, d_C_master, n, T_out, precisions, alpha, beta)
            push!(runtime_results[name], runtime_ms)
            @printf("        %-28s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n--- Benchmarking standard Vendor BLAS syrk! ---")
        for (name, T_prec) in Dict("Vendor BLAS F32" => Float32, "Vendor BLAS F64" => Float64)
            alpha, beta = -1.0, 1.0
            # Pass master arrays
            runtime_ms, gflops = get_syrk_runtime_blas(d_A_master, d_C_master, n, T_prec, alpha, beta)
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