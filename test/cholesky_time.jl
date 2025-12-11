using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions

include("benchmark.jl") 
include("flops.jl")


function benchmark_op(op, reset_op, backend)
    # 1. Warm-up
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

function get_runtime_pure(A_spd_fp64, n::Int, T_prec::DataType)
    local A_clean
    
    if T_prec == Float16
        scale_factor = maximum(abs, A_spd_fp64)
        A_clean = Float16.(A_spd_fp64 ./ scale_factor) + 100*I
    else
        A_clean = T_prec.(A_spd_fp64)
    end

    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op = () -> potrf_recursive!(A_perf, 4096)
    reset_op = () -> copyto!(A_perf, A_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_potrf(T_prec, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function get_runtime_mixed(A_spd_fp64, n::Int, precisions::Vector)
    backend = KernelAbstractions.get_backend(A_spd_fp64)

    op = () -> begin
        A_to_factor = SymmMixedPrec(A_spd_fp64, 'L'; precisions=precisions)
        potrf_recursive!(A_to_factor)
    end
    
    reset_op = () -> () 

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    T_flops = precisions[1]
    flops = flops_potrf(T_flops, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function get_runtime_cusolver(A_spd_fp64, n::Int, T_prec::DataType)
    A_clean = (T_prec == Float64) ? A_spd_fp64 : T_prec.(A_spd_fp64)
    A_perf = copy(A_clean)
    backend = KernelAbstractions.get_backend(A_perf)

    op = () -> CUSOLVER.potrf!('L', A_perf)
    reset_op = () -> copyto!(A_perf, A_clean)
    
    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_potrf(T_prec, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function run_cholesky_benchmarks()
    n_values = [4096, 8192, 16384, 32768, 65536] #256, 512, 1024, 2048, 

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
        "Pure F16" => [Float16],
    )
    # mixed_scenarios = Dict(
    #     "[F32, F64, F64, F64]"      => [Float32, Float64, Float64, Float64],
    #     "[F32, F32, F32, F64]"      => [Float32, Float32, Float32, Float64],
    #     "[F32, F32, F64]"           => [Float32, Float32, Float64],
    #     "[F32, F64, F64]"           => [Float32, Float64, Float64],
    #     "[F16, F32, F32]"           => [Float16, Float32, Float32],
    #     "[F16, F16, F32]"           => [Float16, Float16, Float32],
    #     "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F32, F32, F32, F32, F32, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32],
    #     "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
    #     "[F16, F32, F64]"           => [Float16, Float32, Float64],
    #     "[F32, F64]"                => [Float32, Float64],
    #     "[F16, F64]"                => [Float16, Float64],
    #     "[F16, F32]"                => [Float16, Float32],
    # )
    mixed_scenarios = Dict(
        "[F16, F32, F64]" => [Float16, Float32, Float64],
        "[F16, F16, F32, F64]" => [Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64]
    )

    println("🚀 Starting Cholesky Benchmark...")

    for n in n_values
        A_cpu_rand = randn(Float64, n, n) * .01
        A_cpu_rand = A_cpu_rand * A_cpu_rand' + (n * 10) * I
        A_gpu = CuArray(A_cpu_rand)
        A_cpu_rand = nothing 
        A_spd_fp64 = A_gpu
        A_gpu = nothing

        println("\n" * "="^80)
        println("Benchmarking Matrix Size (n x n) = $n x $n")
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            runtime_ms, gflops = get_runtime_pure(A_spd_fp64, n, precisions[1])
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            runtime_ms, gflops = get_runtime_mixed(A_spd_fp64, n, precisions)
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end
        
        println("\n--- Standard CUSOLVER.potrf! ---")
        for (name, T_prec) in Dict("CUSOLVER F32" => Float32, "CUSOLVER F64" => Float64)
            runtime_ms, gflops = get_runtime_cusolver(A_spd_fp64, n, T_prec)
            @printf("    %-25s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
        end
    end
    
    println("\n" * "="^80)
    println("✅ Benchmark complete.")
    println("="^80)
end

run_cholesky_benchmarks()