using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions

include("benchmark.jl") 


function get_runtime_pure(n::Int, T_prec::DataType)
    A_cpu = randn(Float64, n, n)
    A_spd_fp64 = CuArray(A_cpu * A_cpu' + (n * 100) * I)
    backend = KernelAbstractions.get_backend(A_spd_fp64)
    
    time_ns = run_manual_benchmark(backend) do
        potrf_recursive!(T_prec.(A_spd_fp64), 4096)
    end
    
    return time_ns / 1_000_000
end


function get_runtime_mixed(n::Int, precisions::Vector)
    A_cpu = randn(Float64, n, n)
    A_spd_fp64 = CuArray(A_cpu * A_cpu' + (n * 100) * I)
    backend = KernelAbstractions.get_backend(A_spd_fp64)

    time_ns = run_manual_benchmark(backend) do
        potrf_recursive!(SymmMixedPrec(A_spd_fp64, 'L'; precisions=precisions))
    end
    
    return time_ns / 1_000_000
end

"""
    get_runtime_cusolver(n::Int, T_prec::DataType)

Creates a matrix of size `n x n`, runs the standard CUSOLVER.potrf! 
factorization using precision `T_prec`, and returns the runtime.
"""
function get_runtime_cusolver(n::Int, T_prec::DataType)
    A_cpu = randn(Float64, n, n)
    A_spd_fp64 = CuArray(A_cpu * A_cpu' + (n * 100) * I)
    backend = KernelAbstractions.get_backend(A_spd_fp64)
    A_spd_base = (T_prec == Float64) ? A_spd_fp64 : T_prec.(A_spd_fp64)

    time_ns = run_manual_benchmark(backend) do
        CUSOLVER.potrf!('L', copy(A_spd_base))
    end
    
    return time_ns / 1_000_000
end

"""
    run_cholesky_benchmarks()

Main function to orchestrate the Cholesky factorization benchmarks across
different matrix sizes and precision scenarios.
"""
function run_cholesky_benchmarks()
    n_values = [1024, 2048, 4096, 8192, 16384, 32768, 65536] 

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
    )
    mixed_scenarios = Dict(
        "[F32, F64, F64, F64]"      => [Float32, Float64, Float64, Float64],
        "[F32, F32, F32, F64]"      => [Float32, Float32, Float32, Float64],
        "[F32, F32, F64]"           => [Float32, Float32, Float64],
        "[F32, F64, F64]"           => [Float32, Float64, Float64],
        "[F16, F32, F32]"           => [Float16, Float32, Float32],
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F64]"           => [Float16, Float32, Float64],
        "[F32, F64]"                => [Float32, Float64],
        "[F16, F64]"                => [Float16, Float64],
        "[F16, F32]"                => [Float16, Float32],
    )

    println("🚀 Starting Cholesky Benchmark...")

    for n in n_values
        println("\n" * "="^80)
        println("Benchmarking Matrix Size (n x n) = $n x $n")
        
        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            runtime_ms = get_runtime_pure(n, precisions[1])
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            runtime_ms = get_runtime_mixed(n, precisions)
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
        
        println("\n--- Standard CUSOLVER.potrf! ---")
        for (name, T_prec) in Dict("CUSOLVER F32" => Float32, "CUSOLVER F64" => Float64)
            runtime_ms = get_runtime_cusolver(n, T_prec)
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
    end
    
    println("\n" * "="^80)
    println("✅ Benchmark complete.")
    println("="^80)
end

run_cholesky_benchmarks()