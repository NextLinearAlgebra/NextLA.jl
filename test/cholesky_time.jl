using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions

include("benchmark.jl") 

function run_cholesky_benchmarks()
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

    println("🚀 Starting Cholesky Benchmark...")

    for n in n_values
        println("\n" * "="^80)
        println("Benchmarking Matrix Size (n x n) = $n x $n")
        
        A_cpu = randn(Float64, n, n)
        A_spd_fp64 = CuArray(A_cpu * A_cpu' + (n*100) * I)
        
        backend = KernelAbstractions.get_backend(A_spd_fp64)

        println("\n--- Pure Precision Scenarios ---")
        for (name, precisions) in pure_scenarios
            T_prec = precisions[1] 
            
            time_ns = run_manual_benchmark(backend) do
                A_to_factor = copy(T_prec.(A_spd_fp64))
                potrf_recursive!(A_to_factor, 4096)
            end
            runtime_ms = time_ns / 1_000_000
            
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end

        println("\n--- Mixed Precision Scenarios ---")
        for (name, precisions) in mixed_scenarios
            time_ns = run_manual_benchmark(backend) do
                A_to_factor = SymmMixedPrec(copy(A_spd_fp64), 'L'; precisions=precisions)
                potrf_recursive!(A_to_factor)
            end
            runtime_ms = time_ns / 1_000_000
            
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end
        
        GC.gc(true)
        CUDA.reclaim()

        println("\n--- Standard CUSOLVER.potrf! ---")
        for (name, T_prec) in Dict("CUSOLVER F32" => Float32, "CUSOLVER F64" => Float64)
            local A_spd_base
            if T_prec == Float64
                A_spd_base = A_spd_fp64
            else
                A_spd_base = T_prec.(A_spd_fp64)
            end
            
            time_ns = run_manual_benchmark(backend) do
                CUSOLVER.potrf!('L', copy(A_spd_base))
            end
            runtime_ms = time_ns / 1_000_000
            @printf("    %-25s | Runtime: %8.3f ms\n", name, runtime_ms)
        end

        A_cpu = nothing
        A_spd_fp64 = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
    
    println("\n" * "="^80)
    println("✅ Benchmark complete.")
    println("="^80)
end

run_cholesky_benchmarks()