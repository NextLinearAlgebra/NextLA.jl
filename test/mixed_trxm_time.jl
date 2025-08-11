using Test, CUDA, LinearAlgebra, Printf, KernelAbstractions

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

function get_runtime_recursive(A_cpu, B_cpu, n::Int, T_prec, op_char, side, uplo, trans, alpha)
    A_perf = CuArray{T_prec}(A_cpu)
    B_clean = CuArray{T_prec}(B_cpu)
    B_perf = copy(B_clean)
    backend = KernelAbstractions.get_backend(A_perf)
    
    op = () -> unified_rectrxm!(side, uplo, trans, alpha, op_char, A_perf, B_perf)
    reset_op = () -> copyto!(B_perf, B_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000
    
    flops = flops_trsm(T_prec, n, n)
    gflops = calculate_gflops(flops, min_time_ns)
    
    return runtime_ms, gflops
end

function get_runtime_mixed(A_cpu, B_cpu, n::Int, precisions, op_char, side, uplo, trans, alpha)
    T_base = precisions[1]
    A_gpu = CuArray(A_cpu)
    B_clean = CuArray{T_base}(B_cpu)
    B_perf = copy(B_clean)
    backend = KernelAbstractions.get_backend(B_perf)

    op = () -> begin
        A_mixed_perf = TriMixedPrec(A_gpu, uplo; precisions=precisions)
        unified_rectrxm!(side, uplo, trans, alpha, op_char, A_mixed_perf, B_perf)
    end
    
    reset_op = () -> copyto!(B_perf, B_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_trsm(T_base, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end

function get_runtime_cublas(A_cpu, B_cpu, n::Int, T_prec, op_char, side, uplo, trans, alpha)
    A_blas = CuArray{T_prec}(A_cpu)
    B_blas_clean = CuArray{T_prec}(B_cpu)
    B_blas = copy(B_blas_clean)
    backend = KernelAbstractions.get_backend(A_blas)

    op = () -> begin
        if op_char == 'S'
            CUBLAS.trsm!(side, uplo, trans, 'N', T_prec(alpha), A_blas, B_blas)
        else
            C_blas = similar(B_blas)
            CUBLAS.trmm!(side, uplo, trans, 'N', T_prec(alpha), A_blas, B_blas, C_blas)
        end
    end
    reset_op = () -> copyto!(B_blas, B_blas_clean)

    min_time_ns = benchmark_op(op, reset_op, backend)
    runtime_ms = min_time_ns / 1_000_000

    flops = flops_trsm(T_prec, n, n)
    gflops = calculate_gflops(flops, min_time_ns)

    return runtime_ms, gflops
end


function run_tr_benchmarks()
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    uplo = 'U'
    side = 'L'
    alpha = 1.0f0
    trans = 'N'

    test_scenarios = Dict(
        "Recursive Float64" => [Float64],
        "Recursive Float32" => [Float32],
        "Recursive Float16" => [Float16],
        "TriMixed: [F16, F64]" => [Float16, Float64],
        "TriMixed: [F64, F16]" => [Float64, Float16],
        "TriMixed: [F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "TriMixed: [F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        "TriMixed: [F16, F16, F32]" => [Float16, Float16, Float32],
        "TriMixed: [F32, F64]" => [Float32, Float64],
        "TriMixed: [F16, F32]" => [Float16, Float32],
        "TriMixed: [F16, F64]" => [Float16, Float64],
        "TriMixed: [F32, F32, F64]" => [Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F32, F64]" => [Float32, Float32, Float32, Float32, Float64],
    )

    for op_char in ['S', 'M']
        op_name = op_char == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^80)
        println("🚀 Starting Benchmark for $op_name (side='$side', uplo='$uplo')...")
        
        for n in sizes
            println("\n" * "="^80)
            println("Benchmarking Matrix Size (n x n) = $n x $n")

            A_cpu = Matrix(UpperTriangular(rand(Float64, n, n)))
            A_cpu .+= Diagonal(fill(Float64(n), n))
            B_cpu = rand(Float64, n, n)
            
            println("\n--- Custom & Mixed Precision Scenarios ---")
            for (name, precisions) in test_scenarios
                local runtime_ms, gflops
                if startswith(name, "Recursive")
                    T_prec = precisions[1]
                    runtime_ms, gflops = get_runtime_recursive(A_cpu, B_cpu, n, T_prec, op_char, side, uplo, trans, alpha)
                else 
                    runtime_ms, gflops = get_runtime_mixed(A_cpu, B_cpu, n, precisions, op_char, side, uplo, trans, alpha)
                end
                @printf("    %-45s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
            end

            println("\n--- Standard CUBLAS Baselines ---")
            for T_prec in [Float64, Float32]
                runtime_ms, gflops = get_runtime_cublas(A_cpu, B_cpu, n, T_prec, op_char, side, uplo, trans, alpha)
                cublas_name = "CUBLAS F$(sizeof(T_prec)*8)"
                @printf("    %-45s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", cublas_name, runtime_ms, gflops)
            end
            
            A_cpu, B_cpu = (nothing, nothing)
            GC.gc(true); CUDA.reclaim()
        end
    end
    
    println("\n" * "="^80)
    println("✅ All benchmarks complete.")
    println("="^80)
end

run_tr_benchmarks()