using Test, AMDGPU, LinearAlgebra, Printf, KernelAbstractions

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
    backend = AMDGPU.rocbackend()
    A_perf = KernelAbstractions.allocate(backend, T_prec, size(A_cpu)...)
    copyto!(A_perf, A_cpu)
    B_clean = KernelAbstractions.allocate(backend, T_prec, size(B_cpu)...)
    copyto!(B_clean, B_cpu)
    B_perf = copy(B_clean)
    
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
    backend = AMDGPU.rocbackend()
    A_gpu = KernelAbstractions.allocate(backend, eltype(A_cpu), size(A_cpu)...)
    copyto!(A_gpu, A_cpu)
    B_clean = KernelAbstractions.allocate(backend, T_base, size(B_cpu)...)
    copyto!(B_clean, B_cpu)
    B_perf = copy(B_clean)

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

function get_runtime_blas(A_cpu, B_cpu, n::Int, T_prec, op_char, side, uplo, trans, alpha)
    backend = AMDGPU.rocbackend()
    A_blas = KernelAbstractions.allocate(backend, T_prec, size(A_cpu)...)
    copyto!(A_blas, A_cpu)
    B_blas_clean = KernelAbstractions.allocate(backend, T_prec, size(B_cpu)...)
    copyto!(B_blas_clean, B_cpu)
    B_blas = copy(B_blas_clean)

    op = () -> begin
        if op_char == 'S'
            trsm!(side, uplo, trans, 'N', T_prec(alpha), A_blas, B_blas)
        else
            C_blas = similar(B_blas)
            trmm!(side, uplo, trans, 'N', T_prec(alpha), A_blas, B_blas, C_blas)
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
    sizes = [4096, 8192, 16384, 32768, 65536]
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

    for op_char in ['S']
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
                @printf("        %-45s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
            end

            println("\n--- Standard Vendor BLAS Baselines ---")
            for T_prec in [Float64, Float32]
                runtime_ms, gflops = get_runtime_blas(A_cpu, B_cpu, n, T_prec, op_char, side, uplo, trans, alpha)
                blas_name = "Vendor BLAS F$(sizeof(T_prec)*8)"
                @printf("        %-45s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", blas_name, runtime_ms, gflops)
            end
            
            A_cpu, B_cpu = (nothing, nothing)
            GC.gc(true)
        end
    end
    
    println("\n" * "="^80)
    println("✅ All benchmarks complete.")
    println("="^80)
end

run_tr_benchmarks()