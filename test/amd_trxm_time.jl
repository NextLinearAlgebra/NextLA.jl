using Test, AMDGPU, LinearAlgebra, Printf, KernelAbstractions

include("benchmark.jl")
include("flops.jl")

# 1. UPDATED: Consistent, clean benchmark harness
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

# 2. UPDATED: Takes Master GPU Arrays, casts/copies on device
function get_runtime_recursive(d_A_master, d_B_master, n::Int, T_prec, op_char, side, uplo, trans, alpha)
    # GPU-to-GPU cast
    A_perf = T_prec.(d_A_master)
    B_clean = T_prec.(d_B_master)
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

# 3. UPDATED: Takes Master GPU Arrays
function get_runtime_mixed(d_A_master, d_B_master, n::Int, precisions, op_char, side, uplo, trans, alpha)
    T_base = precisions[1]
    
    # We assume A starts as F64 (master) and TriMixedPrec handles the internal precision.
    # We copy it to ensure we don't modify the master (though TRSM 'A' is usually read-only).
    A_gpu = copy(d_A_master) 
    
    # B is the target, so we cast to the base precision
    B_clean = T_base.(d_B_master)
    B_perf = copy(B_clean)

    backend = KernelAbstractions.get_backend(A_gpu)

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

# 4. UPDATED: Takes Master GPU Arrays
function get_runtime_blas(d_A_master, d_B_master, n::Int, T_prec, op_char, side, uplo, trans, alpha)
    # Fast GPU-to-GPU cast
    A_blas = T_prec.(d_A_master)
    B_blas_clean = T_prec.(d_B_master)
    B_blas = copy(B_blas_clean)

    backend = AMDGPU.ROCBackend()

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
    sizes = [4096, 8192, 16384, 20480, 24576, 28672, 32768, 65536]
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

            # 5. NEW: Direct GPU Generation
            d_A_master = ROCArray{Float64}(undef, n, n)
            d_B_master = ROCArray{Float64}(undef, n, n)
            
            AMDGPU.rand!(d_A_master)
            AMDGPU.rand!(d_B_master)
            
            # Diagonal Dominance on GPU (mimics A_cpu .+= Diagonal(n))
            # Note: We don't need to explicitly UpperTriangular() the data 
            # because the BLAS/Recursive calls respect the 'uplo' flag.
            view(d_A_master, diagind(d_A_master)) .+= Float64(n)
            
            AMDGPU.synchronize()
            
            println("\n--- Custom & Mixed Precision Scenarios ---")
            for (name, precisions) in test_scenarios
                local runtime_ms, gflops
                if startswith(name, "Recursive")
                    T_prec = precisions[1]
                    runtime_ms, gflops = get_runtime_recursive(d_A_master, d_B_master, n, T_prec, op_char, side, uplo, trans, alpha)
                else 
                    runtime_ms, gflops = get_runtime_mixed(d_A_master, d_B_master, n, precisions, op_char, side, uplo, trans, alpha)
                end
                @printf("        %-45s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", name, runtime_ms, gflops)
            end

            println("\n--- Standard Vendor BLAS Baselines ---")
            for T_prec in [Float64, Float32]
                runtime_ms, gflops = get_runtime_blas(d_A_master, d_B_master, n, T_prec, op_char, side, uplo, trans, alpha)
                blas_name = "Vendor BLAS F$(sizeof(T_prec)*8)"
                @printf("        %-45s | Runtime: %8.3f ms | GFLOPS: %8.2f\n", blas_name, runtime_ms, gflops)
            end
            
            GC.gc(true)
        end
    end
    
    println("\n" * "="^80)
    println("✅ All benchmarks complete.")
    println("="^80)
end

run_tr_benchmarks()