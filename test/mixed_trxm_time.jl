function run_benchmarks()
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
        "TriMixed: [F16, F16, F32]" => [Float16, Float16, Float32],
        "TriMixed: [F32, F64]" => [Float32, Float64],
        "TriMixed: [F16, F32]" => [Float16, Float32],
        "TriMixed: [F16, F64]" => [Float16, Float64],
        "TriMixed: [F32, F32, F64]" => [Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F32, F64]" => [Float32, Float32, Float32, Float32, Float64],
    )

    for func in ['S', 'M']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^70)
        println("🚀 Starting Benchmark for $op_name (uplo='$uplo')...")
        println("="^70)

        for n in sizes
            println("\n--- Benchmarking Matrix Size: $n x $n ---")

            A_cpu = Matrix(UpperTriangular(rand(Float64, n, n)))
            A_cpu .+= Diagonal(fill(Float64(n), n))
            B_cpu = rand(Float64, n, n)
            A_gpu_base = CuArray(A_cpu)

            for (name, prec_list) in test_scenarios
                T_Base = prec_list[1]
                B_clean_copy = CuArray{T_Base}(B_cpu)
                B_perf = copy(B_clean_copy)
                backend = get_backend(A_gpu_base)
                
                local perf_time_ns
                if startswith(name, "Recursive")
                    A_perf = CuArray{T_Base}(A_gpu_base)
                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_perf, B_clean_copy)
                        unified_rectrxm!(side, uplo, trans, alpha, func, A_perf, B_perf)
                    end
                else
                    A_mixed_perf = TriMixedPrec(A_gpu_base, uplo; precisions=prec_list)
                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_perf, B_clean_copy)
                        unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed_perf, B_perf)
                    end
                end
                runtime_ms = perf_time_ns / 1_000_000
                @printf "  %-40s | Runtime: %.4f ms\n" "'$name'" runtime_ms
            end

            GC.gc(true)
            CUDA.reclaim()

            println("  --- CUBLAS Baselines ---")
            for T in [Float64, Float32]
                A_blas = CuArray{T}(A_cpu)
                B_blas_clean = CuArray{T}(B_cpu)
                B_blas = copy(B_blas_clean)
                backend = get_backend(A_blas)

                time_ns = run_manual_benchmark(backend) do
                    copyto!(B_blas, B_blas_clean)
                    if func == 'S'
                        CUBLAS.trsm!(side, uplo, trans, 'N', T(alpha), A_blas, B_blas)
                    else
                        C_blas = similar(B_blas)
                        CUBLAS.trmm!(side, uplo, trans, 'N', T(alpha), A_blas, B_blas, C_blas)
                    end
                end
                runtime_ms = time_ns / 1_000_000
                cublas_name = "CUBLAS F$(sizeof(T)*8)"
                @printf "  %-40s | Runtime: %.4f ms\n" "'$cublas_name'" runtime_ms
            end
            
            A_cpu = nothing
            B_cpu = nothing
            A_gpu_base = nothing
            GC.gc(true)
            CUDA.reclaim()
        end
    end
end


println("Running Benchmarks...")
run_benchmarks()
println("\n✅ Benchmarks complete.")