function run_all_tests()
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] 
    m = 64
    uplo = 'U'
    side = 'L'
    alpha = 1.0f0 # Use Float32 for alpha
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
    

    # Dictionaries to store results for each function type ('M' or 'S')
    all_results_accuracy = Dict{Char, Dict{String, Vector{Float64}}}()
    all_results_runtime = Dict{Char, Dict{String, Vector{Float64}}}()


    for func in ['S', 'M']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^70)
        println("🚀 Starting Benchmark for $op_name (uplo='$uplo')...")
        println("="^70)

        # Initialize result dictionaries for this function type
        results_accuracy = Dict(name => Float64[] for name in keys(test_scenarios))
        results_runtime = Dict(name => Float64[] for name in keys(test_scenarios))
        cublas_runtime = Dict(
            "CUBLAS F64" => Float64[], 
            "CUBLAS F32" => Float64[]
        )

        for n in sizes
            println("\n--- Testing Matrix Size: $n x $n ---")

            A_cpu = Matrix(UpperTriangular(rand(Float64, n, n)))
            diag_strength = Float64(n)
            A_cpu .+= Diagonal(fill(diag_strength, n))
            B_cpu = rand(Float64, n, m)
            
            # --- Calculate Ground Truth Solution (FP64) ---
            A_sol_gpu = CuArray(A_cpu)
            B_sol_gpu = CuArray(B_cpu)
            if func == 'S' # TRSM: B <- alpha * inv(A) * B
                CUBLAS.trsm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu)
            else # TRMM: B <- alpha * A * B
                CUBLAS.trmm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu, similar(B_sol_gpu))
                # Note: TRMM output must be a different matrix, so we recalculate
                B_sol_gpu = alpha .* (A_cpu * B_cpu)
            end

            # --- Benchmark Recursive and Mixed-Precision Implementations ---
            for (name, prec_list) in test_scenarios
                T_Base = prec_list[1]
                A_test_gpu = CuArray(A_cpu) # Use original for mixed-prec constructor
                B_test_gpu = CuArray{T_Base}(B_cpu)
                B_clean_copy = copy(B_test_gpu)

                # Execute once for accuracy check
                if startswith(name, "Recursive")
                    unified_rectrxm!(side, uplo, trans, alpha, func, CuArray{T_Base}(A_test_gpu), B_test_gpu)
                else
                    A_mixed = TriMixedPrec(A_test_gpu, uplo; precisions=prec_list)
                    unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed, B_test_gpu)
                end

                error_norm = norm(CuArray{Float64}(B_test_gpu) .- B_sol_gpu)
                solution_norm = norm(B_sol_gpu)
                relative_error = iszero(solution_norm) ? 0.0 : error_norm / solution_norm
                push!(results_accuracy[name], -log10(max(relative_error, 1e-18)))
                
                # Benchmark for performance
                backend = get_backend(A_test_gpu)
                local perf_time_ns
                if startswith(name, "Recursive")
                    A_perf = CuArray{T_Base}(A_test_gpu)
                    B_perf = CuArray{T_Base}(B_cpu)
                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_perf, B_clean_copy)
                        unified_rectrxm!(side, uplo, trans, alpha, func, A_perf, B_perf)
                    end
                else
                    A_mixed_perf = TriMixedPrec(A_test_gpu, uplo; precisions=prec_list)
                    B_perf = CuArray{T_Base}(B_cpu)
                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_perf, B_clean_copy)
                        unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed_perf, B_perf)
                    end
                end
                runtime_ms = perf_time_ns / 1_000_000
                push!(results_runtime[name], runtime_ms)
                println("  '$name' | Rel. Error: $(round(relative_error, sigdigits=3)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
            end

            # --- Benchmark CUBLAS Baselines ---
            println("  --- CUBLAS Baselines ---")
            for T in [Float64, Float32]
                A_blas = CuArray{T}(A_cpu)
                B_blas = CuArray{T}(B_cpu)
                B_clean = copy(B_blas)
                backend = get_backend(A_blas)

                time_ns = run_manual_benchmark(backend) do
                    copyto!(B_blas, B_clean)
                    if func == 'S'
                        CUBLAS.trsm!(side, uplo, trans, 'N', T(alpha), A_blas, B_blas)
                    else
                        # CUBLAS trmm can perform C = alpha*op(A)*B, but for fair comparison,
                        # we use it as B <- alpha*op(A)*B, which requires a temporary matrix
                        # or performs the operation out-of-place. Let's do out-of-place.
                        C_blas = similar(B_blas)
                        CUBLAS.trmm!(side, uplo, trans, 'N', T(alpha), A_blas, B_blas, C_blas)
                    end
                end
                runtime_ms = time_ns / 1_000_000
                cublas_name = "CUBLAS F$(sizeof(T)*8)"
                push!(cublas_runtime[cublas_name], runtime_ms)
                println("  '$cublas_name' | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
            end
        end
        all_results_accuracy[func] = results_accuracy
        all_results_runtime[func] = results_runtime
        all_cublas_runtime[func] = cublas_runtime
    end
    return sizes, all_results_accuracy, all_results_runtime, all_cublas_runtime
end


function plot_accuracy_results(sizes, results, func_char::Char)
    op_name = func_char == 'S' ? "TRSM" : "TRMM"
    results_to_plot = filter(p -> !startswith(p.first, "Recursive"), results)
    plt = plot(
        title="$op_name Accuracy vs. Matrix Size",
        ylabel="-log10(Relative Error) [Higher is Better]",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        fontfamily="Computer Modern",
        legend=:outertopright,
        size=(800, 600),
        dpi=300
    )
    for (i, (name, data)) in enumerate(results_to_plot)
        plot!(plt, sizes, data, label=name, lw=2, marker=:auto, markersize=4)
    end
    return plt
end

function plot_runtime_results(sizes, rec_results, cublas_results, func_char::Char)
    op_name = func_char == 'S' ? "TRSM" : "TRMM"
    plt = plot(
        title="$op_name Performance vs. Matrix Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        fontfamily="Computer Modern",
        legend=:outertopright,
        size=(800, 600),
        dpi=300
    )
    # Plot recursive and mixed-precision results with solid lines
    for (i, (name, data)) in enumerate(rec_results)
        plot!(plt, sizes, data, label=name, lw=2, marker=:auto, markersize=4)
    end
    # Plot CUBLAS baselines with dashed lines
    for (i, (name, data)) in enumerate(cublas_results)
        plot!(plt, sizes, data, label=name, lw=2.5, linestyle=:dash, color=i+length(rec_results))
    end
    return plt
end


# --- Main Execution ---
sizes, all_accuracy, all_runtime, all_cublas = run_all_tests()

for func in ['S'] #, 'M']
    op_name = func == 'S' ? "trsm" : "trmm"
    println("\n✅ Generating plots for $op_name...")

    accuracy_plot = plot_accuracy_results(sizes, all_accuracy[func], func)
    runtime_plot = plot_runtime_results(sizes, all_runtime[func], all_cublas[func], func)

    savefig(accuracy_plot, "$(op_name)_accuracy_results.png")
    savefig(runtime_plot, "$(op_name)_runtime_results.png")
    println("Plots saved as $(op_name)_accuracy_results.png and $(op_name)_runtime_results.png")
end