function run_all_tests()
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
        "TriMixed: [F32, F32, F64]" => [Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F64]" => [Float32, Float32, Float32, Float64],
        "TriMixed: [F32, F32, F32, F32, F64]" => [Float32, Float32, Float32, Float32, Float64],
    )
    
    all_results_accuracy = Dict{Char, Dict{String, Vector{Float64}}}()
    all_results_runtime = Dict{Char, Dict{String, Vector{Float64}}}()
    all_cublas_runtime = Dict{Char, Dict{String, Vector{Float64}}}()

    for func in ['S'] #, 'M']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n" * "="^70)
        println("🚀 Starting Benchmark for $op_name (uplo='$uplo')...")
        println("="^70)

        results_accuracy = Dict(name => Float64[] for name in keys(test_scenarios))
        results_runtime = Dict(name => Float64[] for name in keys(test_scenarios))
        cublas_runtime = Dict(
            "CUBLAS F64" => Float64[],
            "CUBLAS F32" => Float64[]
        )

        for n in sizes
            println("\n--- Testing Matrix Size: $n x $n ---")

            A_cpu = Matrix(UpperTriangular(rand(Float64, n, n)))
            A_cpu .+= Diagonal(fill(Float64(n)*100, n))
            B_cpu = rand(Float64, n, n)
            
            local B_sol_cpu
            if func == 'S'
                let A_sol_gpu = CuArray(A_cpu), B_sol_gpu_temp = CuArray(B_cpu)
                    CUBLAS.trsm!(side, uplo, trans, 'N', alpha, A_sol_gpu, B_sol_gpu_temp)
                    B_sol_cpu = collect(B_sol_gpu_temp)
                end
            else
                B_sol_cpu = alpha .* (A_cpu * B_cpu)
            end
            
            local solution_norm = norm(B_sol_cpu)

            for (name, prec_list) in test_scenarios
                T_Base = prec_list[1]
                
                try
                    local A_test, B_test
                    
                    if startswith(name, "Recursive")
                        A_test = CuArray{T_Base}(A_cpu)
                    else
                        A_test = TriMixedPrec(CuArray(A_cpu), uplo; precisions=prec_list)
                    end
                    B_test = CuArray{T_Base}(B_cpu)

                    backend = get_backend(B_test)
                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_test, B_cpu)
                        unified_rectrxm!(side, uplo, trans, alpha, func, A_test, B_test)
                    end
                    runtime_ms = perf_time_ns / 1_000_000
                    
                    error_norm = norm(collect(B_test) .- B_sol_cpu)
                    relative_error = iszero(solution_norm) ? 0.0 : error_norm / solution_norm
                    
                    push!(results_accuracy[name], -log10(max(relative_error, 1e-18)))
                    push!(results_runtime[name], runtime_ms)

                    println("  '$name' | Rel. Error: $(round(relative_error, sigdigits=3)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")

                catch e
                    if e isa CUDA.OutOfMemoryError
                        println("  '$name' | SKIPPED due to OutOfMemoryError")
                        push!(results_runtime[name], NaN)
                        push!(results_accuracy[name], NaN)
                    else
                        rethrow(e)
                    end
                end
                
                A_test = nothing
                B_test = nothing
                GC.gc(true)
                CUDA.reclaim()
            end

            println("  --- CUBLAS Baselines ---")
            for T in [Float64, Float32]
                try
                    local A_blas = CuArray{T}(A_cpu)
                    local B_blas = CuArray{T}(B_cpu)
                    backend = get_backend(A_blas)

                    time_ns = run_manual_benchmark(backend) do
                        copyto!(B_blas, B_cpu)
                        if func == 'S'
                            CUBLAS.trsm!(side, uplo, trans, 'N', T(alpha), A_blas, B_blas)
                        else
                            C_blas = similar(B_blas)
                            CUBLAS.trmm!(side, uplo, trans, 'N', T(alpha), A_blas, B_blas, C_blas)
                        end
                    end
                    runtime_ms = time_ns / 1_000_000
                    cublas_name = "CUBLAS F$(sizeof(T)*8)"
                    push!(cublas_runtime[cublas_name], runtime_ms)
                    println("  '$cublas_name' | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
                catch e
                    if e isa CUDA.OutOfMemoryError
                        cublas_name = "CUBLAS F$(sizeof(T)*8)"
                        println("  '$cublas_name' | SKIPPED due to OutOfMemoryError")
                        push!(cublas_runtime[cublas_name], NaN)
                    else
                        rethrow(e)
                    end
                end
                GC.gc(true)
                CUDA.reclaim()
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
    for (i, (name, data)) in enumerate(rec_results)
        plot!(plt, sizes, data, label=name, lw=2, marker=:auto, markersize=4)
    end
    for (i, (name, data)) in enumerate(cublas_results)
        plot!(plt, sizes, data, label=name, lw=2.5, linestyle=:dash, color=i+length(rec_results))
    end
    return plt
end

sizes, all_accuracy, all_runtime, all_cublas = run_all_tests()

for func in ['S']
    op_name = func == 'S' ? "trsm" : "trmm"
    println("\n✅ Generating plots for $op_name...")

    accuracy_plot = plot_accuracy_results(sizes, filter(p -> !isnan(p.second[end]), all_accuracy[func]), func)
    runtime_plot = plot_runtime_results(sizes, filter(p -> !isnan(p.second[end]), all_runtime[func]), filter(p -> !isnan(p.second[end]), all_cublas[func]), func)

    savefig(accuracy_plot, "$(op_name)_accuracy_results.png")
    savefig(runtime_plot, "$(op_name)_runtime_results.png")
    println("Plots saved as $(op_name)_accuracy_results.png and $(op_name)_runtime_results.png")
end