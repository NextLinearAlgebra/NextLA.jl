using Plots
using KernelAbstractions 
include("benchmark.jl")

function run_all_tests()
    sizes = [256, 512, 1024, 2048, 4096, 8192] 
    m = 64
    uplo = 'U'
    side = 'L'
    alpha = 1.0
    trans = 'N'

    test_scenarios = Dict(
        "Pure Float64" => [Float64],
        "Pure Float32" => [Float32],
        "Pure Float16" => [Float16],
        "TriMixed: [F32, F64]" => [Float32, Float64],
        "TriMixed: [F16, F32]" => [Float16, Float32], 
        "TriMixed: [F16, F64]" => [Float16, Float64],
        "TriMixed: [F64, F32, F64]" => [Float64, Float32, Float64],
        "TriMixed: [F16, F16, F32]" => [Float16, Float16, Float32], 
        "TriMixed: [F32, F16, F64]" => [Float32, Float16, Float64],
        "TriMixed: [F32, F32, F32, F64]" => [Float32, Float16, Float32, Float64],
    )

    all_results_accuracy = Dict{Char, Dict{String, Vector{Float64}}}()
    all_results_runtime = Dict{Char, Dict{String, Vector{Float64}}}()

    for func in ['M', 'S']
        op_name = func == 'S' ? "TRSM" : "TRMM"
        println("\n=======================================================")
        println("Starting Accuracy & Runtime Benchmark for $op_name (uplo='$uplo')...")
        println("=======================================================")

        results_accuracy = Dict(name => Float64[] for name in keys(test_scenarios))
        results_runtime = Dict(name => Float64[] for name in keys(test_scenarios))

        for n in sizes
            println("\n--- Testing Matrix Size: $n x $n ---")

            # Generate CPU data once per size
            A_cpu = Matrix(UpperTriangular(rand(Float64, n, n)))
            A_cpu += Diagonal(fill(10.0, n))
            B_cpu = rand(Float64, n, m)

            # --- Calculate Reference Solution ---
            local B_solution_gpu_alpha
            if func == 'S' # TRSM: inv(A) * B
                A_blas_gpu = CuArray(A_cpu)
                B_solution_gpu = CuArray(B_cpu)
                B_solution_gpu_alpha = alpha .* B_solution_gpu
                CUBLAS.trsm!(side, uplo, trans, 'N', 1.0, A_blas_gpu, B_solution_gpu_alpha)
            else # TRMM: A * B
                A_blas_gpu = CuArray(A_cpu)
                B_solution_gpu = CuArray(B_cpu)
                B_solution_gpu_alpha = alpha .* B_solution_gpu
                CUBLAS.trmm!(side, uplo, trans, 'N', 1.0, A_blas_gpu, B_solution_gpu_alpha, B_solution_gpu_alpha)
            end


            for (name, prec_list) in test_scenarios
                T_Base = prec_list[end]
                B_gpu = CuArray{T_Base}(B_cpu)

                # --- Run the recursive implementation ---
                if startswith(name, "Pure")
                    A_gpu = CuArray{T_Base}(A_cpu)
                    B_gpu .= alpha .* B_gpu
                    if func == 'S'
                        threshold = 256
                    else 
                        threshold = 16
                    end
                    unified_rec(func, side, uplo, A_gpu, B_gpu, threshold)
                else
                    A_gpu = CuArray(A_cpu)
                    A_mixed = TriMixedPrec(A_gpu, uplo; precisions=prec_list)
                    unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed, B_gpu)
                end

                # --- Calculate and store accuracy ---
                error_norm = norm(CuArray{Float64}(B_gpu) .- B_solution_gpu_alpha)
                solution_norm = norm(B_solution_gpu_alpha)
                relative_error = error_norm / solution_norm
                push!(results_accuracy[name], -log10(max(relative_error, 1e-18))) # Avoid log(0)
                println("  Scenario: '$name' | Relative Error: $relative_error")

                # --- Benchmark and store runtime ---
                local perf_time_ns
                if name == "Pure Float64" || name == "Pure Float32"
                    A_gpu_perf = CuArray{T_Base}(A_cpu)
                    B_gpu_perf = CuArray{T_Base}(B_cpu)
                    backend = get_backend(A_gpu_perf)
                    B_clean_copy = copy(B_gpu_perf) # Create a clean copy to reset B_gpu_perf each iter

                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_gpu_perf, B_clean_copy) # Reset B_gpu_perf to its original state
                        unified_rec(func, side, uplo, A_gpu_perf, B_gpu_perf, threshold)
                    end
                else
                    A_gpu_perf = CuArray(A_cpu)
                    A_mixed_perf = TriMixedPrec(A_gpu_perf, uplo; precisions=prec_list)
                    B_gpu_perf = CuArray{T_Base}(B_cpu)

                    backend = get_backend(A_gpu_perf)
                    B_clean_copy = copy(B_gpu_perf) # Create a clean copy to reset B_gpu_perf each iter

                    perf_time_ns = run_manual_benchmark(backend) do
                        copyto!(B_gpu_perf, B_clean_copy) # Reset B_gpu_perf to its original state
                        unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed_perf, B_gpu_perf)
                    end
                end

                runtime_ms = perf_time_ns / 1_000_000
                push!(results_runtime[name], runtime_ms)
                println("                       | Minimum Runtime: $runtime_ms ms")
            end
        end
        all_results_accuracy[func] = results_accuracy
        all_results_runtime[func] = results_runtime
    end
    return sizes, all_results_accuracy, all_results_runtime
end

function plot_accuracy_results(sizes, results, func_char::Char)
    op_name = func_char == 'S' ? "TRSM" : "TRMM"
    results_to_plot = filter(pair -> pair.first != "Pure Float64", results)
    markers = [:circle, :utriangle, :diamond, :square, :star5, :dtriangle, :hexagon, :cross, :xcross, :pentagon, :star6]
    plt = plot(
        title="$op_name Accuracy vs. Matrix Size",
        ylabel="-log10(Relative Error) [Higher is Better]",
        xlabel="Matrix Size (n x n)",
        fontfamily="Computer Modern",
        legend=:outertopright,
        size=(800, 600),
        dpi=300
    )
    for (i, (name, data)) in enumerate(results_to_plot)
        plot!(plt, sizes, data, label=name, lw=2.5, marker=markers[i], markersize=5)
    end
    return plt
end

function plot_runtime_results(sizes, results, func_char::Char)
    op_name = func_char == 'S' ? "TRSM" : "TRMM"
    markers = [:circle, :utriangle, :diamond, :square, :star5, :dtriangle, :hexagon, :cross, :xcross, :pentagon, :star6]
    plt = plot(
        title="$op_name Performance vs. Matrix Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Minimum Runtime (ms) [Lower is Better]",
        yaxis=:log,
        fontfamily="Computer Modern",
        legend=:outertopright,
        size=(800, 600),
        dpi=300
    )
    for (i, (name, data)) in enumerate(results)
        plot!(plt, sizes, data, label=name, lw=2.5, marker=markers[i], markersize=5)
    end
    return plt
end

# --- Main Execution ---
sizes, all_accuracy_results, all_runtime_results = run_all_tests()

for func in ['S', 'M']
    op_name = func == 'S' ? "trsm" : "trmm"

    # Get results for the specific function
    accuracy_results = all_accuracy_results[func]
    runtime_results = all_runtime_results[func]

    # Create and save plots
    accuracy_plot = plot_accuracy_results(sizes, accuracy_results, func)
    runtime_plot = plot_runtime_results(sizes, runtime_results, func)

    savefig(accuracy_plot, "$(op_name)_accuracy_results.png")
    savefig(runtime_plot, "$(op_name)_runtime_results.png")

    println("Plots for $op_name saved.")
end