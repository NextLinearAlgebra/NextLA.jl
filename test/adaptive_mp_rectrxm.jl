using Plots
using MatrixDepot    
using SparseArrays  
using BFloat16s
include("benchmark.jl")

const PRECISION_MAP = Dict(
    "q52"  => Nothing, #idk what these are
    "bf16" => BFloat16, #idk what these are
    "f16"  => Float16,
    "f32"  => Float32,
    "f64"  => Float64
)


function run_depot_benchmark()
    m = 64
    uplo = 'L'
    side = 'L'
    alpha = 1.0
    trans = 'N'

    
    matrices_to_test = [
        "HB/saylr3",               
        "DRIVCAV/cavity18",         
        "HB/psmigr_1",            
        "LeGresley/LeGresley_2508",
        "FIDAP/ex37",              
    ]

    
    test_scenarios = Dict(
        "Pure Float64"             => [Float64],
        "Pure Float32"             => [Float32],
        "Pure Float16"             => [Float16],
        "Manual: [F32, F64]"       => [Float32, Float64], 
        "Manual: [F16, F32]"       => [Float16, Float32],
        "Adaptive: eps=1e-6"       => [],
        "Adaptive: eps=1e-7"       => [],
        "Adaptive: eps=1e-8"       => [],
        "Adaptive: eps=1e-9"       => [],
        "Adaptive: eps=1e-10"       => [],
        "Adaptive: eps=1e-5"       => [],
        "Adaptive: eps=1e-4"       => [],
    )

    all_results = Dict()

    for matrix_name in matrices_to_test
        println("###   Testing Matrix: $matrix_name")

        A_raw = matrixdepot(matrix_name)
        A_cpu = Matrix(LowerTriangular(Matrix{Float64}(A_raw)))
        
        n = size(A_cpu, 1)
        println("--- Matrix Size: $n x $n ---")

        max_abs_val = maximum(abs.(A_cpu))
        is_f16_safe = max_abs_val <= 65504.0
        println("--- Max matrix value: $max_abs_val. Float16 safe: $is_f16_safe ---")

        B_cpu = rand(Float16, n, m)

        all_results[matrix_name] = Dict()

        for func in ['S', 'M']
            op_name = func == 'S' ? "TRSM" : "TRMM"
            println("\n--- Operation: $op_name ---")
            
            A_blas_gpu = CuArray(A_cpu)
            B_solution_gpu_alpha = CuArray(alpha .* B_cpu)
            if func == 'S'
                CUBLAS.trsm!(side, uplo, trans, 'N', 1.0, A_blas_gpu, B_solution_gpu_alpha)
            else # TRMM
                CUBLAS.trmm!(side, uplo, trans, 'N', 1.0, A_blas_gpu, B_solution_gpu_alpha, B_solution_gpu_alpha)
            end

            all_results[matrix_name][op_name] = Dict()

            for (name, manual_prec_list) in test_scenarios
                println("  Scenario: '$name'")
                local prec_list = manual_prec_list

                if startswith(name, "Adaptive:")
                    epsilon_str = split(name, "=")[2]
                    epsilon_target = parse(Float64, epsilon_str)
                    n_min = n > 256 ? div(n, 2^5) : 16

                    precision_strings = adaptive_precision_LT(A_cpu, [3, 4, 5], n_min, epsilon_target)
                    
                    prec_list = [PRECISION_MAP[s] for s in precision_strings]
                    push!(prec_list, Float32) # Add final FP64 for the base case

                    println("    -> Adaptively selected precisions: ", [string(p) for p in prec_list])
                end

                if isempty(prec_list)
                    println("    -> SKIPPING: No precisions defined.")
                    continue
                end

                # if !is_f16_safe && (Float16 in prec_list)
                #     println("      -> SKIPPING: Matrix values exceed Float16 range for this scenario.")
                #     all_results[matrix_name][op_name][name] = (NaN, NaN)
                #     continue
                # end

                T_Base = prec_list[end]
                A_mixed_gpu = TriMixedPrec(CuArray(A_cpu), uplo; precisions=prec_list)
                B_gpu = CuArray{T_Base}(B_cpu)
                unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed_gpu, B_gpu)

                error_norm = norm(CuArray{Float64}(B_gpu) .- B_solution_gpu_alpha)
                solution_norm = norm(B_solution_gpu_alpha)
                relative_error = max(error_norm / solution_norm, 1e-20) # Avoid error of 0 for log plot
                println("      Relative Error: $relative_error")

                A_mixed_perf = TriMixedPrec(CuArray(A_cpu), uplo; precisions=prec_list)
                B_gpu_perf = CuArray{T_Base}(B_cpu)
                # Create a clean version of B to reset the data in each benchmark iteration
                B_clean_copy = copy(B_gpu_perf)

                backend = get_backend(A_blas_gpu)

                # Use a 'do'-block to pass the operation to our manual benchmark function
                perf_time_ns = run_manual_benchmark(backend) do
                    # Inside the benchmark loop, always reset B_gpu_perf to its original state
                    copyto!(B_gpu_perf, B_clean_copy)
                    # The actual function call to be timed
                    unified_rectrxm!(side, uplo, trans, alpha, func, A_mixed_perf, B_gpu_perf)
                end

                runtime_ms = perf_time_ns / 1_000_000
                println("      Minimum Runtime: $runtime_ms ms")

                all_results[matrix_name][op_name][name] = (relative_error, runtime_ms)
            end
        end
    end
    return all_results
end



function plot_benchmark_results_vs_epsilon(all_results)
    println("\n--- Generating Plots ---")

    for (matrix_name, matrix_data) in all_results
        clean_matrix_name = replace(matrix_name, "/" => "-") # Sanitize for filename
        
        for (op_name, op_data) in matrix_data
            println("  Plotting for $matrix_name - $op_name...")

            # --- Separate data into adaptive and baseline scenarios ---
            adaptive_epsilons = []
            adaptive_accuracies = []
            adaptive_runtimes = []
            baseline_data = Dict()

            for (scenario, values) in op_data
                if startswith(scenario, "Adaptive:")
                    epsilon_str = split(scenario, "=")[2]
                    epsilon_val = parse(Float64, epsilon_str)
                    push!(adaptive_epsilons, epsilon_val)
                    push!(adaptive_accuracies, -log10(values[1]))
                    push!(adaptive_runtimes, values[2])
                else
                    baseline_data[scenario] = (-log10(values[1]), values[2])
                end
            end
            
            # Sort the adaptive data by epsilon to draw lines correctly
            p = sortperm(adaptive_epsilons)
            adaptive_epsilons = adaptive_epsilons[p]
            adaptive_accuracies = adaptive_accuracies[p]
            adaptive_runtimes = adaptive_runtimes[p]
            
            # --- Create Accuracy vs. Epsilon Plot ---
            acc_plot = plot(
                title = "$matrix_name\n$op_name Accuracy vs. Epsilon",
                xlabel = "Epsilon (Target Error)",
                ylabel = "-log10(Relative Error) [Higher is Better]",
                xaxis = :log,
                legend = :outertopright,
                size = (900, 700),
                dpi = 300
            )
            
            # Plot the main adaptive method line
            plot!(acc_plot, adaptive_epsilons, adaptive_accuracies, label="Adaptive Method", lw=3, marker=:circle)

            # Plot horizontal lines for baselines
            for (name, values) in baseline_data
                hline!(acc_plot, [values[1]], label=name, linestyle=:dash)
            end
            
            savefig(acc_plot, "$(clean_matrix_name)_$(op_name)_Accuracy_vs_Epsilon.png")

            # --- Create Performance vs. Epsilon Plot ---
            perf_plot = plot(
                title = "$matrix_name\n$op_name Performance vs. Epsilon",
                xlabel = "Epsilon (Target Error)",
                ylabel = "Minimum Runtime (ms) [Lower is Better]",
                xaxis = :log,
                yaxis = :log,
                legend = :outertopright,
                size = (900, 700),
                dpi = 300
            )

            # Plot the main adaptive method line
            plot!(perf_plot, adaptive_epsilons, adaptive_runtimes, label="Adaptive Method", lw=3, marker=:circle)

            # Plot horizontal lines for baselines
            for (name, values) in baseline_data
                hline!(perf_plot, [values[2]], label=name, linestyle=:dash)
            end

            savefig(perf_plot, "$(clean_matrix_name)_$(op_name)_Performance_vs_Epsilon.png")
        end
    end
    println("\nAll plots have been saved to the current directory.")
end

# ===================================================================
# MAIN EXECUTION BLOCK
# ===================================================================

# 1. Run all the benchmarks
results = run_depot_benchmark()

# 2. Generate and save plots from the results
plot_benchmark_results_vs_epsilon(results)