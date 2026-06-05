using CUDA
using LinearAlgebra
using Plots
using NextLA  # Assuming NextLA or DLA exports the necessary types

include("benchmark.jl")

function get_accuracy_pure_lu(A_fp64::CuMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_fp64)
    n = size(A_to_factor, 1)
    
    # We use CUSOLVER.getrf! as the pure LU base case for standard matrices
    CUSOLVER.getrf!(A_to_factor)
    
    # Reconstruct A = L * U
    A_cpu = Matrix(A_to_factor)
    L = tril(A_cpu, -1) + I
    U = triu(A_cpu)
    
    A_reconstructed = Float64.(L * U)
    
    error_norm = norm(A_reconstructed - Matrix(A_fp64))
    orig_norm = norm(Matrix(A_fp64))
    
    return max(error_norm / orig_norm, 1e-20)
end

function get_accuracy_mixed_lu(A_fp64::CuMatrix, precisions::Vector)
    n = size(A_fp64, 1)
    
    # Make a copy for factorization
    A_mixed_input = FullMixedPrec(copy(A_fp64); precisions=precisions)

    # Threshold where we stop recursion and hit the base case (e.g. CUSOLVER)
    threshold = 256
    lu_recursive_mixed!(A_mixed_input, threshold)

    # Reconstruct matrix
    A_result = reconstruct_matrix(A_mixed_input)
    A_cpu = Matrix(A_result)
    
    # Reconstruct A = L * U
    L = tril(A_cpu, -1) + I
    U = triu(A_cpu)
    
    A_reconstructed = Float64.(L * U)
    
    error_norm = norm(A_reconstructed - Matrix(A_fp64))
    orig_norm = norm(Matrix(A_fp64))
    
    return max(error_norm / orig_norm, 1e-20)
end

function check_lu_accuracy()
    n_values = [512, 1024, 2048, 4096]

    pure_scenarios = Dict(
        "Pure F32" => [Float32],
        "Pure F64" => [Float64],
        "Pure F16" => [Float16]
    )
    
    mixed_scenarios = Dict(
        "[F16, F32]"                => [Float16, Float32],
        "[F32, F64]"                => [Float32, Float64],
        "[F16, F64]"                => [Float16, Float64],
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F32, F64]"      => [Float16, Float16, Float32, Float64]
    )

    all_results = Dict()
    for name in keys(pure_scenarios)
        all_results[name] = Float64[]
    end
    for name in keys(mixed_scenarios)
        all_results[name] = Float64[]
    end

    println("="^50)
    println("Starting LU Accuracy Benchmark")
    println("="^50)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Diagonally dominant matrix for stable unpivoted LU
        A_cpu = rand(Float64, n, n)
        A_cpu .+= Diagonal(fill(n * 2.0, n))
        A_fp64 = CuArray(A_cpu)

        for (name, prec_list) in pure_scenarios
            T_prec = prec_list[1]
            rel_err = get_accuracy_pure_lu(A_fp64, T_prec)
            push!(all_results[name], -log10(rel_err))
            println("  $name | Rel. Error: $(round(rel_err, sigdigits=3))")
        end

        for (name, prec_list) in mixed_scenarios
            rel_err = get_accuracy_mixed_lu(A_fp64, prec_list)
            push!(all_results[name], -log10(rel_err))
            println("  $name | Rel. Error: $(round(rel_err, sigdigits=3))")
        end
    end

    # Plotting
    plt = plot(
        title="LU Factorization Accuracy vs. Matrix Size",
        ylabel="-log10(Relative Error) [Higher is Better]",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        legend=:outertopright,
        size=(800, 600),
        dpi=300
    )

    for (name, results) in all_results
        marker_style = occursin("Pure", name) ? :square : :circle
        plot!(plt, n_values, results, label=name, lw=2, marker=marker_style)
    end

    savefig(plt, "lu_accuracy_results.png")
    println("\nPlot saved as lu_accuracy_results.png")
end

check_lu_accuracy()
