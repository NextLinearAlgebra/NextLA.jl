using CUDA
using CUDA.CUSOLVER
using LinearAlgebra
using Plots
using Plots.Measures: mm
using StochasticRounding

# Include your custom data structures, wrappers, and algorithm definitions
# include("wrappers.jl")
# include("matmul.jl")
# include("rectrxm.jl")
# include("fullmixedprec.jl") # <-- Ensure this matches your implementation filename

# ==============================================================================
# --- Accuracy Helper Functions ---
# ==============================================================================

function get_accuracy_cusolver_lu(A_fp64::CuMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_fp64)
    
    # Vendor library baseline (non-pivoting / standard LU)
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

function get_accuracy_pure_lu(A_fp64::CuMatrix, T_prec::DataType, block_size::Int=256)
    A_to_factor = T_prec.(A_fp64)
    
    # Hit your flat recursive driver
    getrf_recursive!(A_to_factor, block_size)
    
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
    # Construct FullMixedPrec using your dynamic quantization constructor
    A_mixed_input = FullMixedPrec(copy(A_fp64); precisions=precisions)

    # Factorize using your overloaded mixed-precision routine
    getrf_recursive!(A_mixed_input)

    # Reconstruct dense GPU matrix using your implementation's built-in helper (§5)
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

# ==============================================================================
# --- Main Accuracy Driver ---
# ==============================================================================

function run_lu_accuracy_benchmark()
    n_values = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64
    )

    pure_scenarios = Dict(
        "Pure16" => Float16,
        "Pure32" => Float32,
        "Pure64" => Float64
    )
    
    mixed_scenarios = Dict(
        # Shallow mixed
        "[F16, F32]"                => [Float16, Float32],
        "[F32, F64]"                => [Float32, Float64],
        "[F16, F64]"                => [Float16, Float64],
        
        # Deep F16 leaves
        "[F16, F16, F32]"           => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        
        # Smooth gradients
        "[F16, F16, F32, F64]"      => [Float16, Float16, Float32, Float64],
        "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64]
    )

    all_results = Dict{String, Vector{Float64}}()
    for name in keys(cusolver_scenarios); all_results[name] = Float64[]; end
    for name in keys(pure_scenarios);     all_results[name] = Float64[]; end
    for name in keys(mixed_scenarios);    all_results[name] = Float64[]; end

    println("="^60)
    println("Starting LU Factorization Accuracy Benchmark")
    println("="^60)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Diagonally dominant matrix ensures stability for unpivoted LU
        A_cpu = rand(Float64, n, n)
        A_cpu .+= Diagonal(fill(n * 2.0, n))
        A_fp64 = CuArray(A_cpu)

        println("\n  [Vendor CUSOLVER Baselines]")
        for (name, T_prec) in cusolver_scenarios
            try
                rel_err = get_accuracy_cusolver_lu(A_fp64, T_prec)
                push!(all_results[name], -log10(rel_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(round(rel_err, sigdigits=3))")
            catch e
                push!(all_results[name], NaN)
                println("    [FAILED] $(rpad(name, 17)) | Error: $(sprint(showerror, e))")
            end
        end

        println("\n  [Pure Recursive Scenarios]")
        for (name, T_prec) in pure_scenarios
            try
                rel_err = get_accuracy_pure_lu(A_fp64, T_prec)
                push!(all_results[name], -log10(rel_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(round(rel_err, sigdigits=3))")
            catch e
                push!(all_results[name], NaN)
                println("    [FAILED] $(rpad(name, 17)) | Error: $(sprint(showerror, e))")
            end
        end

        println("\n  [Mixed Recursive Scenarios]")
        for (name, prec_list) in mixed_scenarios
            try
                rel_err = get_accuracy_mixed_lu(A_fp64, prec_list)
                push!(all_results[name], -log10(rel_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(round(rel_err, sigdigits=3))")
            catch e
                push!(all_results[name], NaN)
                println("    [FAILED] $(rpad(name, 17)) | Error: $(sprint(showerror, e))")
            end
        end

        # Clean up GPU memory between size iterations
        A_cpu = nothing; A_fp64 = nothing
        GC.gc(true); CUDA.reclaim()
    end

    # ==========================================================================
    # --- Plotting Results ---
    # ==========================================================================
    plt = plot(
        title="LU Factorization Accuracy vs. Matrix Size",
        ylabel="-log10(Relative Error) [Higher is Better]",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300,
        margin=5mm
    )

    for (name, results) in all_results
        if any(!isnan, results)
            if occursin("CUSOLVER", name)
                marker_style = :dtriangle
                line_style = :dash
            elseif occursin("Pure", name)
                marker_style = :square
                line_style = :dot
            else
                marker_style = :circle
                line_style = :solid
            end
            plot!(plt, n_values, results, label=name, lw=2, ls=line_style, marker=marker_style, ms=5)
        else
            println("Skipping trace for '$name' (all evaluations failed or produced NaN).")
        end
    end

    output_filename = "getrf_recursive_accuracy.png"
    savefig(plt, output_filename)
    println("\n" * "="^60)
    println("Benchmark Complete! Plot saved as: $output_filename")
    println("="^60)
end

# Run the benchmark
run_lu_accuracy_benchmark()