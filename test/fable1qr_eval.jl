using CUDA
using LinearAlgebra
using Plots
using CUDA.CUSOLVER

# Include your RecursiveQR implementation
# include("recursive_qr.jl")
# using .RecursiveQR  # Assuming the module is loaded

# ==============================================================================
# --- Accuracy Helper Functions ---
# ==============================================================================

function get_accuracy_cusolver_qr(A_fp64::CuMatrix, T_prec::DataType)
    n = size(A_fp64, 1) # Assuming square for this benchmark
    A_to_factor = T_prec.(A_fp64)
    A_cpu_orig = Matrix(A_fp64)
    
    # Vendor library baseline
    _, tau = CUSOLVER.geqrf!(A_to_factor)
    
    # Extract R (upper triangular part)
    R_gpu = triu(A_to_factor)
    R_cpu = Matrix(R_gpu)
    
    # Extract explicit Q
    CUSOLVER.orgqr!(A_to_factor, tau)
    Q_cpu = Matrix(A_to_factor)
    
    # Reconstruct A = Q * R in Float64 to avoid accumulation errors in the check
    A_reconstructed = Float64.(Q_cpu) * Float64.(R_cpu)
    
    # Metrics
    rel_err = norm(A_reconstructed - A_cpu_orig) / norm(A_cpu_orig)
    orth_err = norm(Q_cpu' * Q_cpu - I) / sqrt(n)
    
    return max(rel_err, 1e-20), max(orth_err, 1e-20)
end

function get_accuracy_recursive_qr(A_fp64::CuMatrix, T_prec::DataType, nb::Int)
    n = size(A_fp64, 1)
    A_to_factor = T_prec.(A_fp64)
    A_cpu_orig = Matrix(A_fp64)
    
    # Pre-allocate tau
    tau = CuVector{T_prec}(undef, n)
    
    # Hit your recursive Elmroth-Gustavson driver
    rgeqrf!(A_to_factor, tau; nb=nb)
    
    # Extract R (upper triangular part)
    R_gpu = triu(A_to_factor)
    R_cpu = Matrix(R_gpu)
    
    # Extract explicit Q using your exported wrapper
    explicitQ!(A_to_factor, tau)
    Q_cpu = Matrix(A_to_factor)
    
    # Reconstruct A = Q * R in Float64
    A_reconstructed = Float64.(Q_cpu) * Float64.(R_cpu)
    
    # Metrics
    rel_err = norm(A_reconstructed - A_cpu_orig) / norm(A_cpu_orig)
    orth_err = norm(Q_cpu' * Q_cpu - I) / sqrt(n)
    
    return max(rel_err, 1e-20), max(orth_err, 1e-20)
end

# ==============================================================================
# --- Main Accuracy Driver ---
# ==============================================================================

function run_qr_accuracy_benchmark()
    # Matrix sizes to evaluate
    n_values = [512, 1024, 2048, 4096, 8192]

    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64
    )

    recursive_scenarios = Dict(
        "RecursiveQR F32 (nb=128)" => (Float32, 128),
        "RecursiveQR F32 (nb=256)" => (Float32, 256),
        "RecursiveQR F64 (nb=128)" => (Float64, 128),
        "RecursiveQR F64 (nb=256)" => (Float64, 256)
    )
    
    all_results = Dict{String, Vector{Float64}}()
    for name in keys(cusolver_scenarios); all_results[name] = Float64[]; end
    for name in keys(recursive_scenarios); all_results[name] = Float64[]; end

    println("="^70)
    println("Starting Recursive QR Factorization Accuracy Benchmark")
    println("="^70)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Standard normal distribution for QR
        A_cpu = randn(Float64, n, n)
        A_fp64 = CuArray(A_cpu)

        println("\n  [Vendor CUSOLVER Baselines]")
        for (name, T_prec) in cusolver_scenarios
            try
                rel_err, orth_err = get_accuracy_cusolver_qr(A_fp64, T_prec)
                push!(all_results[name], -log10(rel_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(rpad(round(rel_err, sigdigits=3), 8)) | Orth: $(round(orth_err, sigdigits=3))")
            catch e
                push!(all_results[name], NaN)
                println("    [FAILED] $(rpad(name, 17)) | Error: $(sprint(showerror, e))")
            end
        end

        println("\n  [Elmroth-Gustavson Recursive Scenarios]")
        for (name, (T_prec, nb)) in recursive_scenarios
            try
                rel_err, orth_err = get_accuracy_recursive_qr(A_fp64, T_prec, nb)
                push!(all_results[name], -log10(rel_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(rpad(round(rel_err, sigdigits=3), 8)) | Orth: $(round(orth_err, sigdigits=3))")
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
        title="QR Factorization Accuracy vs. Matrix Size",
        ylabel="-log10(Relative Error) [Higher is Better]",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300,
        margin=5Plots.mm
    )

    for (name, results) in all_results
        if any(!isnan, results)
            if occursin("CUSOLVER", name)
                marker_style = :dtriangle
                line_style = :dash
            else
                marker_style = :circle
                line_style = :solid
            end
            plot!(plt, n_values, results, label=name, lw=2, ls=line_style, marker=marker_style, ms=5)
        else
            println("Skipping trace for '$name' (all evaluations failed or produced NaN).")
        end
    end

    output_filename = "rgeqrf_accuracy.png"
    savefig(plt, output_filename)
    println("\n" * "="^70)
    println("Benchmark Complete! Plot saved as: $output_filename")
    println("="^70)
end

# Run the benchmark
run_qr_accuracy_benchmark()