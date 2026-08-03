using CUDA
using CUDA.CUSOLVER
using LinearAlgebra
using Plots
using Plots.Measures: mm

# Include your custom data structures, wrappers, and algorithm definitions
# include("wrappers.jl")
# include("matmul.jl")
# include("rectrxm.jl")
# include("recqr.jl") # <-- Ensure this filename matches where you saved the implementation

# ==============================================================================
# --- Accuracy Helper Functions ---
# ==============================================================================

function get_accuracy_cusolver_qr(A_fp64::CuMatrix, T_prec::DataType)
    n = size(A_fp64, 1)
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
    
    # Reconstruct A = Q * R in Float64
    A_reconstructed = Float64.(Q_cpu) * Float64.(R_cpu)
    
    rel_err = norm(A_reconstructed - A_cpu_orig) / norm(A_cpu_orig)
    orth_err = norm(Q_cpu' * Q_cpu - I) / sqrt(n)
    
    return max(rel_err, 1e-20), max(orth_err, 1e-20)
end

function get_accuracy_pure_qr(A_fp64::CuMatrix, T_prec::DataType, block_size::Int=256)
    n = size(A_fp64, 1)
    A_to_factor = T_prec.(A_fp64)
    A_cpu_orig = Matrix(A_fp64)
    
    # Hit your flat recursive driver
    # Note: adjust to `geqrf_recursive!` if you are using the alternative naming from your file
    # T_factor = qr_recursive!(A_to_factor, block_size)
    T_factor = geqrf_recursive!(A_to_factor, block_size)
    
    # Unpack stored Y and R to host
    A_cpu = Matrix(A_to_factor)
    R_cpu = triu(A_cpu)
    
    # Y is unit lower trapezoidal (stored strictly below diagonal)
    Y_cpu = tril(A_cpu, -1) + I
    T_cpu = Matrix(T_factor)
    
    # Q = I - Y * T * Y'
    Q_cpu = I - Y_cpu * T_cpu * Y_cpu'
    
    # Reconstruct A = Q * R in Float64
    A_reconstructed = Float64.(Q_cpu) * Float64.(R_cpu)
    
    rel_err = norm(A_reconstructed - A_cpu_orig) / norm(A_cpu_orig)
    orth_err = norm(Q_cpu' * Q_cpu - I) / sqrt(n)
    
    return max(rel_err, 1e-20), max(orth_err, 1e-20)
end

function get_accuracy_mixed_qr(A_fp64::CuMatrix, precisions::Vector, block_size::Int=256)
    n = size(A_fp64, 1)
    A_cpu_orig = Matrix(A_fp64)

    # Construct PanelMixedPrec using your dynamic quantization constructor
    A_mixed_input = PanelMixedPrec(copy(A_fp64); precisions=precisions)

    # Factorize using your overloaded mixed-precision routine
    # Note: adjust to `geqrf_recursive!` if you are using the alternative naming from your file
    # T_factor = qr_recursive_mixed!(A_mixed_input, block_size)
    T_factor = geqrf_recursive!(A_mixed_input, block_size)

    # Reconstruct dense GPU matrix using your implementation's built-in helper
    # This unpacks Y and R and applies any dequantization scaling
    A_result = reconstruct_matrix(A_mixed_input)
    A_cpu = Matrix(A_result)
    
    R_cpu = triu(A_cpu)
    
    # Y is unit lower trapezoidal
    Y_cpu = tril(A_cpu, -1) + I
    T_cpu = Matrix(T_factor)
    
    # Q = I - Y * T * Y'
    Q_cpu = I - Y_cpu * T_cpu * Y_cpu'
    
    # Reconstruct A = Q * R in Float64
    A_reconstructed = Float64.(Q_cpu) * Float64.(R_cpu)
    
    rel_err = norm(A_reconstructed - A_cpu_orig) / norm(A_cpu_orig)
    orth_err = norm(Q_cpu' * Q_cpu - I) / sqrt(n)
    
    return max(rel_err, 1e-20), max(orth_err, 1e-20)
end

# ==============================================================================
# --- Main Accuracy Driver ---
# ==============================================================================

function run_qr_accuracy_benchmark()
    n_values = [512, 1024, 2048, 4096, 8192]

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

    # Dictionaries to store both relative error and orthogonality metrics
    all_rel_err = Dict{String, Vector{Float64}}()
    all_orth_err = Dict{String, Vector{Float64}}()
    
    for name in keys(cusolver_scenarios)
        all_rel_err[name] = Float64[]
        all_orth_err[name] = Float64[]
    end
    for name in keys(pure_scenarios)
        all_rel_err[name] = Float64[]
        all_orth_err[name] = Float64[]
    end
    for name in keys(mixed_scenarios)
        all_rel_err[name] = Float64[]
        all_orth_err[name] = Float64[]
    end

    println("="^70)
    println("Starting QR Factorization (Compact-WY) Accuracy Benchmark")
    println("="^70)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Standard normal distribution for testing QR stability
        A_cpu = randn(Float64, n, n)
        A_fp64 = CuArray(A_cpu)

        println("\n  [Vendor CUSOLVER Baselines]")
        for (name, T_prec) in cusolver_scenarios
            try
                rel_err, orth_err = get_accuracy_cusolver_qr(A_fp64, T_prec)
                push!(all_rel_err[name], -log10(rel_err))
                push!(all_orth_err[name], -log10(orth_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(rpad(round(rel_err, sigdigits=3), 8)) | Orth: $(round(orth_err, sigdigits=3))")
            catch e
                push!(all_rel_err[name], NaN)
                push!(all_orth_err[name], NaN)
                println("    [FAILED] $(rpad(name, 17)) | Error: $(sprint(showerror, e))")
            end
        end

        println("\n  [Pure Recursive Scenarios]")
        for (name, T_prec) in pure_scenarios
            try
                rel_err, orth_err = get_accuracy_pure_qr(A_fp64, T_prec)
                push!(all_rel_err[name], -log10(rel_err))
                push!(all_orth_err[name], -log10(orth_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(rpad(round(rel_err, sigdigits=3), 8)) | Orth: $(round(orth_err, sigdigits=3))")
            catch e
                push!(all_rel_err[name], NaN)
                push!(all_orth_err[name], NaN)
                println("    [FAILED] $(rpad(name, 17)) | Error: $(sprint(showerror, e))")
            end
        end

        println("\n  [Mixed Recursive Scenarios]")
        for (name, prec_list) in mixed_scenarios
            try
                rel_err, orth_err = get_accuracy_mixed_qr(A_fp64, prec_list)
                push!(all_rel_err[name], -log10(rel_err))
                push!(all_orth_err[name], -log10(orth_err))
                println("    $(rpad(name, 25)) | Rel. Error: $(rpad(round(rel_err, sigdigits=3), 8)) | Orth: $(round(orth_err, sigdigits=3))")
            catch e
                push!(all_rel_err[name], NaN)
                push!(all_orth_err[name], NaN)
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
    
    # 1. Plot Relative Error
    plt_rel = plot(
        title="Compact-WY QR Rel. Error vs. Size",
        ylabel="-log10(||QR-A|| / ||A||)",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300,
        margin=5mm
    )

    for (name, results) in all_rel_err
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
            plot!(plt_rel, n_values, results, label=name, lw=2, ls=line_style, marker=marker_style, ms=5)
        end
    end

    output_filename_rel = "qr_compactwy_mixed_rel_error.png"
    savefig(plt_rel, output_filename_rel)
    
    # 2. Plot Orthogonality Error
    plt_orth = plot(
        title="Compact-WY QR Orthogonality vs. Size",
        ylabel="-log10(||Q'Q-I|| / √n)",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300,
        margin=5mm
    )

    for (name, results) in all_orth_err
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
            plot!(plt_orth, n_values, results, label=name, lw=2, ls=line_style, marker=marker_style, ms=5)
        end
    end
    
    output_filename_orth = "qr_compactwy_mixed_orthogonality.png"
    savefig(plt_orth, output_filename_orth)

    println("\n" * "="^70)
    println("Benchmark Complete!")
    println("Plots saved as: $output_filename_rel")
    println("                $output_filename_orth")
    println("="^70)
end

# Run the benchmark
run_qr_accuracy_benchmark()