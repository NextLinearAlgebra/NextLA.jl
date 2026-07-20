using CUDA
using LinearAlgebra
using Plots

# Include your custom data structures and algorithm definitions
# include("wrappers.jl")
# include("rectrxm.jl")
# include("your_mixed_prec_code.jl")

# ==============================================================================
# --- Accuracy Helper Functions ---
# ==============================================================================

"""
Unpacks a hierarchical FullMixedPrec matrix back into a dense CPU Matrix,
strictly preserving the host parametric type T_Base to prevent precision loss.
"""
function reconstruct_matrix(A::FullMixedPrec{T_Base}) where {T_Base}
    n, m = size(A)

    if A.BaseCase !== nothing
        base = Array(A.BaseCase)
        if A.base_scale !== nothing
            return T_Base(A.base_scale) .* T_Base.(base)
        else
            return T_Base.(base)
        end
    end

    n1 = size(A.A11, 1)

    A11 = reconstruct_matrix(A.A11)
    A22 = reconstruct_matrix(A.A22)

    A12 = T_Base.(Array(A.A12))
    A21 = T_Base.(Array(A.A21))

    if A.A12_scale !== nothing
        A12 .*= T_Base(A.A12_scale)
    end

    if A.A21_scale !== nothing
        A21 .*= T_Base(A.A21_scale)
    end

    result = zeros(T_Base, n, m)

    result[1:n1, 1:n1] .= A11
    result[1:n1, n1+1:end] .= A12
    result[n1+1:end, 1:n1] .= A21
    result[n1+1:end, n1+1:end] .= A22

    return result
end

function get_accuracy_pure_lu(A_fp64::CuMatrix, T_prec::DataType, block_size::Int=2048)
    A_to_factor = T_prec.(A_fp64)
    
    # Hit your flat recursive driver directly
    lu_recursive!(A_to_factor, block_size)
    
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
    # Make a copy for factorization using your constructor
    A_mixed_input = FullMixedPrec(copy(A_fp64); precisions=precisions)

    # Factorize using your overloaded recursive mixed-precision routine
    lu_recursive!(A_mixed_input)

    # Reconstruct matrix using the parametric unpacking helper
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

function get_accuracy_cusolver_lu(A_fp64::CuMatrix, T_prec::DataType)
    A_to_factor = T_prec.(A_fp64)
    
    # Vendor library baseline
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

# ==============================================================================
# --- Main Accuracy Driver ---
# ==============================================================================

function check_lu_accuracy()
    n_values = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    pure_scenarios = Dict(
        "Pure16" => Float16,
        "Pure32" => Float32,
        "Pure64" => Float64
    )
    
    cusolver_scenarios = Dict(
        "CUSOLVER F32" => Float32,
        "CUSOLVER F64" => Float64
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

    all_results = Dict()
    for name in keys(pure_scenarios);     all_results[name] = Float64[]; end
    for name in keys(cusolver_scenarios); all_results[name] = Float64[]; end
    for name in keys(mixed_scenarios);    all_results[name] = Float64[]; end

    println("="^50)
    println("Starting LU Accuracy Benchmark")
    println("="^50)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Diagonally dominant matrix for stable unpivoted LU
        A_cpu = rand(Float64, n, n)
        A_cpu .+= Diagonal(fill(n * 2.0, n))
        A_fp64 = CuArray(A_cpu)

        println("\n  --- Standard CUSOLVER Scenarios ---")
        for (name, T_prec) in cusolver_scenarios
            rel_err = get_accuracy_cusolver_lu(A_fp64, T_prec)
            push!(all_results[name], -log10(rel_err))
            println("    $name | Rel. Error: $(round(rel_err, sigdigits=3))")
        end

        println("\n  --- Pure Recursive Scenarios ---")
        for (name, T_prec) in pure_scenarios
            rel_err = get_accuracy_pure_lu(A_fp64, T_prec)
            push!(all_results[name], -log10(rel_err))
            println("    $name | Rel. Error: $(round(rel_err, sigdigits=3))")
        end

        println("\n  --- Mixed Recursive Scenarios ---")
        for (name, prec_list) in mixed_scenarios
            rel_err = get_accuracy_mixed_lu(A_fp64, prec_list)
            push!(all_results[name], -log10(rel_err))
            println("    $name | Rel. Error: $(round(rel_err, sigdigits=3))")
        end

        A_cpu = nothing; A_fp64 = nothing; GC.gc(true); CUDA.reclaim()
    end

    # Plotting
    plt = plot(
        title="LU Factorization Accuracy vs. Matrix Size",
        ylabel="-log10(Relative Error) [Higher is Better]",
        xlabel="Matrix Size (n x n)",
        xaxis=:log2,
        legend=:outertopright,
        size=(1000, 700),
        dpi=300
    )

    for (name, results) in all_results
        if occursin("CUSOLVER", name)
            marker_style = :dtriangle
        elseif occursin("Pure", name)
            marker_style = :square
        else
            marker_style = :circle
        end
        plot!(plt, n_values, results, label=name, lw=2, marker=marker_style)
    end

    savefig(plt, "lu_accuracy_results.png")
    println("\nPlot saved as lu_accuracy_results.png")
end

check_lu_accuracy()