using CUDA
using CUDA.CUSOLVER
using LinearAlgebra
using Plots
using KernelAbstractions
using StochasticRounding

# Include the file where you saved this latest getrf_recursive! / getrf_npvt! implementation
# include("lu_fullmixed_npvt.jl") # <-- Ensure this filename matches your implementation file
include("benchmark.jl")         # Must contain run_single_benchmark

# ==============================================================================
# --- Performance Timing Helper Functions ---
# ==============================================================================

function benchmark_op(op, reset_op, backend)
    reset_op()
    op()
    KernelAbstractions.synchronize(backend)

    min_time_ns = Inf
    for _ in 1:5
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    
    return min_time_ns
end

function run_time_pure_lu(A_fp32::CuMatrix{Float32}, block_size::Int=4096)
    backend = KernelAbstractions.get_backend(A_fp32)

    A_work  = copy(A_fp32)
    A_clean = copy(A_fp32)
    
    # Hits getrf_recursive!(A::AbstractMatrix, block_size), routing to getrf_npvt! at leaf
    op       = () -> getrf_recursive!(A_work, block_size)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000 # Convert to milliseconds
end

function run_time_mixed_lu(A_fp32::CuMatrix{Float32}, precisions::Vector{DataType})
    backend = KernelAbstractions.get_backend(A_fp32)
    local A_mixed_input
    
    # Construct FullMixedPrec tree hierarchy during the reset phase so dynamic 
    # FP16 quantization, stochastic rounding, and structure allocation are excluded from LU kernel timing
    reset_op = () -> begin
        A_mixed_input = FullMixedPrec(copy(A_fp32); precisions=precisions)
        KernelAbstractions.synchronize(backend)
    end
    
    # Hits getrf_recursive!(A::FullMixedPrec), utilizing TriMixedPrec extraction and -Float64(s21 * s12) scale folding
    op = () -> getrf_recursive!(A_mixed_input)
    
    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000
end

# ==============================================================================
# --- Main Timing Driver ---
# ==============================================================================

function check_lu_time_hierarchical()
    # 2k to 16k captures asymptotic GPU Tensor Core scaling without excessive runtimes
    n_values = [2048, 4096, 8192, 16384]

    # Map your specific target scenarios to precision vectors
    mixed_scenarios = Dict(
        "Mixed [F16, F16, F32]"      => [Float16, Float16, Float32],
        "Mixed [F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32]
    )

    all_results = Dict()
    all_results["Pure F32 (block=4096)"] = Float64[]
    for name in keys(mixed_scenarios)
        all_results[name] = Float64[]
    end

    println("="^60)
    println("Starting getrf_npvt! Hierarchical Benchmark (2k - 16k)")
    println("="^60)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Diagonally dominant matrix ensures stability for non-pivoted LU
        A_cpu = rand(Float32, n, n)
        A_cpu .+= Diagonal(fill(Float32(n * 2.0), n))
        A_fp32 = CuArray(A_cpu)

        # 1. Benchmark Pure Float32 Baseline (block_size = 4096 matches BaseCase threshold)
        runtime_ms = run_time_pure_lu(A_fp32, 4096)
        push!(all_results["Pure F32 (block=4096)"], runtime_ms)
        println("    $(rpad("Pure F32 (block=4096)", 32)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")

        # 2. Benchmark Mixed-Precision Hierarchies
        for (name, prec_list) in mixed_scenarios
            runtime_ms = run_time_mixed_lu(A_fp32, prec_list)
            push!(all_results[name], runtime_ms)
            println("    $(rpad(name, 32)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        # Aggressive memory cleanup between scaling steps to prevent VRAM fragmentation
        A_cpu = nothing; A_fp32 = nothing; GC.gc(true); CUDA.reclaim()
    end

    # ==========================================================================
    # --- Plotting Results ---
    # ==========================================================================
    plt = plot(
        title="Non-Pivoting Hierarchical LU Performance vs. Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300
    )

    for (name, results) in all_results
        if occursin("Pure", name)
            linestyle = :dash
            marker_style = :square
        else
            linestyle = :solid
            marker_style = :circle
        end
        plot!(plt, n_values, results, label=name, lw=2, linestyle=linestyle, marker=marker_style)
    end

    savefig(plt, "getrf_recursive_npvt_runtimes.png")
    println("\nPlot saved as getrf_recursive_npvt_runtimes.png")
end

check_lu_time_hierarchical()