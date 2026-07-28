using CUDA
using CUDA.CUBLAS
using LinearAlgebra
using Plots
using KernelAbstractions

# Include your L1 implementation (or paste your L1 functions above this line)
# include("codex_l1.jl") 
include("benchmark.jl") # Must contain run_single_benchmark

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

function run_time_l1_lu(A_fp32::CuMatrix{Float32}, tlow::DataType, leaf_size::Int)
    backend = KernelAbstractions.get_backend(A_fp32)

    A_work = copy(A_fp32)
    A_clean = copy(A_fp32)
    
    op = () -> lu_nopiv_recursive_mixed!(A_work; leaf=leaf_size, Tlow=tlow, check=false)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000 # Convert to milliseconds
end

# ==============================================================================
# --- Main Timing Driver ---
# ==============================================================================

function check_lu_time_l1()
    # 2k to 16k captures asymptotic GPU performance without excessive runtimes
    n_values = [2048, 4096, 8192, 16384]

    # Map your target configs to (Tlow_precision, leaf_block_size)
    # Since L1 halves dimensions at each step, different leaf sizes control how many 
    # recursive levels execute before hitting the base case.
    scenarios = Dict(
        "Pure F32 (leaf=256)"           => (Float32, 256),
        "Mixed [F16, F16, F32] (leaf=512)" => (Float16, 512),
        "Mixed [F16, F16, F16, F32] (leaf=256)" => (Float16, 256)
    )

    all_results = Dict()
    for name in keys(scenarios)
        all_results[name] = Float64[]
    end

    println("="^60)
    println("Starting L1 LU Performance Benchmark (2k - 16k)")
    println("="^60)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Diagonally dominant matrix ensures conditioning & stability without pivoting
        A_cpu = rand(Float32, n, n)
        A_cpu .+= Diagonal(fill(Float32(n * 2.0), n))
        A_fp32 = CuArray(A_cpu)

        for (name, (tlow, leaf_sz)) in scenarios
            runtime_ms = run_time_l1_lu(A_fp32, tlow, leaf_sz)
            push!(all_results[name], runtime_ms)
            println("    $(rpad(name, 38)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        A_cpu = nothing; A_fp32 = nothing; GC.gc(true); CUDA.reclaim()
    end

    # Plotting
    plt = plot(
        title="L1 LU Factorization Performance vs. Matrix Size",
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

    savefig(plt, "l1_lu_runtime_results.png")
    println("\nPlot saved as l1_lu_runtime_results.png")
end

check_lu_time_l1()