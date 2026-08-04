using CUDA
using CUDA.CUSOLVER
using LinearAlgebra
using Plots
using KernelAbstractions
using StochasticRounding

# Include the file where you saved this latest qr_recursive! / qr_recursive_mixed! implementation
# include("recqr.jl") # <-- Ensure this filename matches your implementation file
include("benchmark.jl")         # Must contain run_single_benchmark

# ==============================================================================
# --- Config ---
# ==============================================================================

# _form_T is O(k^3) on a single CPU core for the base-case panel width k, so a
# 5x-repeat benchmark loop on a multi-minute op is brutal. Drop this to 1 while
# you're just trying to get a number; bump back up once you trust the timing.
const REPEATS = 3

# ==============================================================================
# --- Performance Timing Helper Functions ---
# ==============================================================================

function benchmark_op(op, reset_op, backend; label::String="")
    reset_op()
    t0 = time()
    op()
    KernelAbstractions.synchronize(backend)
    println("        [warm-up call finished in $(round(time() - t0, digits=2)) s]")

    min_time_ns = Inf
    for i in 1:REPEATS
        reset_op()
        t0 = time()
        time_ns = run_single_benchmark(op, backend)
        println("        [$label rep $i/$REPEATS: $(round(time_ns / 1_000_000, digits=2)) ms  (wall $(round(time() - t0, digits=2)) s)]")
        min_time_ns = min(min_time_ns, time_ns)
    end

    return min_time_ns
end

function run_time_cusolver_qr(A_fp32::CuMatrix{Float32})
    backend = KernelAbstractions.get_backend(A_fp32)

    A_work  = copy(A_fp32)
    A_clean = copy(A_fp32)

    # Vendor baseline
    op       = () -> CUSOLVER.geqrf!(A_work)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend; label="CUSOLVER F32")
    return time_ns / 1_000_000 # Convert to milliseconds
end

function run_time_pure_qr(A_fp32::CuMatrix{Float32}, block_size::Int=4096)
    backend = KernelAbstractions.get_backend(A_fp32)

    A_work  = copy(A_fp32)
    A_clean = copy(A_fp32)

    # Hits qr_recursive!(A::AbstractMatrix, block_size), routing to CUSOLVER at leaf
    op       = () -> qr_recursive!(A_work, block_size)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend; label="Pure F32 (block=$block_size)")
    return time_ns / 1_000_000
end

function run_time_mixed_qr(A_fp32::CuMatrix{Float32}, precisions::Vector{DataType}, block_size::Int=4096)
    backend = KernelAbstractions.get_backend(A_fp32)
    local A_mixed_input

    # Construct PanelMixedPrec tree hierarchy during the reset phase so dynamic
    # FP16 quantization, stochastic rounding, and structure allocation are excluded from QR kernel timing
    reset_op = () -> begin
        A_mixed_input = PanelMixedPrec(copy(A_fp32); precisions=precisions)
        KernelAbstractions.synchronize(backend)
    end

    # Hits qr_recursive_mixed!(A::PanelMixedPrec)
    op = () -> qr_recursive_mixed!(A_mixed_input, block_size)

    time_ns = benchmark_op(op, reset_op, backend; label=join(string.(precisions), ","))
    return time_ns / 1_000_000
end

# ==============================================================================
# --- Main Timing Driver ---
# ==============================================================================

function check_qr_time_hierarchical()
    # 2k to 16k captures asymptotic GPU Tensor Core scaling without excessive runtimes
    # (Note: O(N^3) scaling means 65536 might take considerable time for QR)
    n_values = [2048, 4096, 8192, 16384, 32768, 65536]

    # Map your specific target scenarios to precision vectors
    mixed_scenarios = Dict(
        "Mixed [F16, F16, F32]"                => [Float16, Float16, Float32],
        "Mixed [F16, F16, F16, F32]"           => [Float16, Float16, Float16, Float32],
        "Mixed [F16, F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float16, Float32],
        "Mixed [F16, F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float16, Float32]
    )

    all_results = Dict()
    all_results["CUSOLVER F32"] = Float64[]
    all_results["Pure F32 (block=4096)"] = Float64[]
    for name in keys(mixed_scenarios)
        all_results[name] = Float64[]
    end

    println("="^60)
    println("Starting qr_recursive_mixed! Hierarchical Benchmark")
    println("REPEATS = $REPEATS per config")
    println("="^60)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")

        # Standard normal distribution is ideal for testing QR performance/stability
        A_cpu = randn(Float32, n, n)
        A_fp32 = CuArray(A_cpu)

        # 1. Benchmark CUSOLVER Baseline
        runtime_ms = run_time_cusolver_qr(A_fp32)
        push!(all_results["CUSOLVER F32"], runtime_ms)
        println("    $(rpad("CUSOLVER F32", 36)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")

        # 2. Benchmark Pure Float32 Baseline
        # NOTE: this exercises qr_recursive!'s own base case _qr_base! -> _form_T.
        # _form_T is O(k^3) on CPU in k = min(n, block_size), so block_size here
        # directly controls the worst-case panel width hitting that CPU recurrence.
        # block_size=4096 means for n<=4096 the WHOLE matrix goes through _form_T
        # in one shot (that's what stalled). Left at 256 so it actually recurses.
        runtime_ms = run_time_pure_qr(A_fp32, 256)
        push!(all_results["Pure F32 (block=4096)"], runtime_ms)
        println("    $(rpad("Pure F32 (block=4096)", 36)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")

        # 3. Benchmark Mixed-Precision Hierarchies
        # NOTE: block_size=4096 here too -> at n=4096 and above, the trailing
        # base-case QR calls inside qr_recursive_mixed! can still hit _form_T
        # with k up to 4096. Expect these to be slow at n>=4096 for the same reason.
        for (name, prec_list) in mixed_scenarios
            runtime_ms = run_time_mixed_qr(A_fp32, prec_list, 4096)
            push!(all_results[name], runtime_ms)
            println("    $(rpad(name, 36)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        # Aggressive memory cleanup between scaling steps to prevent VRAM fragmentation
        A_cpu = nothing; A_fp32 = nothing; GC.gc(true); CUDA.reclaim()
    end

    # ==========================================================================
    # --- Plotting Results ---
    # ==========================================================================
    plt = plot(
        title="Hierarchical Block QR Performance vs. Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300
    )

    for (name, results) in all_results
        if occursin("CUSOLVER", name)
            linestyle = :dot
            marker_style = :dtriangle
        elseif occursin("Pure", name)
            linestyle = :dash
            marker_style = :square
        else
            linestyle = :solid
            marker_style = :circle
        end
        plot!(plt, n_values[1:length(results)], results, label=name, lw=2, linestyle=linestyle, marker=marker_style)
    end

    output_filename = "qr_recursive_mixed_runtimes.png"
    savefig(plt, output_filename)
    println("\nPlot saved as $output_filename")
end

check_qr_time_hierarchical()