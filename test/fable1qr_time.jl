using CUDA
using CUDA.CUSOLVER
using LinearAlgebra
using Plots
using KernelAbstractions

# Include the file where you saved the RecursiveQR module
# include("recursive_qr.jl") # <-- Ensure this matches your implementation file
# using .RecursiveQR         # Import the module explicitly

include("benchmark.jl")      # Must contain run_single_benchmark

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

function run_time_cusolver_qr(A_orig::CuMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(A_orig)

    A_work  = copy(A_orig)
    A_clean = copy(A_orig)
    
    # Vendor baseline: allocates tau internally or overwrites it
    op       = () -> CUSOLVER.geqrf!(A_work)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000 # Convert to milliseconds
end

function run_time_recursive_qr(A_orig::CuMatrix{T}, nb::Int) where {T}
    backend = KernelAbstractions.get_backend(A_orig)

    A_work  = copy(A_orig)
    A_clean = copy(A_orig)
    tau     = CuVector{T}(undef, min(size(A_orig)...))
    
    # Hits rgeqrf!(A::StridedCuMatrix, tau; nb)
    op       = () -> rgeqrf!(A_work, tau; nb=nb)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000
end

# ==============================================================================
# --- Main Timing Driver ---
# ==============================================================================

function check_recursive_qr_runtimes()
    # 2k to 16k is a sweet spot for capturing GPU compute saturation.
    # Note: QR requires 4/3 N^3 FLOPS, so scaling past 16k/32k takes considerable time.
    n_values = [2048, 4096, 8192, 16384, 32768]

    # Map your specific target scenarios
    scenarios = Dict(
        "RecursiveQR F32 (nb=128)" => (Float32, 128),
        "RecursiveQR F32 (nb=256)" => (Float32, 256),
        "RecursiveQR F64 (nb=128)" => (Float64, 128),
        "RecursiveQR F64 (nb=256)" => (Float64, 256)
    )

    all_results = Dict()
    all_results["CUSOLVER F32"] = Float64[]
    all_results["CUSOLVER F64"] = Float64[]
    for name in keys(scenarios)
        all_results[name] = Float64[]
    end

    println("="^65)
    println("Starting RecursiveQR (Dense RGEQR3) Runtime Benchmark")
    println("="^65)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Float32 Setup
        A_cpu_f32 = randn(Float32, n, n)
        A_gpu_f32 = CuArray(A_cpu_f32)

        # Float64 Setup
        A_cpu_f64 = randn(Float64, n, n)
        A_gpu_f64 = CuArray(A_cpu_f64)

        # 1. Benchmark CUSOLVER Baselines
        time_cusolver_f32 = run_time_cusolver_qr(A_gpu_f32)
        push!(all_results["CUSOLVER F32"], time_cusolver_f32)
        println("    $(rpad("CUSOLVER F32", 30)) | Runtime: $(round(time_cusolver_f32, sigdigits=4)) ms")

        time_cusolver_f64 = run_time_cusolver_qr(A_gpu_f64)
        push!(all_results["CUSOLVER F64"], time_cusolver_f64)
        println("    $(rpad("CUSOLVER F64", 30)) | Runtime: $(round(time_cusolver_f64, sigdigits=4)) ms")

        # 2. Benchmark Recursive Scenarios
        for (name, (T_prec, nb)) in scenarios
            if T_prec == Float32
                runtime_ms = run_time_recursive_qr(A_gpu_f32, nb)
            else
                runtime_ms = run_time_recursive_qr(A_gpu_f64, nb)
            end
            push!(all_results[name], runtime_ms)
            println("    $(rpad(name, 30)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        # Aggressive memory cleanup
        A_cpu_f32 = nothing; A_gpu_f32 = nothing
        A_cpu_f64 = nothing; A_gpu_f64 = nothing
        GC.gc(true); CUDA.reclaim()
    end

    # ==========================================================================
    # --- Plotting Results ---
    # ==========================================================================
    plt = plot(
        title="RecursiveQR Performance vs. CUSOLVER Base",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        legend=:outertopleft,
        size=(1050, 700),
        dpi=300
    )

    for (name, results) in all_results
        if occursin("CUSOLVER", name)
            linestyle = :dash
            marker_style = :dtriangle
            lw = 2
        else
            linestyle = :solid
            marker_style = :circle
            lw = 2
        end
        
        # Visually separate F32 and F64 curves by assigning distinct marker colors
        color = occursin("F64", name) ? :auto : :auto
        
        plot!(plt, n_values, results, label=name, lw=lw, linestyle=linestyle, marker=marker_style)
    end

    output_filename = "recursive_qr_runtimes.png"
    savefig(plt, output_filename)
    println("\nPlot saved as $output_filename")
end

check_recursive_qr_runtimes()