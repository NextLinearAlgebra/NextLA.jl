using CUDA
using LinearAlgebra
using Plots
using KernelAbstractions

# Include the file where you saved the RecursiveMixedLU module
include("RecursiveMixedLU.jl") # <-- Ensure this filename matches your implementation file
using .RecursiveMixedLU        # Bring mixed_lu! and MixedLUWorkspace into scope

include("benchmark.jl")        # Must contain run_single_benchmark

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

function run_time_recursive_mixed_lu(
    A_input::CuMatrix{Twork}, 
    nb_val::Int, 
    ::Type{Tlow}
) where {Twork<:Union{Float32,Float64}, Tlow}
    backend = KernelAbstractions.get_backend(A_input)
    n = size(A_input, 1)

    A_work  = copy(A_input)
    A_clean = copy(A_input)
    
    # Preallocate workspace during warm-up/setup so scratch buffer allocation 
    # and view reshaping overhead are strictly excluded from the timing loop
    ws = MixedLUWorkspace(Tlow, Twork, n)
    
    # Hits mixed_lu!(A; nb, Tlow, ws)
    op       = () -> mixed_lu!(A_work; nb=nb_val, Tlow=Tlow, ws=ws)
    reset_op = () -> copyto!(A_work, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000 # Convert to milliseconds
end

# ==============================================================================
# --- Main Timing Driver ---
# ==============================================================================

function check_lu_time_panel_mixed()
    # 2k to 16k matrix sizes as requested
    n_values = [2048, 4096, 8192, 16384]

    # Configure precision scenarios: (Label, Twork, Tlow, nb_size)
    scenarios = [
        ("Pure Float64 (nb=512)",    Float64, Float64, 512),
        ("Mixed F64 / F32 (nb=512)", Float64, Float32, 512),
        ("Pure Float32 (nb=512)",    Float32, Float32, 512),
        ("Mixed F32 / F16 (nb=512)", Float32, Float16, 512)
    ]

    all_results = Dict{String, Vector{Float64}}()
    for (name, _, _, _) in scenarios
        all_results[name] = Float64[]
    end

    println("="^65)
    println("Starting Panel-Downcast RecursiveLU Performance Benchmark")
    println("="^65)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        # Generate diagonally dominant test matrices for both FP32 and FP64
        A_cpu_f64 = rand(Float64, n, n)
        A_cpu_f64 .+= Diagonal(fill(Float64(n * 2.0), n))
        A_gpu_f64 = CuArray(A_cpu_f64)

        A_cpu_f32 = Float32.(A_cpu_f64)
        A_gpu_f32 = CuArray(A_cpu_f32)

        for (name, Twork, Tlow, nb_val) in scenarios
            A_target = (Twork == Float64) ? A_gpu_f64 : A_gpu_f32
            
            runtime_ms = run_time_recursive_mixed_lu(A_target, nb_val, Tlow)
            push!(all_results[name], runtime_ms)
            
            println("    $(rpad(name, 30)) | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        # Aggressive memory cleanup between scaling steps to prevent VRAM fragmentation
        A_cpu_f64 = nothing; A_gpu_f64 = nothing;
        A_cpu_f32 = nothing; A_gpu_f32 = nothing;
        GC.gc(true); CUDA.reclaim()
    end

    # ==========================================================================
    # --- Plotting Results ---
    # ==========================================================================
    plt = plot(
        title="Panel-Downcast LU Performance vs. Matrix Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        legend=:outertopright,
        size=(1050, 700),
        dpi=300
    )

    for (name, _, _, _) in scenarios
        results = all_results[name]
        if occursin("Pure", name)
            linestyle = :dash
            marker_style = :square
        else
            linestyle = :solid
            marker_style = :circle
        end
        plot!(plt, n_values, results, label=name, lw=2, linestyle=linestyle, marker=marker_style)
    end

    savefig(plt, "recursive_mixed_lu_panel_runtimes.png")
    println("\nPlot saved as recursive_mixed_lu_panel_runtimes.png")
end

check_lu_time_panel_mixed()