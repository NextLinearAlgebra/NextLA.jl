using CUDA
using LinearAlgebra
using Plots
using NextLA
using KernelAbstractions

include("benchmark.jl")

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

function run_time_pure_lu(A_fp64::CuMatrix, T_prec::DataType, block_size::Int=2048)
    backend = KernelAbstractions.get_backend(A_fp64)

    A_to_factor = T_prec.(A_fp64)
    A_clean = copy(A_to_factor)
    
    op = () -> lu_recursive!(A_to_factor, block_size)
    reset_op = () -> copyto!(A_to_factor, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000
end

function run_time_mixed_lu(A_fp64::CuMatrix, precisions::Vector)
    backend = KernelAbstractions.get_backend(A_fp64)
    local A_mixed_input
    
    reset_op = () -> begin
        A_mixed_input = FullMixedPrec(copy(A_fp64); precisions=precisions)
        KernelAbstractions.synchronize(backend)
    end
    
    op = () -> lu_recursive_mixed!(A_mixed_input)
    
    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000
end

function run_time_cusolver_lu(A_fp64::CuMatrix, T_prec::DataType)
    backend = KernelAbstractions.get_backend(A_fp64)

    A_to_factor = T_prec.(A_fp64)
    A_clean = copy(A_to_factor)
    
    op = () -> CUSOLVER.getrf!(A_to_factor)
    reset_op = () -> copyto!(A_to_factor, A_clean)

    time_ns = benchmark_op(op, reset_op, backend)
    return time_ns / 1_000_000
end

# ==============================================================================
# --- Main Timing Driver ---
# ==============================================================================

function check_lu_time()
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
        "Mixed [F16, F32]"                => [Float16, Float32],
        "Mixed [F32, F64]"                => [Float32, Float64],
        "Mixed [F16, F64]"                => [Float16, Float64],
        
        # Deep F16 leaves
        "Mixed [F16, F16, F32]"           => [Float16, Float16, Float32],
        "Mixed [F16, F16, F16, F32]"      => [Float16, Float16, Float16, Float32],
        "Mixed [F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        
        # Smooth gradients
        "Mixed [F16, F16, F32, F64]"      => [Float16, Float16, Float32, Float64],
        "Mixed [F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64]
    )

    all_results = Dict()
    for name in keys(pure_scenarios);     all_results[name] = Float64[]; end
    for name in keys(cusolver_scenarios); all_results[name] = Float64[]; end
    for name in keys(mixed_scenarios);    all_results[name] = Float64[]; end

    println("="^50)
    println("Starting LU Performance Benchmark")
    println("="^50)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        A_cpu = rand(Float64, n, n)
        A_cpu .+= Diagonal(fill(n * 2.0, n))
        A_fp64 = CuArray(A_cpu)

        println("\n  --- Standard CUSOLVER Scenarios ---")
        for (name, T_prec) in cusolver_scenarios
            runtime_ms = run_time_cusolver_lu(A_fp64, T_prec)
            push!(all_results[name], runtime_ms)
            println("    $name | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        println("\n  --- Pure Recursive Scenarios ---")
        for (name, T_prec) in pure_scenarios
            runtime_ms = run_time_pure_lu(A_fp64, T_prec)
            push!(all_results[name], runtime_ms)
            println("    $name | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        println("\n  --- Mixed Precision Scenarios ---")
        for (name, prec_list) in mixed_scenarios
            runtime_ms = run_time_mixed_lu(A_fp64, prec_list)
            push!(all_results[name], runtime_ms)
            println("    $name | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        A_cpu = nothing; A_fp64 = nothing; GC.gc(true); CUDA.reclaim()
    end

    # Plotting
    plt = plot(
        title="LU Factorization Performance vs. Matrix Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        legend=:outertopright,
        size=(1000, 700),
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
        plot!(plt, n_values, results, label=name, lw=2, linestyle=linestyle, marker=marker_style)
    end

    savefig(plt, "lu_runtime_results.png")
    println("\nPlot saved as lu_runtime_results.png")
end

check_lu_time()