using CUDA
using LinearAlgebra
using Plots
using NextLA

include("benchmark.jl")

function run_time_pure_lu(A_fp64::CuMatrix, T_prec::DataType)
    n = size(A_fp64, 1)
    backend = get_backend(A_fp64)

    A_to_factor = T_prec.(A_fp64)
    A_clean = copy(A_to_factor)
    
    # Warmup
    CUSOLVER.getrf!(A_to_factor)

    time_ns = run_manual_benchmark(backend) do
        copyto!(A_to_factor, A_clean)
        CUSOLVER.getrf!(A_to_factor)
    end
    
    return time_ns / 1_000_000
end

function run_time_mixed_lu(A_fp64::CuMatrix, precisions::Vector)
    n = size(A_fp64, 1)
    backend = get_backend(A_fp64)
    
    A_mixed_input = FullMixedPrec(copy(A_fp64); precisions=precisions)
    # A cleaner copy for reset, though we just re-instantiate or copy the base arrays
    A_clean_fp64 = copy(A_fp64)
    
    threshold = 256
    
    # Warmup
    lu_recursive_mixed!(A_mixed_input, threshold)

    time_ns = run_manual_benchmark(backend) do
        # Re-initialize or reset the structure (a bit heavy, but safe)
        A_mixed_perf = FullMixedPrec(copy(A_clean_fp64); precisions=precisions)
        lu_recursive_mixed!(A_mixed_perf, threshold)
    end

    return time_ns / 1_000_000
end

function check_lu_time()
    n_values = [512, 1024, 2048, 4096]

    pure_scenarios = Dict(
        "CUSOLVER F32" => [Float32],
        "CUSOLVER F64" => [Float64],
        "CUSOLVER F16" => [Float16]
    )
    
    mixed_scenarios = Dict(
        "Mixed [F16, F32]"                => [Float16, Float32],
        "Mixed [F32, F64]"                => [Float32, Float64],
        "Mixed [F16, F16, F32]"           => [Float16, Float16, Float32]
    )

    all_results = Dict()
    for name in keys(pure_scenarios)
        all_results[name] = Float64[]
    end
    for name in keys(mixed_scenarios)
        all_results[name] = Float64[]
    end

    println("="^50)
    println("Starting LU Performance Benchmark")
    println("="^50)

    for n in n_values
        println("\n--- Testing Matrix Size: $n x $n ---")
        
        A_cpu = rand(Float64, n, n)
        A_cpu .+= Diagonal(fill(n * 2.0, n))
        A_fp64 = CuArray(A_cpu)

        for (name, prec_list) in pure_scenarios
            T_prec = prec_list[1]
            runtime_ms = run_time_pure_lu(A_fp64, T_prec)
            push!(all_results[name], runtime_ms)
            println("  $name | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end

        for (name, prec_list) in mixed_scenarios
            runtime_ms = run_time_mixed_lu(A_fp64, prec_list)
            push!(all_results[name], runtime_ms)
            println("  $name | Runtime: $(round(runtime_ms, sigdigits=4)) ms")
        end
    end

    # Plotting
    plt = plot(
        title="LU Factorization Performance vs. Matrix Size",
        xlabel="Matrix Size (n x n)",
        ylabel="Runtime (ms) [Lower is Better]",
        xaxis=:log2,
        yaxis=:log10,
        legend=:outertopleft,
        size=(800, 600),
        dpi=300
    )

    for (name, results) in all_results
        linestyle = occursin("CUSOLVER", name) ? :dash : :solid
        marker_style = occursin("CUSOLVER", name) ? :square : :circle
        plot!(plt, n_values, results, label=name, lw=2, linestyle=linestyle, marker=marker_style)
    end

    savefig(plt, "lu_runtime_results.png")
    println("\nPlot saved as lu_runtime_results.png")
end

check_lu_time()
