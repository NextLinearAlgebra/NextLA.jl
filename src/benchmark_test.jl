using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("for_testing.jl")  # Include the file with the two functions

function trsm_flops(n, m)
    flops_add = 0.5 * n * m * (m-1.0)
    flops_mult = 0.5 * n * m * (m+1.0)
    return flops_add + flops_mult
end

function benchmark_functions()
    sizes = [256, 512, 1024, 2048, 4096, 8192]  # Matrix sizes for benchmarking
    m_values = [128, 256, 512]  # Different values of m for benchmarking

    left_upper_no_transpose_runtimes = Dict()
    left_upper_transpose_runtimes = Dict()

    for m in m_values
        left_upper_no_transpose_runtimes[m] = Float64[]
        left_upper_transpose_runtimes[m] = Float64[]

        for n in sizes
            A = Matrix(UpperTriangular(rand(n, n) .+ 1))
            A .+= Diagonal(10 * ones(n))  # Ensure well-conditioned
            A = CuArray(A)
            B = CuArray(rand(n, m) .+ 1)
            A_c = copy(A)
            B_c = copy(B)

            # Benchmark left_upper_no_transpose
            time_no_transpose = @belapsed (CUDA.@sync left_upper_no_transpose($A, $B))
            push!(left_upper_no_transpose_runtimes[m], time_no_transpose)

            # Benchmark left_upper_transpose
            time_transpose = @belapsed (CUDA.@sync left_upper_transpose($A_c, $B_c))
            push!(left_upper_transpose_runtimes[m], time_transpose)

            println("Size: $n x $m | No Transpose: $time_no_transpose s | Transpose: $time_transpose s")
        end
    end

    return sizes, m_values, left_upper_no_transpose_runtimes, left_upper_transpose_runtimes
end

# Run the benchmark
sizes, m_values, no_transpose_runtimes, transpose_runtimes = benchmark_functions()

# Generate and save plots
for m in m_values
    p = plot(
        sizes,
        no_transpose_runtimes[m],
        label = "No Transpose",
        xlabel = "Matrix Size (n)",
        ylabel = "Runtime (s)",
        title = "Runtime Comparison (m=$m)",
        lw = 2,
        marker = :square,
        markersize = 4,
        color = :blue,
        legend = :topleft
    )
    plot!(
        p,
        sizes,
        transpose_runtimes[m],
        label = "Transpose",
        lw = 2,
        marker = :diamond,
        markersize = 4,
        color = :red
    )

    # Save the plot
    savefig(p, "runtime_comparison_m_$m.png")
end
