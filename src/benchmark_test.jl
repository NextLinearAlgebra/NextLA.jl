using LinearAlgebra
using CUDA
using BenchmarkTools
using Statistics
using Plots
include("for_testing.jl")  # Include the file with the functions

function trsm_flops(n, m)
    flops_add = 0.5 * n * m * (m-1.0)
    flops_mult = 0.5 * n * m * (m+1.0)
    return flops_add + flops_mult
end

function benchmark_functions(num_trials=10)
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    m = 256  # Fixed m value
    runtimes = Dict()

    cases = [
        ("R", "U", right_upper_no_transpose, right_upper_transpose),
        ("R", "L", right_lower_no_transpose, right_lower_transpose),
        ("L", "U", left_upper_no_transpose, left_upper_transpose),
        ("L", "L", left_lower_no_transpose, left_lower_transpose)
    ]

    for (side, uplo, no_transpose_func, transpose_func) in cases
        runtimes[(side, uplo)] = Dict("no_transpose" => Float64[], "transpose" => Float64[])

        for n in sizes
            A = CuArray(if uplo == "U"
                Matrix(UpperTriangular(rand(n, n) .+ 1))
            else
                Matrix(LowerTriangular(rand(n, n) .+ 1))
            end .+ Diagonal(10 * ones(n)))

            B = if side == "R"
                CuArray(rand(m, n) .+ 1)
            else
                CuArray(rand(n, m) .+ 1)
            end

            # Benchmark no_transpose
            times_no_transpose = [
                @belapsed CUDA.@sync $no_transpose_func($(copy(A)), $(copy(B)))
                for _ in 1:num_trials
            ]
            avg_time_no_transpose = mean(times_no_transpose)
            push!(runtimes[(side, uplo)]["no_transpose"], avg_time_no_transpose)

            # Benchmark transpose
            times_transpose = [
                @belapsed CUDA.@sync $transpose_func($(copy(A)), $(copy(B)))
                for _ in 1:num_trials
            ]
            avg_time_transpose = mean(times_transpose)
            push!(runtimes[(side, uplo)]["transpose"], avg_time_transpose)

            println("Case: $side, $uplo | Size: $n x $m | No Transpose: $avg_time_no_transpose s | Transpose: $avg_time_transpose s")
        end
    end

    return sizes, runtimes
end

# Run the benchmark
sizes, runtimes = benchmark_functions(10)  # Run 10 trials for each case

# Generate and save plots
for ((side, uplo), data) in runtimes
    p = plot(
        sizes,
        data["no_transpose"],
        label = "No Transpose",
        xlabel = "Matrix Size (n)",
        ylabel = "Runtime (s)",
        title = "Runtime Comparison ($side, $uplo, m=256)",
        lw = 2,
        marker = :square,
        markersize = 4,
        color = :blue,
        legend = :topleft
    )
    plot!(
        p,
        sizes,
        data["transpose"],
        label = "Transpose",
        lw = 2,
        marker = :diamond,
        markersize = 4,
        color = :red
    )

    # Save the plot
    savefig(p, "runtime_$(side)_$(uplo)_comparison_m_256.png")
end
