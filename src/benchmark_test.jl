using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots
include("for_testing.jl")  # Include the file with the functions

function trsm_flops(n, m)
    flops_add = 0.5 * n * m * (m-1.0)
    flops_mult = 0.5 * n * m * (m+1.0)
    return flops_add + flops_mult
end

function benchmark_functions()
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    m = 256 
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
            if uplo == "U"
                A = CuArray( Matrix(UpperTriangular(rand(n, n) .+ 1)) .+ Diagonal(10 * ones(n)) )
            else
                A = CuArray( Matrix(LowerTriangular(rand(n, n) .+ 1)) .+ Diagonal(10 * ones(n)) )
            end

            if side == "R"
                B = CuArray(rand(m, n) .+ 1)
            else
                B = CuArray(rand(n, m) .+ 1)
            end
            A_nt = copy(A)
            B_nt = copy(B)

            # Benchmark no_transpose
            time_no_transpose = @belapsed CUDA.@sync $no_transpose_func($A_nt, $B_nt)
            push!(runtimes[(side, uplo)]["no_transpose"], time_no_transpose)

            # Benchmark transpose
            time_transpose = @belapsed CUDA.@sync $transpose_func($A, $B)
            push!(runtimes[(side, uplo)]["transpose"], time_transpose)

            println("Case: $side, $uplo | Size: $n x $m | No Transpose: $time_no_transpose s | Transpose: $time_transpose s")
        end
    end

    return sizes, runtimes
end

# Run the benchmark
sizes, runtimes = benchmark_functions()

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
    savefig(p, "runtime_$(side)_$(uplo)_8_comparison_m_256.png")
end
