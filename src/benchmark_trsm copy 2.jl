using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_trsm_2 copy.jl")  # Include performant_trsm_2_copy.jl file

function benchmark_trsm()
    sizes = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
             544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]
    m_values = [1, 2, 4, 8, 16, 32, 64, 128]  # Different numbers of columns in B

    results = Dict()

    for m in m_values
        trsm_2_copy_runtimes = Float64[]
        cuda_trsm_runtimes = Float64[]

        for n in sizes
            # Generate random lower triangular matrix A and random matrix B
            A = CuArray(Matrix(LowerTriangular(rand(n, n))))
            B = CuArray(Matrix(rand(n, m)))

            Ac = copy(A)  # Copy of A for cuBLAS trsm
            Bc = copy(B)  # Copy of B for cuBLAS trsm

            # Benchmark for performant_trsm_2_2!
            time_trsm_2_copy = @benchmark performant_trsm_2_2!('L', 'L', 'N', $A, $B) samples=50 seconds=2
            median_runtime_trsm_2_copy = median(time_trsm_2_copy).time / 1e6
            push!(trsm_2_copy_runtimes, median_runtime_trsm_2_copy)
            println("performant_trsm_2_2! - Size: $n x $n, m: $m | Runtime: $median_runtime_trsm_2_copy ms")

            # Benchmark for cuBLAS trsm
            time_cuda_trsm = @benchmark CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, $Ac, $Bc) samples=50 seconds=2
            median_runtime_cuda_trsm = median(time_cuda_trsm).time / 1e6
            push!(cuda_trsm_runtimes, median_runtime_cuda_trsm)
            println("cuBLAS trsm - Size: $n x $n, m: $m | Runtime: $median_runtime_cuda_trsm ms")
        end

        results[m] = (trsm_2_copy_runtimes, cuda_trsm_runtimes)
    end

    return sizes, results
end

# Run the benchmark
sizes, results = benchmark_trsm()

# Create plots for each m value
for m in keys(results)
    trsm_2_copy_runtimes, cuda_trsm_runtimes = results[m]
    
    plot(
        sizes,
        trsm_2_copy_runtimes,
        label = "performant_trsm_2_2!",
        xlabel = "Matrix Size (n x n)",
        ylabel = "Runtime (ms)",
        title = "Runtime Comparison for B with m=$m",
        lw = 2,
        marker = :square,
        markersize = 4,
        color = :green,
        legend = :topleft,
        yscale = :log10
    )
    plot!(
        sizes,
        cuda_trsm_runtimes,
        label = "cuBLAS trsm",
        lw = 2,
        marker = :diamond,
        markersize = 4,
        color = :red
    )

    # Save the plot
    savefig("trsm_comparison_m_$m.png")
end

