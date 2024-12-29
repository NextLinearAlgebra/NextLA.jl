using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_trsm_2.jl")  # Include performant_trsm_2.jl file

function benchmark_trsm()
    sizes = [30, 45, 64, 102, 128, 250, 512, 750, 1024]  # Sizes ≤ 1024

    trsm_2_runtimes = Float64[]  # Store runtimes for performant_trsm_2!
    cuda_trsm_runtimes = Float64[]  # Store runtimes for cuBLAS trsm

    for n in sizes
        # Generate random lower triangular matrix A and random matrix B
        A = CuArray(Matrix(LowerTriangular(rand(n, n))))
        B = CuArray(Matrix(rand(n, 1)))

        Ab = copy(A)
        Bb = copy(B)

        Ac = copy(A)  # Copy of A for cuBLAS trsm
        Bc = copy(B)  # Copy of B for cuBLAS trsm

        # -----------------------------
        # Benchmark for performant_trsm_2!
        # -----------------------------
        time_trsm_2 = @benchmark performant_trsm_2!('L', 'L', 'N', $Ab, $Bb) samples=1000
        median_runtime_trsm_2 = median(time_trsm_2).time / 1e6  # Convert nanoseconds to milliseconds
        push!(trsm_2_runtimes, median_runtime_trsm_2)
        println("performant_trsm_2! - Size: $n x $n | Runtime: $median_runtime_trsm_2 ms")

        # -----------------------------
        # Benchmark for cuBLAS trsm
        # -----------------------------
        time_cuda_trsm = @benchmark CUDA.CUBLAS.trsm!(
            'L',      # Side (Left)
            'L',      # Uplo (Lower triangular)
            'N',      # No transpose
            'N',      # Non-diagonal elements
            1.0,      # alpha (scalar)
            $Ac,      # A (lower triangular matrix)
            $Bc       # B (right-hand side)
        ) samples=1000

        median_runtime_cuda_trsm = median(time_cuda_trsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(cuda_trsm_runtimes, median_runtime_cuda_trsm)
        println("cuBLAS trsm - Size: $n x $n | Runtime: $median_runtime_cuda_trsm ms")
    end

    return sizes, trsm_2_runtimes, cuda_trsm_runtimes
end

# Run the benchmark
sizes, trsm_2_runtimes, cuda_trsm_runtimes = benchmark_trsm()

# Plot the results
plot(
    sizes, 
    trsm_2_runtimes, 
    label = "performant_trsm_2!", 
    xlabel = "Matrix Size (n x n)", 
    ylabel = "Runtime (ms)", 
    title = "Performance Comparison: performant_trsm_2! vs cuBLAS trsm", 
    lw = 2, 
    marker = :s, 
    markersize = 8, 
    grid = true
)

# Add cuBLAS trsm to the same plot
plot!(
    sizes, 
    cuda_trsm_runtimes, 
    label = "cuBLAS trsm", 
    lw = 2, 
    marker = :d, 
    markersize = 8
)

# Save the plot as an image
savefig("trsm_2_vs_cuda.png")
