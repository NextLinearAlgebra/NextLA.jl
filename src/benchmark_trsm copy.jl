using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_trsm_2.jl")  # Include performant_trsm_2.jl file
include("performant_trsm_2 copy.jl")  # Include performant_trsm_2_copy.jl file

function benchmark_trsm()
    sizes = [64, 98, 132, 167, 201, 235, 269, 304, 338, 372,
             407, 441, 475, 509, 544, 578, 612, 646, 681, 715, 749,
             784, 818, 852, 886, 921, 955, 989, 1024]
    # sizes = [30, 45, 64, 102, 128, 250, 350, 512, 750, 850, 950, 1000, 1024]  # Sizes ≤ 1024

    trsm_2_runtimes = Float64[]  # Runtimes for performant_trsm_2!
    trsm_2_copy_runtimes = Float64[]  # Runtimes for performant_trsm_2_copy!
    cuda_trsm_runtimes = Float64[]  # Runtimes for cuBLAS trsm

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
        time_trsm_2 = @benchmark performant_trsm_2!('L', 'L', 'N', $A, $B) samples=100
        median_runtime_trsm_2 = median(time_trsm_2).time / 1e6  # Convert nanoseconds to milliseconds
        push!(trsm_2_runtimes, median_runtime_trsm_2)
        println("performant_trsm_2! - Size: $n x $n | Runtime: $median_runtime_trsm_2 ms")

        # -----------------------------
        # Benchmark for performant_trsm_2_copy!
        # -----------------------------
        time_trsm_2_copy = @benchmark performant_trsm_2_2!('L', 'L', 'N', $Ab, $Bb) samples=100
        median_runtime_trsm_2_copy = median(time_trsm_2_copy).time / 1e6  # Convert nanoseconds to milliseconds
        push!(trsm_2_copy_runtimes, median_runtime_trsm_2_copy)
        println("performant_trsm_2 copy! - Size: $n x $n | Runtime: $median_runtime_trsm_2_copy ms")

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
        ) samples=100

        median_runtime_cuda_trsm = median(time_cuda_trsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(cuda_trsm_runtimes, median_runtime_cuda_trsm)
        println("cuBLAS trsm - Size: $n x $n | Runtime: $median_runtime_cuda_trsm ms")
    end

    return sizes, trsm_2_runtimes, trsm_2_copy_runtimes, cuda_trsm_runtimes
end

# Run the benchmark
sizes, trsm_2_runtimes, trsm_2_copy_runtimes, cuda_trsm_runtimes = benchmark_trsm()

# Simplified Plot
plot(
    sizes,
    trsm_2_runtimes,
    label = "performant_trsm_2!",
    xlabel = "Matrix Size (n x n)",
    ylabel = "Runtime (ms)",
    lw = 2,
    marker = :circle,
    markersize = 6,
    color = :blue
)
plot!(
    sizes,
    trsm_2_copy_runtimes,
    label = "performant_trsm_2_copy!",
    lw = 2,
    marker = :square,
    markersize = 6,
    color = :green
)
plot!(
    sizes,
    cuda_trsm_runtimes,
    label = "cuBLAS trsm",
    lw = 2,
    marker = :diamond,
    markersize = 6,
    color = :red
)


# Save the plot
savefig("trsm_comparison_no_mutates.png")
