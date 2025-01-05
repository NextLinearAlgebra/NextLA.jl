using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots
using CSV  # Add the CSV package

include("performant_rectrsm.jl")  # Include performant_rectrsm.jl file
include("performant_rectrsm copy.jl")  # Include performant_rectrsm_copy.jl file

function benchmark_rectrsm()
    sizes = [30, 64, 128, 256, 512, 1024, 2048, 4096]
    rectrsm_runtimes = Float64[]  # Store runtimes for performant_rectrsm!
    rectrsm_copy_runtimes = Float64[]  # Store runtimes for performant_rectrsm_copy!
    trsm_runtimes = Float64[]  # Store runtimes for cuBLAS trsm!

    for n in sizes
        # Generate random lower triangular matrix A and random matrix B
        A = CuArray(Matrix(LowerTriangular(rand(n, n))))  # Lower triangular matrix
        B = CuArray(Matrix(rand(n, 1)))  # Vector B

        Ab = copy(A)
        Bb = copy(B)

        Ac = copy(A)
        Bc = copy(B)

        # -----------------------------
        # Benchmark for performant_rectrsm!
        # -----------------------------
        time_rectrsm = @benchmark performant_rectrsm!($A, $n, $B) samples=100
        median_runtime_rectrsm = median(time_rectrsm).time / 1e6  # Convert to milliseconds
        push!(rectrsm_runtimes, median_runtime_rectrsm)
        println("performant_rectrsm! - Size: $n x $n | Runtime: $median_runtime_rectrsm ms")

        # -----------------------------
        # Benchmark for performant_rectrsm_copy!
        # -----------------------------
        time_rectrsm_copy = @benchmark performant_rectrsm_copy!($Ab, $n, $Bb) samples=100
        median_runtime_rectrsm_copy = median(time_rectrsm_copy).time / 1e6  # Convert to milliseconds
        push!(rectrsm_copy_runtimes, median_runtime_rectrsm_copy)
        println("performant_rectrsm_copy! - Size: $n x $n | Runtime: $median_runtime_rectrsm_copy ms")

        # -----------------------------
        # Benchmark for cuBLAS trsm
        # -----------------------------
        time_trsm = @benchmark CUDA.CUBLAS.trsm!(
            'L',  # Side (Left)
            'L',  # Uplo (Lower triangular)
            'N',  # No transpose
            'N',  # Non-diagonal elements
            1.0,  # alpha (scalar)
            $Ac,  # A
            $Bc   # B
        ) samples=100
        median_runtime_trsm = median(time_trsm).time / 1e6  # Convert to milliseconds
        push!(trsm_runtimes, median_runtime_trsm)
        println("cuBLAS trsm - Size: $n x $n | Runtime: $median_runtime_trsm ms")
    end

    return sizes, rectrsm_runtimes, rectrsm_copy_runtimes, trsm_runtimes
end

# Run the benchmark
sizes, rectrsm_runtimes, rectrsm_copy_runtimes, trsm_runtimes = benchmark_rectrsm()

# Plot the results
plot(
    sizes,
    rectrsm_runtimes,
    label = "performant_rectrsm!",
    xlabel = "Matrix Size (n x n)",
    ylabel = "Runtime (ms)",
    lw = 2,
    marker = :circle,
    markersize = 6,
    color = :blue
)
plot!(
    sizes,
    rectrsm_copy_runtimes,
    label = "performant_rectrsm_copy!",
    lw = 2,
    marker = :square,
    markersize = 6,
    color = :green
)
# plot!(
#     sizes,
#     trsm_runtimes,
#     label = "cuBLAS trsm",
#     lw = 2,
#     marker = :diamond,
#     markersize = 6,
#     color = :red
# )

# Save the plot
savefig("performant_rectrsm_comparison.png")
