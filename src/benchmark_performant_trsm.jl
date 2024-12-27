using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots
using CSV  # Add the CSV package

# include("performant_trsm.jl")  # Include performant_trsm.jl file
include("performant_rectrsm.jl")  # Include performant_rectrsm.jl file

function benchmark_rectrsm()
    sizes = [30, 45, 64, 102, 128, 250, 512, 750, 1024, 2048, 4096, 5000, 6000, 8192, 10000]
    # sizes = [30, 45, 64, 102, 128, 250, 512, 750, 1024, 2048, 4096]
    rectrsm_runtimes = Float64[]  # Store runtimes for performant_rectrsm! (in milliseconds)
    trsm_runtimes = Float64[]  # Store runtimes for cuBLAS trsm (in milliseconds)

    for n in sizes
        # Generate a random lower triangular matrix A and random matrix B
        A = CuArray(Matrix(LowerTriangular(rand(n, n))))  # Lower triangular matrix
        B = CuArray(Matrix(rand(n, 1)))  # k=1

        Ac = copy(A)  # Copy of A for trsm!
        Bc = copy(B)  # Copy of B for trsm!

        # -----------------------------
        # Benchmark for performant_rectrsm!
        # -----------------------------
        time_rectrsm = @benchmark performant_rectrsm!($A, $n, $B) samples=100  # Run it 100 times
        median_runtime_rectrsm = median(time_rectrsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(rectrsm_runtimes, median_runtime_rectrsm)  # Save the runtime for this matrix size
        println("performant_rectrsm! - Size: $n x $n | Runtime: $median_runtime_rectrsm ms")

        # -----------------------------
        # Benchmark for cuBLAS trsm
        # -----------------------------
        # cuBLAS trsm call: Solve A * X = B
        # Arguments: handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb
        time_CUDAtrsm = @benchmark CUDA.CUBLAS.trsm!(
            'L',      # Side (Left)
            'L',      # Uplo (Lower triangular)
            'N',      # No transpose
            'N',      # Non-diagonal elements
            1.0,      # alpha (scalar)
            $Ac,      # A (lower triangular matrix)
            $Bc,      # B (right-hand side)
        ) samples=100  # Run it 100 times

        median_runtime_trsm = median(time_CUDAtrsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(trsm_runtimes, median_runtime_trsm)  # Save the runtime for this matrix size
        println("cuBLAS trsm - Size: $n x $n | Runtime: $median_runtime_trsm ms")
    end

    return sizes, rectrsm_runtimes, trsm_runtimes
end

# Run the benchmark
sizes, rectrsm_runtimes, trsm_runtimes = benchmark_rectrsm()

# # Save data to CSV
# data = DataFrame(
#     Size = sizes,
#     Performant_Rectrsm = rectrsm_runtimes,
#     CUDA_Trsm = trsm_runtimes
# )

# # Write to CSV file
# CSV.write("benchmark_results.csv", data)

# Plot the results
plot(
    sizes, 
    rectrsm_runtimes, 
    label = "performant_rectrsm!", 
    xlabel = "Matrix Size (n x n)", 
    ylabel = "Runtime (ms)", 
    title = "Performance of rectrsm vs cuBLAS trsm on GPU", 
    lw = 2, 
    marker = :o, 
    markersize = 8, 
    grid = true,
    # xaxis=:log,  # Logarithmic scale for the x-axis
    # yaxis=:log,  # Logarithmic scale for the y-axis
)

# Add cuBLAS trsm performance to the same plot
plot!(
    sizes, 
    trsm_runtimes, 
    label = "cuBLAS trsm", 
    lw = 2, 
    marker = :s, 
    markersize = 8
)

# Save the plot as an image
savefig("performant_rectrsm_vs_cublas_trsm_2.png")
