using LinearAlgebra
using BenchmarkTools
using Plots

include("rectrsm.jl")  # Include rectrsm.jl file
include("trsm.jl")  # Include trsm.jl file

function benchmark_rectrsm()
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # Matrix sizes to test
    rectrsm_runtimes = Float64[]  # Store runtimes for rectrsm! (in milliseconds)
    trsm_runtimes = Float64[]  # Store runtimes for trsm! (in milliseconds)

    for n in sizes
        # Generate a random lower triangular matrix A and random matrix B
        A = Matrix(LowerTriangular(rand(n, n)))  # Lower triangular matrix
        B = Matrix(rand(n, 1))  # k=1
        Ac = copy(A)  # Copy of A for trsm!
        Bc = copy(B)  # Copy of B for trsm!

        # -----------------------------
        # Benchmark for rectrsm!
        # -----------------------------
        time_rectrsm = @benchmark rectrsm!($A, $n, $B) samples=100  # Run it 100 times
        median_runtime_rectrsm = median(time_rectrsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(rectrsm_runtimes, median_runtime_rectrsm)  # Save the runtime for this matrix size
        println("rectrsm! - Size: $n x $n | Runtime: $median_runtime_rectrsm ms")
        
        # -----------------------------
        # Benchmark for trsm!
        # -----------------------------
        # Using 'L' for left side, 'L' for lower triangular, and 'N' for no transpose
        time_trsm = @benchmark trsm!('L', 'L', 'N', $Ac, $Bc) samples=100  # Run it 100 times
        median_runtime_trsm = median(time_trsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(trsm_runtimes, median_runtime_trsm)  # Save the runtime for this matrix size
        println("trsm! - Size: $n x $n | Runtime: $median_runtime_trsm ms")
    end

    return sizes, rectrsm_runtimes, trsm_runtimes
end


# Run the benchmark
sizes, rectrsm_runtimes, trsm_runtimes = benchmark_rectrsm()

# Plot the results
plot(
    sizes, 
    rectrsm_runtimes, 
    label = "rectrsm!", 
    xlabel = "Matrix Size (n x n)", 
    ylabel = "Runtime (ms)", 
    title = "Performance of rectrsm! vs trsm! on CPU", 
    lw = 2, 
    marker = :o, 
    markersize = 8, 
    grid = true
)

# Add trsm! performance to the same plot
plot!(
    sizes, 
    trsm_runtimes, 
    label = "trsm!", 
    lw = 2, 
    marker = :s, 
    markersize = 8
)

# Save the plot as an image
savefig("rectrsm_vs_trsm_performance_plot_2.png")
