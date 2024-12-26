using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_trsm.jl")  # Include performant_trsm.jl file
include("performant_rectrsm.jl")  # Include performant_rectrsm.jl file

function benchmark_rectrsm()
    sizes = [
        64, 128, 256, 512, 1024, 2048, 4096, 8192,
        #  16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152  # New larger sizes
    ]
    rectrsm_runtimes = Float64[]  # Store runtimes for performant_rectrsm! (in milliseconds)
    trsm_runtimes = Float64[]  # Store runtimes for performant_trsm! (in milliseconds)

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
        # Benchmark for performant_trsm!
        # -----------------------------
        # Using 'L' for left side, 'L' for lower triangular, and 'N' for no transpose
        time_trsm = @benchmark performant_trsm!('L', 'L', 'N', $Ac, $Bc) samples=100  # Run it 100 times
        median_runtime_trsm = median(time_trsm).time / 1e6  # Convert nanoseconds to milliseconds
        push!(trsm_runtimes, median_runtime_trsm)  # Save the runtime for this matrix size
        println("performant_trsm! - Size: $n x $n | Runtime: $median_runtime_trsm ms")
    end

    return sizes, rectrsm_runtimes, trsm_runtimes
end

# Run the benchmark
sizes, rectrsm_runtimes, trsm_runtimes = benchmark_rectrsm()

# Plot the results
plot(
    sizes, 
    rectrsm_runtimes, 
    label = "performant_rectrsm!", 
    xlabel = "Matrix Size (n x n)", 
    ylabel = "Runtime (ms)", 
    title = "Performance of performant_rectrsm! vs performant_trsm!", 
    lw = 2, 
    marker = :o, 
    markersize = 8, 
    grid = true,
    xaxis=:log,  # Logarithmic scale for the x-axis
)

# Add performant_trsm! performance to the same plot
plot!(
    sizes, 
    trsm_runtimes, 
    label = "performant_trsm!", 
    lw = 2, 
    marker = :s, 
    markersize = 8
)

# Save the plot as an image
savefig("performant_rectrsm_vs_performant_trsm_2.png")
