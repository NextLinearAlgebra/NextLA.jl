using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_rectrsm.jl")  # Include performant_rectrsm.jl file

function trsm_flops(t, m, n)
    flops_add= 0.5 * n * m * (m-1.0)
    flops_mult= 0.5 * n * m * (m+1.0)
     return flops_add + flops_mult
 end

function benchmark_rectrsm()
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 10240]
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384] #, 32768]
    m_values = [256]  # Different values of m for benchmarking
    rectrsm_runtimes = Dict()  # Dictionary to store runtimes for different m values
    trsm_runtimes = Dict()  # Dictionary to store cuBLAS runtimes for different m values

#    for m in m_values
       # rectrsm_runtimes[m] = Float64[]  # Initialize an empty list for each m value
        #trsm_runtimes[m] = Float64[]  # Initialize an empty list for each m value

        for n in sizes
            m = m_values[1]
            rectrsm_runtimes[m] = Float64[]  # Initialize an empty list for each m value
            trsm_runtimes[m] = Float64[]  
            # Generate random lower triangular matrix A and random matrix B (of size n x m)
            A = CuArray(Matrix(LowerTriangular(rand(n, n))))  # Lower triangular matrix
            B = CuArray(Matrix(rand(n, m)))  # Matrix B of size n x m

            Ac = copy(A)
            Bc = copy(B)

            # -----------------------------
            # Benchmark for performant_rectrsm!
            # -----------------------------
            time_rectrsm = @belapsed (CUDA.@sync performant_rectrsm!($A, $n, $B)) #@benchmark performant_rectrsm!($A, $n, $B) samples=100
            median_runtime_rectrsm =time_rectrsm  #median(time_rectrsm).time / 1e6  # Convert to milliseconds
            recgflopss = (trsm_flops(Float64, n, m)/10^9) / median_runtime_rectrsm
            push!(rectrsm_runtimes[m], median_runtime_rectrsm)
            println("performant_rectrsm! - Size: $n x $m | Runtime: $median_runtime_rectrsm s Gflops/s: $recgflopss")

            # -----------------------------
            # Benchmark for cuBLAS trsm
            # -----------------------------
            time_trsm = @belapsed (CUDA.@sync CUDA.CUBLAS.trsm!( #@benchmark CUDA.CUBLAS.trsm!(
                'L',  # Side (Left)
                'L',  # Uplo (Lower triangular)
                'N',  # No transpose
                'N',  # Non-diagonal elements
                1.0,  # alpha (scalar)
                $Ac,  # A
                $Bc   # B
            ))# samples=100
            median_runtime_trsm = time_trsm #median(time_trsm).time / 1e6  # Convert to milliseconds
            cugflopss = (trsm_flops(Float64, n, m)/10^9) / median_runtime_trsm
            push!(trsm_runtimes[m], median_runtime_trsm)
            println("cuBLAS trsm - Size: $n x $m | Runtime: $median_runtime_trsm s Gflops/s: $cugflopss")
        end
#    end

    return sizes, rectrsm_runtimes, trsm_runtimes
end

# Run the benchmark
sizes, rectrsm_runtimes, trsm_runtimes = benchmark_rectrsm()


# Generate and save separate plots for each value of m
for m in keys(rectrsm_runtimes)
    # Convert sizes to Float64 to ensure compatibility
    x_values = Float64.(sizes)
    
    # Create a new plot for each m value
    p = plot(
        x_values,
        rectrsm_runtimes[m],  # Performant Rectrsm!
        label = "performant_rectrsm! (m=$m)",
        xlabel = "Matrix Size (n x n)",
        ylabel = "Runtime (s)",
        lw = 2,
        marker = :circle,
        markersize = 6,
        color = :blue,
        xscale = :log2,  # Use logarithmic scale for x-axis
        yscale = :log10  # Use logarithmic scale for y-axis
    )
    plot!(
        x_values,
        trsm_runtimes[m],  # cuBLAS trsm
        label = "cuBLAS trsm (m=$m)",
        lw = 2,
        marker = :square,
        markersize = 6,
        color = :red,
        linestyle = :dash
    )

    # Set the title
    title!("Performance Comparison (m=$m)")

    # Save the plot for this m value
    savefig(p, "performant_rectrsm_1024t_comparison_m_$m.png")
end
