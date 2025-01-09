using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_rectrsm.jl")  # Include performant_rectrsm.jl file

function trsm_flops(t, m, n)
    flops_add = 0.5 * n * m * (m-1.0)
    flops_mult = 0.5 * n * m * (m+1.0)
    return flops_add + flops_mult
end

function benchmark_rectrsm()
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    m_values = [256]  # Different values of m for benchmarking
    rectrsm_runtimes = Dict()
    trsm_runtimes = Dict()

    for m in m_values
        rectrsm_runtimes[m] = Float64[]
        trsm_runtimes[m] = Float64[]

        for n in sizes
            A = CuArray(Matrix(LowerTriangular(rand(n, n))))
            B = CuArray(Matrix(rand(n, m)))

            Ac = copy(A)
            Bc = copy(B)

            time_rectrsm = @belapsed (CUDA.@sync performant_rectrsm!($A, $n, $B))
            recgflopss = (trsm_flops(Float64, n, m)/10^9) / time_rectrsm
            push!(rectrsm_runtimes[m], time_rectrsm)
            println("performant_rectrsm! - Size: $n x $m | Runtime: $time_rectrsm s Gflops/s: $recgflopss")

            time_trsm = @belapsed (CUDA.@sync CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, $Ac, $Bc))
            cugflopss = (trsm_flops(Float64, n, m)/10^9) / time_trsm
            push!(trsm_runtimes[m], time_trsm)
            println("cuBLAS trsm - Size: $n x $m | Runtime: $time_trsm s Gflops/s: $cugflopss")
        end
    end

    return sizes, rectrsm_runtimes, trsm_runtimes
end

# Run the benchmark
sizes, rectrsm_runtimes, trsm_runtimes = benchmark_rectrsm()

# Generate and save plots
for m in keys(rectrsm_runtimes)
    p = plot(
        sizes,
        rectrsm_runtimes[m],
        label = "performant_rectrsm!",
        xlabel = "Matrix Size (n)",
        ylabel = "Runtime (s)",
        title = "TRSM Runtime Comparison (m=$m)",
        lw = 2,
        marker = :square,
        markersize = 4,
        color = :green,
        legend = :topleft,
        # yscale = :log10
    )
    plot!(
        p,
        sizes,
        trsm_runtimes[m],
        label = "cuBLAS trsm",
        lw = 2,
        marker = :diamond,
        markersize = 4,
        color = :red
    )

    # Save the plot
    savefig(p, "rectrsm_comparison_linear_$m.png")
end
