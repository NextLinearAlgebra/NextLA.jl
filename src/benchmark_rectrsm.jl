using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots
using CSV, Tables


include("performant_rectrsm.jl")  # Include performant_rectrsm.jl file

function trsm_flops(t, m, n)
    flops_add = 0.5 * n * m * (m-1.0)
    flops_mult = 0.5 * n * m * (m+1.0)
    return flops_add + flops_mult
end

function benchmark_rectrsm(T)
    #gpus, #threads, n, rhs, limit(block size), flops, rectime, recgflops/s, stateofart_time, stateofart_gflops/s
    Timing = zeros(10)
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    m_values = [128, 256, 1024]  # Different values of m for benchmarking
   
    Timing = zeros(10)

    for m in m_values

        for n in sizes

            Timing[1] = 1;
            Timing[2] = Threads.nthreads();
            Timing[3] = n;
            Timing[4] = m;
	    Timing[5] = 256;
            Timing[6] = trsm_flops(T, n, m)

            A = CuArray(Matrix(LowerTriangular(rand(n, n))))
            B = CuArray(Matrix(rand(n, m)))

            Ac = copy(A)
            Bc = copy(B)

            time_rectrsm = @belapsed (CUDA.@sync performant_rectrsm!($A, $n, $B))
            Timing[7] = time_rectrsm
	    recgflopss = (trsm_flops(T, n, m)/10^9) / time_rectrsm
	    Timing[8] = recgflopss
            println("performant_rectrsm! - Size: $n x $m | Runtime: $time_rectrsm s Gflops/s: $recgflopss")

            time_trsm = @belapsed (CUDA.@sync CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, $Ac, $Bc))
            Timing[9] = time_trsm
	    cugflopss = (trsm_flops(T, n, m)/10^9) / time_trsm
	    Timing[10] = cugflopss
            println("cuBLAS trsm - Size: $n x $m | Runtime: $time_trsm s Gflops/s: $cugflopss")
	    CSV.write("timings_trsm_Bnonsquare_CUDAـ$(T).csv",  Tables.table(transpose(Timing)), writeheader=false, append=true)

        end
    end

end

# Run the benchmark
for T in [Float32, Float64]
    benchmark_rectrsm(T)
end
#=
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
    savefig(p, "rectrsm_comparison_extended_$m.png")
end
=#
