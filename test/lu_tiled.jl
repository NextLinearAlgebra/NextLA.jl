using NextLA
using Test
using CUDA
using LinearAlgebra
using Printf
using KernelAbstractions

include("benchmark.jl")

@testset "Accuracy Test for Tiled LU Kernel" begin
    n_sizes = [32, 64, 128, 256]
    tile_size = 16
    tolerance = 1e-10

    @testset "P * A = L * U Tiled Factorization" begin
        for n in n_sizes
            A_host = rand(Float64, n, n)
            A_orig = copy(A_host)
            d_A = CuArray(A_host)
            d_A, d_P = tile_lu_factor!(d_A, tile_size)

            A_res = Array(d_A)
            P_res = Array(d_P)
            L = UnitLowerTriangular(A_res)
            U = UpperTriangular(A_res)
            A_reconstructed = inv(P_res) * L * U

            diff_norm = norm(A_reconstructed - A_orig)
            ref_norm  = norm(A_orig)
            rel_error = (ref_norm > 0) ? (diff_norm / ref_norm) : diff_norm
            println("Size: $n x $n, Tile: $tile_size | Result Diff (Relative Error): $rel_error")
            @test rel_error < tolerance
        end
    end
end

function benchmark_op(op, reset_op, backend)
    reset_op()
    op()
    KernelAbstractions.synchronize(backend)
    min_time_ns = Inf
    for _ in 1:10
        reset_op()
        time = run_single_benchmark(op, backend)
        min_time_ns = min(min_time_ns, time)
    end
    return min_time_ns
end

function run_lutiled_benchmark()
    n_sizes = [32, 64, 128, 256, 512, 1024]
    tile_size = 16
    println("="^90)
    @printf("%-6s | %-18s | %-18s | %-15s\n", "N", "Time Custom (ms)", "Time CUSOLVER (ms)", "Speedup (Ref/KA)")
    println("="^90)

    for n in n_sizes
        A_host = rand(Float64, n, n)
        d_A = CuArray(A_host)
        d_A_init = CuArray(A_host)
        backend = KernelAbstractions.get_backend(d_A)

        op_custom = () -> tile_lu_factor!(d_A, tile_size)
        reset_custom = () -> copyto!(d_A, d_A_init)
        time_custom_ns = benchmark_op(op_custom, reset_custom, backend)
        time_custom_ms = time_custom_ns / 1_000_000

        d_A_cusolver = CuArray(A_host)
        op_cusolver = () -> CUDA.CUSOLVER.getrf!(d_A_cusolver)
        reset_cusolver = () -> copyto!(d_A_cusolver, d_A_init)
        time_cusolver_ns = benchmark_op(op_cusolver, reset_cusolver, backend)
        time_cusolver_ms = time_cusolver_ns / 1_000_000

        ratio = time_cusolver_ms / time_custom_ms
        @printf("%6d | %18.4f | %18.4f | %15.4fx\n", n, time_custom_ms, time_cusolver_ms, ratio)
        CUDA.reclaim()
    end
    println("-"^90)
end

run_lutiled_benchmark()
