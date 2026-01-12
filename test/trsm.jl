using Test
using CUDA
using LinearAlgebra

include("benchmark.jl")


@testset "Accuracy Test for TRSM Kernels" begin
    # Matrix sizes to test
    sizes = [16, 32, 128, 256, 1024, 250, 275, 300, 325, 350, 750]

    # Number of columns/rows in B to test
    m_sizes = [1, 8, 64, 256]

    # Tolerance for accuracy check
    tolerance = 1e-5

    ###########################################################################
    # Test LeftLowerTRSM! (solves A * X = B with A lower-triangular on the left)
    ###########################################################################
    @testset "LeftLowerTRSM!" begin
        for n in sizes
            for m in m_sizes
                # Prepare CPU matrices
                A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1))
                A .+= Diagonal(10.0 * ones(Float32, n))
                B = rand(Float32, n, m) .+ 1

                # Copies for baseline
                Ac = copy(A)
                Bc = copy(B)

                # Move to GPU
                A_gpu = CuArray(A)
                B_gpu = CuArray(B)

                # Call our kernel
                LeftLowerTRSM!(A_gpu, B_gpu)

                # Baseline with BLAS trsm!
                # CUBLAS.BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ac, Bc)
                CUBLAS.BLAS.trsm!('L','L','N','N', one(eltype(A)), Ac, Bc)

                # Compute relative error
                result_diff = norm(Array(B_gpu) .- Bc) / norm(Bc)
                println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                @test result_diff < tolerance
            end
        end
    end

    ###########################################################################
    # Test LeftUpperTRSM! (solves A * X = B with A upper-triangular on the left)
    ###########################################################################
    @testset "LeftUpperTRSM!" begin
        for n in sizes
            for m in m_sizes
                # Prepare CPU matrices
                A = Matrix(UpperTriangular(rand(Float32, n, n) .+ 1.0))
                A .+= Diagonal(10.0 * ones(Float32, n))
                B = rand(Float32, n, m) .+ 1.0

                # Copies for baseline
                Ac = copy(A)
                Bc = copy(B)

                # Move to GPU
                A_gpu = CuArray(A)
                B_gpu = CuArray(B)

                # Call our kernel
                LeftUpperTRSM!(A_gpu, B_gpu)

                # Baseline with BLAS trsm!
                # CUBLAS.BLAS.trsm!('L', 'U', 'N', 'N', 1.0, Ac, Bc)
                CUBLAS.BLAS.trsm!('L','U','N','N', one(eltype(A)), Ac, Bc)

                # Compute relative error
                result_diff = norm(Array(B_gpu) .- Bc) / norm(Bc)
                println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                @test result_diff < tolerance
            end
        end
    end

    ###########################################################################
    # Test RightLowerTRSM! (solves X * A = B with A lower-triangular on the right)
    ###########################################################################
    @testset "RightLowerTRSM!" begin
        for n in sizes
            for m in m_sizes
                # Prepare CPU matrices
                A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1.0))
                A .+= Diagonal(10.0 * ones(Float32, n))
                B = rand(Float32, m, n) .+ 1.0

                # Copies for baseline
                Ac = copy(A)
                Bc = copy(B)

                # Move to GPU
                A_gpu = CuArray(A)
                B_gpu = CuArray(B)

                # Call our kernel
                RightLowerTRSM!(A_gpu, B_gpu)

                # Baseline with BLAS trsm!
                # CUBLAS.BLAS.trsm!('R', 'L', 'N', 'N', 1.0, Ac, Bc)
                CUBLAS.BLAS.trsm!('R','L','N','N', one(eltype(A)), Ac, Bc)

                # Compute relative error
                result_diff = norm(Array(B_gpu) .- Bc) / norm(Bc)
                println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                @test result_diff < tolerance
            end
        end
    end

    ###########################################################################
    # Test RightUpperTRSM! (solves X * A = B with A upper-triangular on the right)
    ###########################################################################
    @testset "RightUpperTRSM!" begin
        for n in sizes
            for m in m_sizes
                # Prepare CPU matrices
                A = Matrix(UpperTriangular(rand(Float32, n, n) .+ 1.0))
                A .+= Diagonal(10.0 * ones(Float32, n))
                B = rand(Float32, m, n) .+ 1.0

                # Copies for baseline
                Ac = copy(A)
                Bc = copy(B)

                # Move to GPU
                A_gpu = CuArray(A)
                B_gpu = CuArray(B)

                # Call our kernel
                RightUpperTRSM!(A_gpu, B_gpu)

                # Baseline with BLAS trsm!
                # CUBLAS.BLAS.trsm!('R', 'U', 'N', 'N', 1.0, Ac, Bc)
                CUBLAS.BLAS.trsm!('R','U','N','N', one(eltype(A)), Ac, Bc)

                # Compute relative error
                result_diff = norm(Array(B_gpu) .- Bc) / norm(Bc)

                println("Size: $n x $n, B size: $(size(B)) | Result Diff (Relative Error): $result_diff")

                @test result_diff < tolerance
            end
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

function run_trsm_benchmark()
    n_sizes = [32, 64, 128, 256, 512, 1024]
    
    println("="^110)
    @printf("%-6s | %-6s | %-18s | %-18s | %-15s\n", "N", "M", "Time Custom (ms)", "Time CUBLAS (ms)", "Speedup (Ref/KA)")
    println("="^110)

    for n in n_sizes
        m = n 
        
        A_rand = rand(Float64, n, n)
        A_host = tril(A_rand) + I * n
        B_host = rand(Float64, n, m)
        
        d_A = CuArray(A_host)
        d_B = CuArray(B_host)
        d_B_ref = CuArray(B_host)
        d_B_init = CuArray(B_host)

        backend = KernelAbstractions.get_backend(d_A)

        op_custom = () -> LeftLowerTRSM!(d_A, d_B)
        reset_custom = () -> copyto!(d_B, d_B_init)
        
        time_custom_ns = benchmark_op(op_custom, reset_custom, backend)
        time_custom_ms = time_custom_ns / 1_000_000

        op_cublas = () -> CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, d_A, d_B_ref)
        reset_cublas = () -> copyto!(d_B_ref, d_B_init)

        time_cublas_ns = benchmark_op(op_cublas, reset_cublas, backend)
        time_cublas_ms = time_cublas_ns / 1_000_000

        ratio = time_cublas_ms / time_custom_ms

        @printf("%6d | %6d | %18.4f | %18.4f | %15.4fx\n", 
                n, m, time_custom_ms, time_cublas_ms, ratio)
        
        CUDA.reclaim()
    end
    println("-"^110)
end

run_trsm_benchmark()