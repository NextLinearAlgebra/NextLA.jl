using Test
using CUDA
using LinearAlgebra


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
