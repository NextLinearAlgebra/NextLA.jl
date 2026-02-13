using CUDA

@testset "TRSM GPU kernels" begin
    if CUDA.functional()
        tol = 1e-5

        @testset "LeftLowerTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                  m in [1, 8, 64]
            A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
            B = rand(Float32, n, m) .+ 1
            Ac, Bc = copy(A), copy(B)
            B_gpu = CuArray(B)

            LeftLowerTRSM!(CuArray(A), B_gpu)
            CUBLAS.BLAS.trsm!('L', 'L', 'N', 'N', 1f0, Ac, Bc)
            @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
        end

        @testset "LeftUpperTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                  m in [1, 8, 64]
            A = Matrix(UpperTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
            B = rand(Float32, n, m) .+ 1
            Ac, Bc = copy(A), copy(B)
            B_gpu = CuArray(B)

            LeftUpperTRSM!(CuArray(A), B_gpu)
            CUBLAS.BLAS.trsm!('L', 'U', 'N', 'N', 1f0, Ac, Bc)
            @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
        end

        @testset "RightLowerTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                   m in [1, 8, 64]
            A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
            B = rand(Float32, m, n) .+ 1
            Ac, Bc = copy(A), copy(B)
            B_gpu = CuArray(B)

            RightLowerTRSM!(CuArray(A), B_gpu)
            CUBLAS.BLAS.trsm!('R', 'L', 'N', 'N', 1f0, Ac, Bc)
            @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
        end

        @testset "RightUpperTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                   m in [1, 8, 64]
            A = Matrix(UpperTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
            B = rand(Float32, m, n) .+ 1
            Ac, Bc = copy(A), copy(B)
            B_gpu = CuArray(B)

            RightUpperTRSM!(CuArray(A), B_gpu)
            CUBLAS.BLAS.trsm!('R', 'U', 'N', 'N', 1f0, Ac, Bc)
            @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
        end
    end
end
