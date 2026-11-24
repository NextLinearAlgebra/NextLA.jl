@testset "TRSM GPU kernels" begin
    backends = available_gpu_backends()
    if isempty(backends)
        @test_skip "No GPU backends available"
    end
    for (backend_name, ArrayType, synchronize) in backends
        @testset "[$backend_name]" begin
            tol = 1e-5

            @testset "LeftLowerTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                      m in [1, 8, 64]
                A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
                B = rand(Float32, n, m) .+ 1
                Ac, Bc = copy(A), copy(B)
                A_gpu = ArrayType(A)
                B_gpu = ArrayType(B)

                LeftLowerTRSM!(A_gpu, B_gpu)
                synchronize(B_gpu)
                LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'N', 1f0, Ac, Bc)
                @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
            end

            @testset "LeftUpperTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                      m in [1, 8, 64]
                A = Matrix(UpperTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
                B = rand(Float32, n, m) .+ 1
                Ac, Bc = copy(A), copy(B)
                A_gpu = ArrayType(A)
                B_gpu = ArrayType(B)

                LeftUpperTRSM!(A_gpu, B_gpu)
                synchronize(B_gpu)
                LinearAlgebra.BLAS.trsm!('L', 'U', 'N', 'N', 1f0, Ac, Bc)
                @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
            end

            @testset "RightLowerTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                     m in [1, 8, 64]
                A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
                B = rand(Float32, m, n) .+ 1
                Ac, Bc = copy(A), copy(B)
                A_gpu = ArrayType(A)
                B_gpu = ArrayType(B)

                RightLowerTRSM!(A_gpu, B_gpu)
                synchronize(B_gpu)
                LinearAlgebra.BLAS.trsm!('R', 'L', 'N', 'N', 1f0, Ac, Bc)
                @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
            end

            @testset "RightUpperTRSM! n=$n, m=$m" for n in [16, 32, 128, 256],
                                                     m in [1, 8, 64]
                A = Matrix(UpperTriangular(rand(Float32, n, n) .+ 1)) .+ Diagonal(10f0 * ones(Float32, n))
                B = rand(Float32, m, n) .+ 1
                Ac, Bc = copy(A), copy(B)
                A_gpu = ArrayType(A)
                B_gpu = ArrayType(B)

                RightUpperTRSM!(A_gpu, B_gpu)
                synchronize(B_gpu)
                LinearAlgebra.BLAS.trsm!('R', 'U', 'N', 'N', 1f0, Ac, Bc)
                @test norm(Array(B_gpu) - Bc) / norm(Bc) < tol
            end
        end
    end
end
