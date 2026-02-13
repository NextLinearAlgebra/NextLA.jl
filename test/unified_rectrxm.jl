using CUDA

@testset "unified_rectrxm! GPU" begin
    if CUDA.functional()
        tol = 1e-14

        @testset "func=$func side=$side uplo=$uplo trans=$trans n=$n m=$m" for
                n     in [16, 32, 128, 256],
                m     in [1, 8, 64],
                side  in ['L', 'R'],
                uplo  in ['L', 'U'],
                trans in ['N', 'T', 'C'],
                func  in ['S', 'M']

            alpha = 1.0

            A = if uplo == 'L'
                Matrix(LowerTriangular(rand(n, n) .+ 1))
            else
                Matrix(UpperTriangular(rand(n, n) .+ 1))
            end
            A += Diagonal(10 * ones(n))

            B = side == 'L' ? rand(n, m) .+ 1 : rand(m, n) .+ 1

            Ac, Bc = copy(A), copy(B)
            A_gpu = CuArray(A)
            B_gpu = CuArray(B)

            unified_rectrxm!(side, uplo, trans, alpha, func, A_gpu, B_gpu)

            if func == 'S'
                CUBLAS.BLAS.trsm!(side, uplo, trans, 'N', alpha, Ac, Bc)
            else
                CUBLAS.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac, Bc)
            end

            rel_err = norm(Array(B_gpu) - Bc) / norm(Bc)
            @test rel_err < tol
        end
    end
end


