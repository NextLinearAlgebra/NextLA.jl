@testset "GEQR2" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "Unblocked QR m=$m, n=$n" for (m, n) in [
            (0, 0), (1, 1), (10, 8), (20, 15), (32, 32), (50, 30),
        ]
            k = min(m, n)
            A = rand(T, m, n)
            A_orig = copy(A)

            tau  = zeros(T, k)
            work = zeros(T, max(1, n))
            NextLA.geqr2!(m, n, A, tau, work)

            if m == 0 || n == 0
                @test size(A) == size(A_orig)
            else
                # Extract R from upper triangle (lower part stores reflectors)
                R = triu(A[1:k, 1:n])

                # Reconstruct Q via LAPACK ormqr
                Q = Matrix{T}(I, m, m)
                LAPACK.ormqr!('L', 'N', A, tau, Q)

                # Residual: ‖A_orig − Q·R‖ / (‖A_orig‖·n·ε)
                res = opnorm(A_orig - Q[:, 1:k] * R, 1) /
                      (opnorm(A_orig, 1) * n * eps(real(T)))
                @test res < 10

                # Orthogonality: ‖QᴴQ − I‖ / (m·ε)
                orth = opnorm(Q' * Q - I, 1) / (m * eps(real(T)))
                @test orth < 10
            end
        end

        @testset "Helper matches kernel" begin
            m, n = 20, 15
            k = min(m, n)
            A = rand(T, m, n)

            A1 = copy(A)
            tau1 = zeros(T, k)
            work = zeros(T, n)
            NextLA.geqr2!(m, n, A1, tau1, work)

            A2 = copy(A)
            tau2 = zeros(T, k)
            NextLA.geqr2!(A2, tau2)

            @test A1 ≈ A2 rtol=rtol
            @test tau1 ≈ tau2 rtol=rtol
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.geqr2!(-1, 5, zeros(5, 5), zeros(5), zeros(5))
        @test_throws ArgumentError NextLA.geqr2!(5, -1, zeros(5, 5), zeros(5), zeros(5))
    end
end
