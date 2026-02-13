@testset "GEQRT" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "Blocked QR m=$m, n=$n, ib=$ib" for (m, n, ib) in [
            (20, 15, 4),
            (32, 32, 8),
            (64, 32, 16),
            (32, 64, 8),
        ]
            k = min(m, n)
            A = rand(T, m, n)
            A_orig = copy(A)

            # Kernel call
            Tm = zeros(T, ib, k)
            tau = zeros(T, k)
            work = zeros(T, ib * n)
            NextLA.geqrt!(m, n, ib, A, Tm, tau, work)

            # Extract R from upper triangle (lower part stores reflectors)
            R = triu(A[1:k, 1:n])

            # Reconstruct Q via LAPACK ormqr
            Q = Matrix{T}(I, m, m)
            LAPACK.ormqr!('L', 'N', A, tau, Q)

            # Residual: ‖A_orig − Q*R‖ / (‖A_orig‖·n·ε)
            res = opnorm(A_orig - Q[:, 1:k] * R, 1) /
                  (opnorm(A_orig, 1) * n * eps(real(T)))
            @test res < 10

            # Orthogonality: ‖Qᴴ Q − I‖ / (m·ε)
            orth = opnorm(Q' * Q - I, 1) / (m * eps(real(T)))
            @test orth < 10
        end

        @testset "Helper matches kernel" begin
            m, n, ib = 32, 24, 8
            k = min(m, n)
            A = rand(T, m, n)

            A1 = copy(A)
            Tm1 = zeros(T, ib, k)
            NextLA.geqrt!(A1, Tm1)

            A2 = copy(A)
            Tm2 = zeros(T, ib, k)
            tau = zeros(T, k)
            work = zeros(T, ib * n)
            NextLA.geqrt!(m, n, ib, A2, Tm2, tau, work)

            @test A1 ≈ A2 rtol=rtol
            @test Tm1 ≈ Tm2 rtol=rtol
        end

        @testset "ib=1 matches unblocked" begin
            m, n = 16, 12
            A = rand(T, m, n)

            A_blocked = copy(A)
            Tm = zeros(T, 1, min(m, n))
            tau_b = zeros(T, min(m, n))
            work_b = zeros(T, n)
            NextLA.geqrt!(m, n, 1, A_blocked, Tm, tau_b, work_b)

            A_unblocked = copy(A)
            tau_u = zeros(T, min(m, n))
            work_u = zeros(T, n)
            NextLA.geqr2!(m, n, A_unblocked, tau_u, work_u)

            @test A_blocked ≈ A_unblocked rtol=rtol
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.geqrt!(-1, 5, 2, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
        @test_throws ArgumentError NextLA.geqrt!(5, -1, 2, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
        @test_throws ArgumentError NextLA.geqrt!(5, 5, -1, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
        @test_throws ArgumentError NextLA.geqrt!(5, 5, 0, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
    end

    @testset "Deterministic" begin
        m, n, ib = 20, 15, 4
        k = min(m, n)
        A = rand(ComplexF64, m, n)

        A1 = copy(A); Tm1 = zeros(ComplexF64, ib, k)
        NextLA.geqrt!(A1, Tm1)

        A2 = copy(A); Tm2 = zeros(ComplexF64, ib, k)
        NextLA.geqrt!(A2, Tm2)

        @test A1 ≈ A2
        @test Tm1 ≈ Tm2
    end
end
