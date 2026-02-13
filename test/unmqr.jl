@testset "UNMQR" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)
        m, n, k, ib = 64, 48, 32, 8

        # Build a valid QR factorization first
        A_qr = rand(T, m, k)
        A_orig = copy(A_qr)
        Tm = zeros(T, ib, k)
        tau = zeros(T, k)
        work_qr = zeros(T, ib * k)
        NextLA.geqrt!(m, k, ib, A_qr, Tm, tau, work_qr)

        @testset "Left, No-Transpose" begin
            C = rand(T, m, n)
            C_orig = copy(C)

            # Kernel call
            work = zeros(T, n, ib)
            NextLA.unmqr!('L', 'N', m, n, k, ib, A_qr, m, Tm, C, work)

            # Orthogonal transform preserves Frobenius norm
            @test norm(C) ≈ norm(C_orig) rtol=rtol

            # Helper must match kernel
            C_h = copy(C_orig)
            NextLA.unmqr!('L', 'N', A_qr, Tm, C_h)
            @test C_h ≈ C rtol=rtol
        end

        @testset "Left, Conjugate-Transpose" begin
            C = rand(T, m, n)
            C_orig = copy(C)

            work = zeros(T, n, ib)
            NextLA.unmqr!('L', 'C', m, n, k, ib, A_qr, m, Tm, C, work)

            @test norm(C) ≈ norm(C_orig) rtol=rtol

            C_h = copy(C_orig)
            NextLA.unmqr!('L', 'C', A_qr, Tm, C_h)
            @test C_h ≈ C rtol=rtol
        end

        @testset "Right, No-Transpose" begin
            C = rand(T, n, m)
            C_orig = copy(C)

            work = zeros(T, n, ib)
            NextLA.unmqr!('R', 'N', n, m, k, ib, A_qr, m, Tm, C, work)

            @test norm(C) ≈ norm(C_orig) rtol=rtol
        end

        @testset "Right, Conjugate-Transpose" begin
            C = rand(T, n, m)
            C_orig = copy(C)

            work = zeros(T, n, ib)
            NextLA.unmqr!('R', 'C', n, m, k, ib, A_qr, m, Tm, C, work)

            @test norm(C) ≈ norm(C_orig) rtol=rtol
        end

        @testset "Q * Qᴴ = I (orthogonality)" begin
            C = Matrix{T}(I, m, m)
            work = zeros(T, m, ib)

            NextLA.unmqr!('L', 'N', m, m, k, ib, A_qr, m, Tm, C, work)
            NextLA.unmqr!('L', 'C', m, m, k, ib, A_qr, m, Tm, C, work)

            @test C ≈ Matrix{T}(I, m, m) rtol=rtol
        end
    end

    @testset "Rectangular tile: k clamped to nq" begin
        # Simulates a wide tile scenario where T_matrix has more columns (k)
        # than the tile has rows (nq). The kernel should clamp k = min(k, nq).
        for T in (Float64, ComplexF64)
            m_tile, n_tile = 16, 32  # wide tile: only 16 reflectors possible
            ib = 4
            # Build a valid QR with 16 reflectors from a 16×16 sub-tile
            k_actual = m_tile
            A_qr = rand(T, m_tile, k_actual)
            Tm = zeros(T, ib, k_actual)
            tau = zeros(T, k_actual)
            work_qr = zeros(T, ib * k_actual)
            NextLA.geqrt!(m_tile, k_actual, ib, A_qr, Tm, tau, work_qr)

            # Apply from left with k = 32 (wider than nq = 16) → should clamp
            C = rand(T, m_tile, 24)
            C_orig = copy(C)
            work = zeros(T, 24, ib)
            # k=n_tile=32 but nq=m_tile=16, must not throw
            @test_nowarn NextLA.unmqr!('L', 'N', m_tile, 24, n_tile, ib,
                                       A_qr, m_tile, Tm, C, work)
            # Frobenius norm preserved by orthogonal transform
            @test norm(C) ≈ norm(C_orig) rtol=test_rtol(T)

            # Helper path: T_matrix is ib×32 but A is 16×16
            Tm_wide = zeros(T, ib, n_tile)  # wider than needed
            Tm_wide[:, 1:k_actual] .= Tm
            C2 = copy(C_orig)
            @test_nowarn NextLA.unmqr!('L', 'N', A_qr, Tm_wide, C2)
            @test norm(C2) ≈ norm(C_orig) rtol=test_rtol(T)
        end
    end

    @testset "Error handling" begin
        m, n, k, ib = 32, 24, 16, 4
        A = zeros(ComplexF64, m, k)
        Tm = zeros(ComplexF64, ib, k)
        C = zeros(ComplexF64, m, n)
        work = zeros(ComplexF64, n, ib)

        @test_throws ArgumentError NextLA.unmqr!('X', 'N', m, n, k, ib, A, m, Tm, C, work)
        @test_throws ArgumentError NextLA.unmqr!('L', 'X', m, n, k, ib, A, m, Tm, C, work)
        @test_throws ArgumentError NextLA.unmqr!('L', 'N', -1, n, k, ib, A, m, Tm, C, work)
        # k > nq should now be clamped, not throw
        C_test = zeros(ComplexF64, m, n)
        @test_nowarn NextLA.unmqr!('L', 'N', m, n, m+1, ib, A, m, Tm, C_test, work)
    end

    @testset "k=0 is a no-op" begin
        C = rand(ComplexF64, 32, 24)
        C_orig = copy(C)
        A = zeros(ComplexF64, 32, 1)
        Tm = zeros(ComplexF64, 4, 1)
        work = zeros(ComplexF64, 24, 4)

        NextLA.unmqr!('L', 'N', 32, 24, 0, 4, A, 32, Tm, C, work)
        @test C ≈ C_orig
    end
end
