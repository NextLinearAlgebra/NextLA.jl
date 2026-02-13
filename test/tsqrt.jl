@testset "TSQRT" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "m=$m, n=$n, ib=$ib" for (m, n, ib) in [
            (32, 24, 8),
            (48, 32, 16),
            (64, 48, 16),
            (20, 20, 10),
        ]
            A1 = triu(rand(T, n, n))
            A2 = rand(T, m, n)

            # Make A1 well-conditioned
            for i in 1:n
                A1[i, i] += 2 * one(T)
            end

            combined_orig = [A1; A2]

            # NextLA kernel
            A1_n = copy(A1); A2_n = copy(A2)
            Tm_n = zeros(T, ib, n)
            tau_n = zeros(T, n)
            work_n = zeros(T, ib * n)
            NextLA.tsqrt!(m, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)

            # LAPACK reference (l=0 for pentagonal)
            A1_l = copy(A1); A2_l = copy(A2)
            Tm_l = zeros(T, ib, n)
            work_l = zeros(T, ib * n)
            lapack_tpqrt!(T, m, n, 0, ib, A1_l, n, A2_l, m, Tm_l, ib, work_l)

            @test A1_n ≈ A1_l rtol=rtol
            @test A2_n ≈ A2_l rtol=rtol

            # R must remain upper triangular
            for i in 1:n, j in 1:i-1
                @test abs(A1_n[i, j]) < rtol * 100
            end

            # Compare R diagonal magnitudes with standard QR
            Q_ref, R_ref = qr(combined_orig)
            for j in 1:n
                if abs(Matrix(R_ref)[j, j]) > rtol * 10
                    @test abs(abs(A1_n[j, j]) - abs(Matrix(R_ref)[j, j])) <
                          rtol * max(abs(A1_n[j, j]), abs(Matrix(R_ref)[j, j])) * 100
                end
            end
        end

        @testset "Helper matches kernel" begin
            m, n, ib = 32, 24, 8
            A1 = triu(rand(T, n, n)); A2 = rand(T, m, n)
            for i in 1:n; A1[i, i] += one(T); end

            A1_h = copy(A1); A2_h = copy(A2)
            Tm_h = zeros(T, ib, n)
            NextLA.tsqrt!(A1_h, A2_h, Tm_h)

            A1_k = copy(A1); A2_k = copy(A2)
            Tm_k = zeros(T, ib, n)
            tau = zeros(T, n)
            work = zeros(T, ib * n)
            NextLA.tsqrt!(m, n, ib, A1_k, A2_k, Tm_k, tau, work)

            @test A1_h ≈ A1_k rtol=rtol
            @test A2_h ≈ A2_k rtol=rtol
        end
    end

    @testset "Rectangular A1 (m1 ≠ n)" begin
        ET = Float64
        rtol_r = test_rtol(ET)

        @testset "Tall A1: m1=$m1, m2=$m2, n=$n, ib=$ib" for (m1, m2, n, ib) in [
            (32, 48, 16, 4),    # A1 is 32×16 (tall), A2 is 48×16
            (64, 32, 32, 8),    # A1 is 64×32 (tall), A2 is 32×32
        ]
            k = min(m1, n)
            A1 = triu(rand(ET, m1, n))
            A2 = rand(ET, m2, n)
            for i in 1:k; A1[i, i] += 2 * one(ET); end

            A1_n = copy(A1); A2_n = copy(A2)
            Tm_n = zeros(ET, ib, k)
            tau_n = zeros(ET, k)
            work_n = zeros(ET, ib * n)
            NextLA.tsqrt!(m2, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)

            @test all(isfinite.(A1_n))
            @test all(isfinite.(A2_n))
            # R should remain upper triangular in the k×n block
            for i in 1:k, j in 1:i-1
                @test abs(A1_n[i, j]) < rtol_r * 100
            end
        end

        @testset "Wide A1: m1=$m1, m2=$m2, n=$n, ib=$ib" for (m1, m2, n, ib) in [
            (16, 32, 32, 4),    # A1 is 16×32 (wide), A2 is 32×32
            (24, 48, 48, 8),    # A1 is 24×48 (wide), A2 is 48×48
        ]
            k = min(m1, n)
            A1 = triu(rand(ET, m1, n))
            A2 = rand(ET, m2, n)
            for i in 1:k; A1[i, i] += 2 * one(ET); end

            A1_n = copy(A1); A2_n = copy(A2)
            Tm_n = zeros(ET, ib, k)
            tau_n = zeros(ET, k)
            work_n = zeros(ET, ib * n)
            NextLA.tsqrt!(m2, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)

            @test all(isfinite.(A1_n))
            @test all(isfinite.(A2_n))
            for i in 1:k, j in 1:i-1
                @test abs(A1_n[i, j]) < rtol_r * 100
            end
        end

        @testset "Helper with rectangular A1" begin
            # A1 is 32×16 (tall), A2 is 48×16
            m1, n, m2 = 32, 16, 48
            ib = 4; k = min(m1, n)
            A1 = triu(rand(ET, m1, n))
            A2 = rand(ET, m2, n)
            for i in 1:k; A1[i, i] += one(ET); end

            A1_h = copy(A1); A2_h = copy(A2)
            Tm_h = zeros(ET, ib, k)
            NextLA.tsqrt!(A1_h, A2_h, Tm_h)

            A1_k = copy(A1); A2_k = copy(A2)
            Tm_k = zeros(ET, ib, k)
            tau_k = zeros(ET, k)
            work_k = zeros(ET, ib * n)
            NextLA.tsqrt!(m2, n, ib, A1_k, A2_k, Tm_k, tau_k, work_k)

            @test A1_h ≈ A1_k rtol=rtol_r
            @test A2_h ≈ A2_k rtol=rtol_r
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.tsqrt!(-1, 8, 4, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
        @test_throws ArgumentError NextLA.tsqrt!(8, -1, 4, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
        @test_throws ArgumentError NextLA.tsqrt!(8, 8, -1, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
    end

    @testset "Block size sweep" begin
        T = Float64
        m, n = 48, 32
        for ib in [4, 8, 16, 32]
            A1 = triu(rand(T, n, n)); A2 = rand(T, m, n)
            for i in 1:n; A1[i, i] += one(T); end

            Tm = zeros(T, ib, n)
            tau = zeros(T, n)
            work = zeros(T, ib * n)
            NextLA.tsqrt!(m, n, ib, A1, A2, Tm, tau, work)

            @test all(isfinite.(A1))
            @test all(isfinite.(A2))
            for i in 1:n, j in 1:i-1
                @test abs(A1[i, j]) < 1e-12 * 100
            end
        end
    end
end
