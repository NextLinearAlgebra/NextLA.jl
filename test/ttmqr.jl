@testset "TTMQR" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "side=$side, trans=$trans" for side in ['L', 'R'],
                                                trans in ['N', 'C']
            n = 24; k = 16; ib = 8

            A1 = randn(T, n, n)
            A2 = randn(T, n, n)
            V = triu(randn(T, n, k))
            Tm = triu(rand(T, ib, k))

            A1_orig = copy(A1); A2_orig = copy(A2)

            # NextLA kernel — use actual side and trans
            A1_n = copy(A1); A2_n = copy(A2)
            work = zeros(T, side == 'L' ? ib * n : n * ib)
            NextLA.ttmqr!(side, trans, n, n, n, n, k, ib,
                          A1_n, A2_n, V, Tm, work)

            # LAPACK reference (l=0 for consistency with parfb l=0 path)
            A1_l = copy(A1); A2_l = copy(A2)
            lapack_tpmqrt!(T, side, trans, 0, V, Tm, A1_l, A2_l)

            @test A1_n ≈ A1_l rtol=rtol
            @test A2_n ≈ A2_l rtol=rtol

            # Matrices must have been modified
            @test !isapprox(A1_n, A1_orig; rtol=rtol) || !isapprox(A2_n, A2_orig; rtol=rtol)

            @test all(isfinite.(A1_n))
            @test all(isfinite.(A2_n))
        end

        @testset "Helper matches kernel (side=$side)" for side in ['L', 'R']
            n = 24; k = 16; ib = 8

            A1 = randn(T, n, n)
            A2 = randn(T, n, n)
            V = triu(randn(T, n, k))
            Tm = triu(rand(T, ib, k))

            A1_h = copy(A1); A2_h = copy(A2)
            NextLA.ttmqr!(side, 'N', A1_h, A2_h, V, Tm)

            A1_k = copy(A1); A2_k = copy(A2)
            work = zeros(T, side == 'L' ? ib * n : n * ib)
            NextLA.ttmqr!(side, 'N', n, n, n, n, k, ib,
                          A1_k, A2_k, V, Tm, work)

            @test A1_h ≈ A1_k rtol=rtol
            @test A2_h ≈ A2_k rtol=rtol
        end

        @testset "trans='T' accepted for real types" begin
            if !(T <: Complex)
                n = 16; k = 8; ib = 4
                A1 = randn(T, n, n)
                A2 = randn(T, n, n)
                V = triu(randn(T, n, k))
                Tm = triu(rand(T, ib, k))
                work = zeros(T, ib * n)
                # Must not throw after the fix
                @test_nowarn NextLA.ttmqr!('L', 'T', n, n, n, n, k, ib,
                                           A1, A2, V, Tm, work)
            end
        end
    end

    @testset "Rectangular tiles (m ≠ n)" begin
        ET = Float64
        rtol_r = test_rtol(ET)

        @testset "Wide A1: side=$side" for side in ['L', 'R']
            # A1/A2 are 16×32 (wide), V is from a ttqrt on 16×32 tiles
            m_tile, n_tile = 16, 32
            k = min(m_tile, n_tile)  # 16
            ib = 4

            A1 = randn(ET, m_tile, n_tile)
            A2 = randn(ET, m_tile, n_tile)
            # For side='L', V is indexed V[1:m2, ...] → m2=m_tile rows
            # For side='R', V is indexed V[1:n2, ...] → n2=n_tile rows
            v_rows = side == 'L' ? m_tile : n_tile
            V = triu(randn(ET, v_rows, k))
            Tm = triu(rand(ET, ib, k))

            if side == 'L'
                work = zeros(ET, ib * n_tile)
                @test_nowarn NextLA.ttmqr!(side, 'N', m_tile, n_tile, m_tile, n_tile,
                                           k, ib, copy(A1), copy(A2), V, Tm, work)
            else
                work = zeros(ET, m_tile * ib)
                @test_nowarn NextLA.ttmqr!(side, 'N', m_tile, n_tile, m_tile, n_tile,
                                           k, ib, copy(A1), copy(A2), V, Tm, work)
            end
        end

        @testset "Helper rectangular: side=$side" for side in ['L', 'R']
            m_tile, n_tile = 16, 32
            k = min(m_tile, n_tile)  # 16
            ib = 4

            A1 = randn(ET, m_tile, n_tile)
            A2 = randn(ET, m_tile, n_tile)
            # For side='L', V is indexed V[1:m2, ...] → m2=m_tile rows
            # For side='R', V is indexed V[1:n2, ...] → n2=n_tile rows
            v_rows = side == 'L' ? m_tile : n_tile
            V = triu(randn(ET, v_rows, k))
            # T_matrix is ib × n_tile (32), but only k=16 reflectors
            Tm_wide = zeros(ET, ib, n_tile)
            Tm_wide[:, 1:k] .= triu(rand(ET, ib, k))

            A1_h = copy(A1); A2_h = copy(A2)
            @test_nowarn NextLA.ttmqr!(side, 'N', A1_h, A2_h, V, Tm_wide)

            # Helper should give same result as kernel with clamped k
            A1_k = copy(A1); A2_k = copy(A2)
            kmax = side == 'L' ? m_tile : n_tile
            k_eff = min(n_tile, kmax, size(V, 2))  # helper clamps
            work = zeros(ET, side == 'L' ? ib * n_tile : m_tile * ib)
            NextLA.ttmqr!(side, 'N', m_tile, n_tile, m_tile, n_tile,
                          k_eff, ib, A1_k, A2_k, V, Tm_wide, work)

            @test A1_h ≈ A1_k rtol=rtol_r
            @test A2_h ≈ A2_k rtol=rtol_r
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.ttmqr!('X', 'N', 8, 8, 8, 8, 4, 2,
            zeros(8, 8), zeros(8, 8), zeros(8, 4), zeros(2, 4), zeros(16))
        @test_throws ArgumentError NextLA.ttmqr!('L', 'X', 8, 8, 8, 8, 4, 2,
            zeros(8, 8), zeros(8, 8), zeros(8, 4), zeros(2, 4), zeros(16))
    end

    @testset "k=0 is a no-op" begin
        A1 = rand(ComplexF64, 8, 8)
        A2 = rand(ComplexF64, 8, 8)
        A1_orig = copy(A1); A2_orig = copy(A2)
        NextLA.ttmqr!('L', 'N', 8, 8, 8, 8, 0, 0,
                       A1, A2, zeros(ComplexF64, 8, 0), zeros(ComplexF64, 0, 0), ComplexF64[])
        @test A1 ≈ A1_orig
        @test A2 ≈ A2_orig
    end
end
