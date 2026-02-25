for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "TTMQR [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
            rtol = test_rtol(T)

            @testset "side=$side, trans=$trans" for side in ['L', 'R'],
                                                    trans in ['N', 'C']
                n = 24
                k = 16
                ib = 8

                A1 = randn(T, n, n)
                A2 = randn(T, n, n)
                V = triu(randn(T, n, k))
                Tm = triu(rand(T, ib, k))

                A1_orig = copy(A1)
                A2_orig = copy(A2)

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                V_n = ArrayType(copy(V))
                Tm_n = ArrayType(copy(Tm))
                work = ArrayType(zeros(T, side == 'L' ? ib * n : n * ib))
                NextLA.ttmqr!(side, trans, n, n, n, n, k, ib,
                              A1_n, A2_n, V_n, Tm_n, work)
                synchronize(A1_n)

                A1_l = copy(A1)
                A2_l = copy(A2)
                lapack_tpmqrt!(T, side, trans, 0, V, Tm, A1_l, A2_l)

                @test Array(A1_n) ≈ A1_l rtol=rtol
                @test Array(A2_n) ≈ A2_l rtol=rtol
                @test !isapprox(Array(A1_n), A1_orig; rtol=rtol) || !isapprox(Array(A2_n), A2_orig; rtol=rtol)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
            end

            @testset "Helper matches kernel (side=$side)" for side in ['L', 'R']
                n = 24
                k = 16
                ib = 8

                A1 = randn(T, n, n)
                A2 = randn(T, n, n)
                V = triu(randn(T, n, k))
                Tm = triu(rand(T, ib, k))

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                V_n = ArrayType(copy(V))
                Tm_n = ArrayType(copy(Tm))
                NextLA.ttmqr!(side, 'N', A1_h, A2_h, V_n, Tm_n)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                work = ArrayType(zeros(T, side == 'L' ? ib * n : n * ib))
                NextLA.ttmqr!(side, 'N', n, n, n, n, k, ib,
                              A1_k, A2_k, V_n, Tm_n, work)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol
                @test A2_h ≈ A2_k rtol=rtol
            end

            @testset "trans='T' accepted for real types" begin
                if !(T <: Complex)
                    n = 16
                    k = 8
                    ib = 4
                    A1 = ArrayType(copy(randn(T, n, n)))
                    A2 = ArrayType(copy(randn(T, n, n)))
                    V = ArrayType(copy(triu(randn(T, n, k))))
                    Tm = ArrayType(copy(triu(rand(T, ib, k))))
                    work = ArrayType(zeros(T, ib * n))
                    @test_nowarn NextLA.ttmqr!('L', 'T', n, n, n, n, k, ib,
                                               A1, A2, V, Tm, work)
                end
            end
        end

        @testset "Rectangular tiles (m ≠ n)" begin
            ET = Float64
            rtol_r = test_rtol(ET)

            @testset "Wide A1: side=$side" for side in ['L', 'R']
                m_tile, n_tile = 16, 32
                k = min(m_tile, n_tile)
                ib = 4

                A1 = randn(ET, m_tile, n_tile)
                A2 = randn(ET, m_tile, n_tile)
                v_rows = side == 'L' ? m_tile : n_tile
                V = triu(randn(ET, v_rows, k))
                Tm = triu(rand(ET, ib, k))

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                V_n = ArrayType(copy(V))
                Tm_n = ArrayType(copy(Tm))
                if side == 'L'
                    work = ArrayType(zeros(ET, ib * n_tile))
                else
                    work = ArrayType(zeros(ET, m_tile * ib))
                end
                @test_nowarn NextLA.ttmqr!(side, 'N', m_tile, n_tile, m_tile, n_tile,
                                           k, ib, A1_n, A2_n, V_n, Tm_n, work)
            end

            @testset "Helper rectangular: side=$side" for side in ['L', 'R']
                m_tile, n_tile = 16, 32
                k = min(m_tile, n_tile)
                ib = 4

                A1 = randn(ET, m_tile, n_tile)
                A2 = randn(ET, m_tile, n_tile)
                v_rows = side == 'L' ? m_tile : n_tile
                V_cpu = triu(randn(ET, v_rows, k))
                Tm_cpu = zeros(ET, ib, n_tile)
                Tm_cpu[:, 1:k] .= triu(rand(ET, ib, k))
                V_n = ArrayType(copy(V_cpu))
                Tm_wide = ArrayType(copy(Tm_cpu))

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                @test_nowarn NextLA.ttmqr!(side, 'N', A1_h, A2_h, V_n, Tm_wide)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                kmax = side == 'L' ? m_tile : n_tile
                k_eff = min(n_tile, kmax, size(V_n, 2))
                work = ArrayType(zeros(ET, side == 'L' ? ib * n_tile : m_tile * ib))
                NextLA.ttmqr!(side, 'N', m_tile, n_tile, m_tile, n_tile,
                              k_eff, ib, A1_k, A2_k, V_n, Tm_wide, work)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol_r
                @test A2_h ≈ A2_k rtol=rtol_r
            end
        end

        @testset "k=0 is a no-op" begin
            A1 = ArrayType(copy(rand(ComplexF64, 8, 8)))
            A2 = ArrayType(copy(rand(ComplexF64, 8, 8)))
            A1_orig = copy(A1)
            A2_orig = copy(A2)
            NextLA.ttmqr!('L', 'N', 8, 8, 8, 8, 0, 0,
                          A1, A2, ArrayType(zeros(ComplexF64, 8, 0)), ArrayType(zeros(ComplexF64, 0, 0)), ArrayType(zeros(ComplexF64, 0)))
            synchronize(A1)
            @test A1 ≈ A1_orig
            @test A2 ≈ A2_orig
        end
    end
end

@testset "TTMQR Error handling" begin
    @test_throws ArgumentError NextLA.ttmqr!('X', 'N', 8, 8, 8, 8, 4, 2,
        zeros(8, 8), zeros(8, 8), zeros(8, 4), zeros(2, 4), zeros(16))
    @test_throws ArgumentError NextLA.ttmqr!('L', 'X', 8, 8, 8, 8, 4, 2,
        zeros(8, 8), zeros(8, 8), zeros(8, 4), zeros(2, 4), zeros(16))
end
