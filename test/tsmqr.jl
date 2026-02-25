for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "TSMQR [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
            rtol = test_rtol(T)

            @testset "side=$side, trans=$trans" for side in ['L', 'R'],
                                                    trans in ['N', 'C']
                if side == 'L'
                    m1, n1, m2, k, ib = 32, 24, 20, 16, 8
                    n2 = n1
                else
                    m1, n1, n2, k, ib = 24, 32, 20, 16, 8
                    m2 = m1
                end

                A1 = rand(T, m1, n1)
                A2 = rand(T, m2, n2)
                V = side == 'L' ? rand(T, m2, k) : rand(T, n2, k)
                Tm = triu(rand(T, ib, k))
                work = ArrayType(zeros(T, side == 'L' ? ib * n1 : ib * m1))

                A1_orig = copy(A1)
                A2_orig = copy(A2)

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                V_n = ArrayType(copy(V))
                Tm_n = ArrayType(copy(Tm))
                NextLA.tsmqr!(side, trans, m1, n1, m2, n2, k, ib,
                              A1_n, A2_n, V_n, Tm_n, work)
                synchronize(A1_n)

                A1_l = copy(A1)
                A2_l = copy(A2)
                lapack_tpmqrt!(T, side, trans, 0, V, Tm, A1_l, A2_l)

                @test Array(A1_n) ≈ A1_l rtol=rtol
                @test Array(A2_n) ≈ A2_l rtol=rtol
                @test !isapprox(Array(A1_n), A1_orig; rtol=rtol) || !isapprox(Array(A2_n), A2_orig; rtol=rtol)
            end

            @testset "Helper matches kernel (side=$side)" for side in ['L', 'R']
                if side == 'L'
                    m1, n1, m2, k, ib = 32, 24, 20, 16, 8
                    n2 = n1
                else
                    m1, n1, n2, k, ib = 24, 32, 20, 16, 8
                    m2 = m1
                end

                A1 = rand(T, m1, n1)
                A2 = rand(T, m2, n2)
                V = side == 'L' ? rand(T, m2, k) : rand(T, n2, k)
                Tm = triu(rand(T, ib, k))

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                V_n = ArrayType(copy(V))
                Tm_n = ArrayType(copy(Tm))
                NextLA.tsmqr!(side, 'N', A1_h, A2_h, V_n, Tm_n)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                work = ArrayType(zeros(T, side == 'L' ? ib * n1 : ib * m1))
                NextLA.tsmqr!(side, 'N', m1, n1, m2, n2, k, ib,
                              A1_k, A2_k, V_n, Tm_n, work)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol
                @test A2_h ≈ A2_k rtol=rtol
            end
        end

        @testset "k=0 is a no-op" begin
            A1 = ArrayType(copy(rand(ComplexF64, 8, 8)))
            A2 = ArrayType(copy(rand(ComplexF64, 8, 8)))
            A1_orig = copy(A1)
            A2_orig = copy(A2)
            NextLA.tsmqr!('L', 'N', 8, 8, 8, 8, 0, 0,
                          A1, A2, ArrayType(zeros(ComplexF64, 8, 0)), ArrayType(zeros(ComplexF64, 0, 0)), ArrayType(zeros(ComplexF64, 0)))
            synchronize(A1)
            @test A1 ≈ A1_orig
            @test A2 ≈ A2_orig
        end
    end
end

@testset "TSMQR Error handling" begin
    @test_throws ArgumentError NextLA.tsmqr!('X', 'N', 8, 8, 8, 8, 4, 2,
        zeros(8, 8), zeros(8, 8), zeros(8, 4), zeros(2, 4), zeros(16))
    @test_throws ArgumentError NextLA.tsmqr!('L', 'X', 8, 8, 8, 8, 4, 2,
        zeros(8, 8), zeros(8, 8), zeros(8, 4), zeros(2, 4), zeros(16))
end
