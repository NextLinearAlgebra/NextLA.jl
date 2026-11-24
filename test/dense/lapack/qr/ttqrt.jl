for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "TTQRT [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
            rtol = test_rtol(T)

            @testset "n=$n, ib=$ib" for (n, ib) in [
                (16, 4),
                (24, 8),
                (32, 8),
                (48, 16),
            ]
                A1 = triu(randn(T, n, n))
                A2 = triu(randn(T, n, n))

                for i in 1:n
                    A1[i, i] += 2 * one(T)
                    A2[i, i] += 2 * one(T)
                end

                A1_orig = copy(A1)
                A2_orig = copy(A2)

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm_n = ArrayType(zeros(T, ib, n))
                tau_n = ArrayType(zeros(T, n))
                work_n = ArrayType(zeros(T, ib * n))
                NextLA.ttqrt!(n, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)
                synchronize(A1_n)

                A1_l = copy(A1)
                A2_l = copy(A2)
                Tm_l = zeros(T, ib, n)
                work_l = zeros(T, ib * n)
                lapack_tpqrt!(T, n, n, n, ib, A1_l, n, A2_l, n, Tm_l, ib, work_l)

                @test Array(A1_n) ≈ A1_l rtol=rtol
                @test Array(A2_n) ≈ A2_l rtol=rtol
                @test !isapprox(Array(A1_n), A1_orig; rtol=rtol) || !isapprox(Array(A2_n), A2_orig; rtol=rtol)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                Tm_cpu = Array(Tm_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
                @test all(isfinite.(Tm_cpu))
            end

            @testset "Helper matches kernel" begin
                n, ib = 24, 8
                A1 = triu(randn(T, n, n))
                A2 = triu(randn(T, n, n))
                for i in 1:n; A1[i, i] += one(T); A2[i, i] += one(T); end

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                Tm_h = ArrayType(zeros(T, ib, n))
                NextLA.ttqrt!(A1_h, A2_h, Tm_h)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                Tm_k = ArrayType(zeros(T, ib, n))
                tau = ArrayType(zeros(T, n))
                work = ArrayType(zeros(T, ib * n))
                NextLA.ttqrt!(n, n, ib, A1_k, A2_k, Tm_k, tau, work)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol
                @test A2_h ≈ A2_k rtol=rtol
            end
        end

        @testset "Rectangular tiles (m ≠ n)" begin
            ET = Float64
            rtol_r = test_rtol(ET)

            @testset "Tall: m=$m, n=$n, ib=$ib" for (m, n, ib) in [
                (32, 16, 4),
                (48, 24, 8),
                (64, 32, 16),
            ]
                k = min(m, n)
                A1 = triu(randn(ET, m, n))
                A2 = triu(randn(ET, m, n))
                for i in 1:k
                    A1[i, i] += 2 * one(ET)
                    A2[i, i] += 2 * one(ET)
                end

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm_n = ArrayType(zeros(ET, ib, k))
                tau_n = ArrayType(zeros(ET, k))
                work_n = ArrayType(zeros(ET, ib * n))
                NextLA.ttqrt!(m, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)
                synchronize(A1_n)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                Tm_cpu = Array(Tm_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
                @test all(isfinite.(Tm_cpu))
                for i in 1:k, j in 1:i-1
                    @test abs(A1_cpu[i, j]) < rtol_r * 100
                end
            end

            @testset "Wide: m=$m, n=$n, ib=$ib" for (m, n, ib) in [
                (16, 32, 4),
                (24, 48, 8),
            ]
                k = min(m, n)
                A1 = triu(randn(ET, m, n))
                A2 = triu(randn(ET, m, n))
                for i in 1:k
                    A1[i, i] += 2 * one(ET)
                    A2[i, i] += 2 * one(ET)
                end

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm_n = ArrayType(zeros(ET, ib, k))
                tau_n = ArrayType(zeros(ET, k))
                work_n = ArrayType(zeros(ET, ib * n))
                NextLA.ttqrt!(m, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)
                synchronize(A1_n)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                Tm_cpu = Array(Tm_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
                @test all(isfinite.(Tm_cpu))
                for i in 1:k, j in 1:i-1
                    @test abs(A1_cpu[i, j]) < rtol_r * 100
                end
            end

            @testset "Helper rectangular: m=$m, n=$n" for (m, n) in [
                (32, 16),
                (16, 32),
            ]
                k = min(m, n)
                ib = min(8, k)
                A1 = triu(randn(ET, m, n))
                A2 = triu(randn(ET, m, n))
                for i in 1:k; A1[i, i] += one(ET); A2[i, i] += one(ET); end

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                Tm_h = ArrayType(zeros(ET, ib, k))
                NextLA.ttqrt!(A1_h, A2_h, Tm_h)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                Tm_k = ArrayType(zeros(ET, ib, k))
                tau = ArrayType(zeros(ET, k))
                work = ArrayType(zeros(ET, ib * n))
                NextLA.ttqrt!(m, n, ib, A1_k, A2_k, Tm_k, tau, work)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol_r
                @test A2_h ≈ A2_k rtol=rtol_r
            end
        end
    end
end

@testset "TTQRT Error handling" begin
    @test_throws ArgumentError NextLA.ttqrt!(-1, 8, 4, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
    @test_throws ArgumentError NextLA.ttqrt!(8, -1, 4, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
    @test_throws ArgumentError NextLA.ttqrt!(8, 8, -1, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
end
