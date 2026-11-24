for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "TSQRT [$backend_name]" begin
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

                for i in 1:n
                    A1[i, i] += 2 * one(T)
                end

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm_n = ArrayType(zeros(T, ib, n))
                tau_n = ArrayType(zeros(T, n))
                work_n = ArrayType(zeros(T, ib * n))
                NextLA.tsqrt!(m, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)
                synchronize(A1_n)

                A1_l = copy(A1)
                A2_l = copy(A2)
                Tm_l = zeros(T, ib, n)
                work_l = zeros(T, ib * n)
                lapack_tpqrt!(T, m, n, 0, ib, A1_l, n, A2_l, m, Tm_l, ib, work_l)

                @test Array(A1_n) ≈ A1_l rtol=rtol
                @test Array(A2_n) ≈ A2_l rtol=rtol

                A1_cpu = Array(A1_n)
                for i in 1:n, j in 1:i-1
                    @test abs(A1_cpu[i, j]) < rtol * 100
                end

                # R diagonal magnitudes vs LAPACK tpqrt (same check as qr() baseline but independent reference)
                for j in 1:n
                    if abs(A1_l[j, j]) > rtol * 10
                        @test abs(abs(A1_cpu[j, j]) - abs(A1_l[j, j])) <
                              rtol * max(abs(A1_cpu[j, j]), abs(A1_l[j, j])) * 100
                    end
                end
            end

            @testset "Helper matches kernel" begin
                m, n, ib = 32, 24, 8
                A1 = triu(rand(T, n, n))
                A2 = rand(T, m, n)
                for i in 1:n; A1[i, i] += one(T); end

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                Tm_h = ArrayType(zeros(T, ib, n))
                NextLA.tsqrt!(A1_h, A2_h, Tm_h)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                Tm_k = ArrayType(zeros(T, ib, n))
                tau = ArrayType(zeros(T, n))
                work = ArrayType(zeros(T, ib * n))
                NextLA.tsqrt!(m, n, ib, A1_k, A2_k, Tm_k, tau, work)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol
                @test A2_h ≈ A2_k rtol=rtol
            end
        end

        @testset "Rectangular A1 (m1 ≠ n)" begin
            ET = Float64
            rtol_r = test_rtol(ET)

            @testset "Tall A1: m1=$m1, m2=$m2, n=$n, ib=$ib" for (m1, m2, n, ib) in [
                (32, 48, 16, 4),
                (64, 32, 32, 8),
            ]
                k = min(m1, n)
                A1 = triu(rand(ET, m1, n))
                A2 = rand(ET, m2, n)
                for i in 1:k; A1[i, i] += 2 * one(ET); end

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm_n = ArrayType(zeros(ET, ib, k))
                tau_n = ArrayType(zeros(ET, k))
                work_n = ArrayType(zeros(ET, ib * n))
                NextLA.tsqrt!(m2, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)
                synchronize(A1_n)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
                for i in 1:k, j in 1:i-1
                    @test abs(A1_cpu[i, j]) < rtol_r * 100
                end
            end

            @testset "Wide A1: m1=$m1, m2=$m2, n=$n, ib=$ib" for (m1, m2, n, ib) in [
                (16, 32, 32, 4),
                (24, 48, 48, 8),
            ]
                k = min(m1, n)
                A1 = triu(rand(ET, m1, n))
                A2 = rand(ET, m2, n)
                for i in 1:k; A1[i, i] += 2 * one(ET); end

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm_n = ArrayType(zeros(ET, ib, k))
                tau_n = ArrayType(zeros(ET, k))
                work_n = ArrayType(zeros(ET, ib * n))
                NextLA.tsqrt!(m2, n, ib, A1_n, A2_n, Tm_n, tau_n, work_n)
                synchronize(A1_n)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
                for i in 1:k, j in 1:i-1
                    @test abs(A1_cpu[i, j]) < rtol_r * 100
                end
            end

            @testset "Helper with rectangular A1" begin
                m1, n, m2 = 32, 16, 48
                ib = 4
                k = min(m1, n)
                A1 = triu(rand(ET, m1, n))
                A2 = rand(ET, m2, n)
                for i in 1:k; A1[i, i] += one(ET); end

                A1_h = ArrayType(copy(A1))
                A2_h = ArrayType(copy(A2))
                Tm_h = ArrayType(zeros(ET, ib, k))
                NextLA.tsqrt!(A1_h, A2_h, Tm_h)
                synchronize(A1_h)

                A1_k = ArrayType(copy(A1))
                A2_k = ArrayType(copy(A2))
                Tm_k = ArrayType(zeros(ET, ib, k))
                tau_k = ArrayType(zeros(ET, k))
                work_k = ArrayType(zeros(ET, ib * n))
                NextLA.tsqrt!(m2, n, ib, A1_k, A2_k, Tm_k, tau_k, work_k)
                synchronize(A1_k)

                @test A1_h ≈ A1_k rtol=rtol_r
                @test A2_h ≈ A2_k rtol=rtol_r
            end
        end

        @testset "Block size sweep" begin
            T = Float64
            m, n = 48, 32
            for ib in [4, 8, 16, 32]
                A1 = triu(rand(T, n, n))
                A2 = rand(T, m, n)
                for i in 1:n; A1[i, i] += one(T); end

                A1_n = ArrayType(copy(A1))
                A2_n = ArrayType(copy(A2))
                Tm = ArrayType(zeros(T, ib, n))
                tau = ArrayType(zeros(T, n))
                work = ArrayType(zeros(T, ib * n))
                NextLA.tsqrt!(m, n, ib, A1_n, A2_n, Tm, tau, work)
                synchronize(A1_n)

                A1_cpu = Array(A1_n)
                A2_cpu = Array(A2_n)
                @test all(isfinite.(A1_cpu))
                @test all(isfinite.(A2_cpu))
                for i in 1:n, j in 1:i-1
                    @test abs(A1_cpu[i, j]) < 1e-12 * 100
                end
            end
        end
    end
end

@testset "TSQRT Error handling" begin
    @test_throws ArgumentError NextLA.tsqrt!(-1, 8, 4, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
    @test_throws ArgumentError NextLA.tsqrt!(8, -1, 4, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
    @test_throws ArgumentError NextLA.tsqrt!(8, 8, -1, zeros(8, 8), zeros(8, 8), zeros(4, 8), zeros(8), zeros(32))
end
