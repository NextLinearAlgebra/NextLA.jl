for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "LARF [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
                rtol = test_rtol(T)

                @testset "side=$side, m=$m, n=$n" for side in ['L', 'R'],
                                                       (m, n) in [(1, 1), (5, 4), (10, 8), (20, 15)]
                    v_len = side == 'L' ? m : n
                    C_init = randn(T, m, n)
                    C_orig = copy(C_init)
                    v = ArrayType(randn(T, v_len))
                    tau = randn(T)
                    work = ArrayType(zeros(T, side == 'L' ? n : m))
                    C = ArrayType(copy(C_init))

                    NextLA.larf!(side, m, n, v, 1, tau, C, work)
                    synchronize(C)

                    v_cpu = Array(v)
                    H = I - tau * (v_cpu * v_cpu')
                    C_ref = side == 'L' ? H * C_orig : C_orig * H
                    @test Array(C) ≈ C_ref rtol=rtol
                end

                @testset "Helper matches kernel" begin
                    m, n = 10, 8
                    C_init = randn(T, m, n)
                    v = ArrayType(randn(T, m))
                    tau = randn(T)
                    C1 = ArrayType(copy(C_init))
                    work = ArrayType(zeros(T, n))
                    NextLA.larf!('L', m, n, v, 1, tau, C1, work)
                    synchronize(C1)

                    C2 = ArrayType(copy(C_init))
                    NextLA.larf!('L', v, 1, tau, C2)
                    synchronize(C2)

                    @test Array(C1) ≈ Array(C2) rtol=rtol
                end

                @testset "τ=0 leaves C unchanged" begin
                    C_init = randn(T, 6, 5)
                    C = ArrayType(copy(C_init))
                    v = ArrayType(randn(T, 6))
                    work = ArrayType(zeros(T, 5))
                    NextLA.larf!('L', 6, 5, v, 1, zero(T), C, work)
                    synchronize(C)
                    @test Array(C) == C_init
                end
        end
    end
end
