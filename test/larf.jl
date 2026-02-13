@testset "LARF" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "side=$side, m=$m, n=$n" for side in ['L', 'R'],
                                               (m, n) in [(1, 1), (5, 4), (10, 8), (20, 15)]
            v_len = side == 'L' ? m : n
            C = randn(T, m, n)
            v = randn(T, v_len)
            tau = randn(T)
            work = zeros(T, side == 'L' ? n : m)

            C_orig = copy(C)

            # Kernel call
            NextLA.larf!(side, m, n, v, 1, tau, C, work)

            # Reference: H = I − τ·v·vᴴ, then H*C (left) or C*H (right)
            H = I - tau * (v * v')
            C_ref = side == 'L' ? H * C_orig : C_orig * H

            @test C ≈ C_ref rtol=rtol
        end

        @testset "Helper matches kernel" begin
            m, n = 10, 8
            C = randn(T, m, n)
            v = randn(T, m)
            tau = randn(T)

            C1 = copy(C)
            work = zeros(T, n)
            NextLA.larf!('L', m, n, v, 1, tau, C1, work)

            C2 = copy(C)
            NextLA.larf!('L', v, 1, tau, C2)

            @test C1 ≈ C2 rtol=rtol
        end

        @testset "τ=0 leaves C unchanged" begin
            C = randn(T, 6, 5);  C0 = copy(C)
            NextLA.larf!('L', 6, 5, randn(T, 6), 1, zero(T), C, zeros(T, 5))
            @test C == C0
        end
    end
end
