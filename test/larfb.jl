@testset "LARFB" begin
    # Note: NextLA.larfb! currently supports only forward ('F'), columnwise ('C') reflectors.
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "side=$side, trans=$trans, m=$m, n=$n" for
                side  in ['L', 'R'],
                trans in ['N', 'C'],
                (m, n) in [(5, 4), (10, 8), (20, 15)]

            k = min(side == 'L' ? m : n, 4)  # small k for tractable reference
            panel_rows = side == 'L' ? m : n

            # Build valid V and T from NextLA's own geqr2! factorization
            panel = randn(T, panel_rows, k)
            tau_vec = zeros(T, k)
            qr_work = zeros(T, k)
            NextLA.geqr2!(panel_rows, k, panel, tau_vec, qr_work)

            # Extract V: unit lower trapezoidal from factored panel
            V = copy(panel)
            for j in 1:k
                for i in 1:j-1
                    V[i, j] = zero(T)   # zero above diagonal
                end
                V[j, j] = one(T)         # unit diagonal
            end

            # Build T via larft
            T_mat = zeros(T, k, k)
            NextLA.larft!('F', 'C', panel_rows, k, V, tau_vec, T_mat)

            C = randn(T, m, n)
            C_orig = copy(C)

            # Kernel call
            NextLA.larfb!(side, trans, 'F', 'C', V, T_mat, C)

            # Reference: H = I − V·T·Vᴴ
            I_mat = Matrix{T}(I, panel_rows, panel_rows)
            H = I_mat - V * T_mat * V'
            Ht = trans == 'N' ? H : H'

            C_ref = side == 'L' ? Ht * C_orig : C_orig * Ht
            @test C ≈ C_ref rtol=rtol * max(1, m, n)
        end

        @testset "Roundtrip H then Hᴴ" begin
            m, n, k = 12, 10, 3

            # Build valid V and T from proper factorization
            panel = randn(T, m, k)
            tau_vec = zeros(T, k)
            qr_work = zeros(T, k)
            NextLA.geqr2!(m, k, panel, tau_vec, qr_work)

            V = copy(panel)
            for j in 1:k
                for i in 1:j-1; V[i, j] = zero(T); end
                V[j, j] = one(T)
            end
            T_mat = zeros(T, k, k)
            NextLA.larft!('F', 'C', m, k, V, tau_vec, T_mat)

            C = randn(T, m, n)
            C_orig = copy(C)

            NextLA.larfb!('L', 'N', 'F', 'C', V, T_mat, C)
            NextLA.larfb!('L', 'C', 'F', 'C', V, T_mat, C)

            # H·Hᴴ = I for proper Householder reflectors, so C should be recovered
            @test C ≈ C_orig rtol=rtol * m
        end

        @testset "Helper matches kernel" begin
            m, n, k = 10, 8, 3

            # Build valid V and T
            panel = randn(T, m, k)
            tau_vec = zeros(T, k)
            qr_work = zeros(T, k)
            NextLA.geqr2!(m, k, panel, tau_vec, qr_work)

            V = copy(panel)
            for j in 1:k
                for i in 1:j-1; V[i, j] = zero(T); end
                V[j, j] = one(T)
            end
            T_mat = zeros(T, k, k)
            NextLA.larft!('F', 'C', m, k, V, tau_vec, T_mat)

            C = randn(T, m, n)

            C1 = copy(C)
            work = zeros(T, n, k)
            NextLA.larfb!('L', 'N', 'F', 'C', m, n, k, V, m, T_mat, C1, work)

            C2 = copy(C)
            NextLA.larfb!('L', 'N', 'F', 'C', V, T_mat, C2)

            @test C1 ≈ C2 rtol=rtol
        end
    end

    @testset "Edge cases" begin
        # k=0 or m=0 or n=0 → no‑op
        @test_nowarn NextLA.larfb!('L', 'N', 'F', 'C', 0, 0, 0,
            zeros(0, 0), 1, zeros(0, 0), zeros(0, 0), zeros(0, 0))
        @test_nowarn NextLA.larfb!('L', 'N', 'F', 'C', 5, 5, 0,
            zeros(5, 0), 5, zeros(0, 0), randn(5, 5), zeros(5, 0))
    end
end
