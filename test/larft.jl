@testset "LARFT" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "direct=$direct, storev=$storev, n=$n, k=$k" for
                direct in ['F'],
                storev in ['C', 'R'],
                (n, k) in [(1, 1), (5, 3), (10, 4), (20, 8)]

            # Generate V with the structure larft expects:
            # storev='C': V is n×k unit lower trapezoidal (V[i,i]=1, V[j,i]=0 for j<i)
            # storev='R': V is k×n unit upper trapezoidal (V[i,i]=1, V[i,j]=0 for j<i)
            if storev == 'C'
                V = randn(T, n, k)
                # Set unit lower trapezoidal: zero above diagonal, ones on diagonal
                for j in 1:k
                    for i in 1:j-1
                        V[i, j] = zero(T)
                    end
                    V[j, j] = one(T)
                end
            else
                V = randn(T, k, n)
                # Set unit upper trapezoidal: zero below diagonal, ones on diagonal
                for i in 1:k
                    for j in 1:i-1
                        V[i, j] = zero(T)
                    end
                    V[i, i] = one(T)
                end
            end
            tau = randn(T, k)

            T_mat = zeros(T, k, k)
            NextLA.larft!(direct, storev, n, k, V, tau, T_mat)

            # T must be upper triangular for 'F'
            @test T_mat ≈ UpperTriangular(T_mat)

            # Diagonal should equal tau
            for i in 1:k
                @test T_mat[i, i] ≈ tau[i]
            end

            # Build H_compact = I − V·T·Vᴴ  and verify against product of individual H(i)
            if storev == 'C'
                H_compact = Matrix{T}(I, n, n) - V * T_mat * V'
            else
                H_compact = Matrix{T}(I, n, n) - V' * T_mat * V
            end

            H_true = Matrix{T}(I, n, n)
            for j in 1:k
                if storev == 'C'
                    vj = V[:, j]
                    Hj = I - tau[j] * (vj * vj')
                else
                    vj = V[j, :]          # row of V as column vector
                    Hj = I - tau[j] * (conj(vj) * transpose(vj))
                end
                H_true = direct == 'F' ? H_true * Hj : Hj * H_true
            end

            err = norm(H_compact - H_true) / max(1.0, norm(H_true))
            @test err < rtol * n
        end

        @testset "Helper matches kernel" begin
            n, k = 10, 4
            V = randn(T, n, k)
            tau = randn(T, k)

            T1 = zeros(T, k, k)
            NextLA.larft!('F', 'C', n, k, V, tau, T1)

            T2 = zeros(T, k, k)
            NextLA.larft!('F', 'C', V, tau, T2)

            @test T1 ≈ T2 rtol=rtol
        end
    end

    @testset "Edge cases" begin
        # n=0 → no‑op
        T_mat = ones(2, 2)
        @test_nowarn NextLA.larft!('F', 'C', 0, 2, zeros(0, 2), zeros(2), T_mat)
    end
end

