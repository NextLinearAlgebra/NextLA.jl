for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "LARFT [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
                rtol = test_rtol(T)

                @testset "direct=$direct, storev=$storev, n=$n, k=$k" for
                        direct in ['F'],
                        storev in ['C', 'R'],
                        (n, k) in [(1, 1), (5, 3), (10, 4), (20, 8)]

                    if storev == 'C'
                        V_init = randn(T, n, k)
                        for j in 1:k
                            for i in 1:j-1
                                V_init[i, j] = zero(T)
                            end
                            V_init[j, j] = one(T)
                        end
                    else
                        V_init = randn(T, k, n)
                        for i in 1:k
                            for j in 1:i-1
                                V_init[i, j] = zero(T)
                            end
                            V_init[i, i] = one(T)
                        end
                    end
                    tau = ArrayType(randn(T, k))
                    V = ArrayType(copy(V_init))
                    T_mat = ArrayType(zeros(T, k, k))

                    NextLA.larft!(direct, storev, n, k, V, tau, T_mat)
                    synchronize(T_mat)

                    T_cpu = Array(T_mat)
                    @test T_cpu ≈ UpperTriangular(T_cpu)
                    tau_cpu = Array(tau)
                    for i in 1:k
                        @test T_cpu[i, i] ≈ tau_cpu[i]
                    end

                    V_cpu = Array(V)
                    if storev == 'C'
                        H_compact = Matrix{T}(I, n, n) - V_cpu * T_cpu * V_cpu'
                    else
                        H_compact = Matrix{T}(I, n, n) - V_cpu' * T_cpu * V_cpu
                    end
                    H_true = Matrix{T}(I, n, n)
                    for j in 1:k
                        if storev == 'C'
                            vj = V_cpu[:, j]
                            Hj = I - tau_cpu[j] * (vj * vj')
                        else
                            vj = V_cpu[j, :]
                            Hj = I - tau_cpu[j] * (conj(vj) * transpose(vj))
                        end
                        H_true = direct == 'F' ? H_true * Hj : Hj * H_true
                    end
                    err = norm(H_compact - H_true) / max(1.0, norm(H_true))
                    @test err < rtol * n
                end

                @testset "Helper matches kernel" begin
                    n, k = 10, 4
                    V_init = randn(T, n, k)
                    tau = ArrayType(randn(T, k))
                    V = ArrayType(copy(V_init))
                    T1 = ArrayType(zeros(T, k, k))
                    NextLA.larft!('F', 'C', n, k, V, tau, T1)
                    synchronize(T1)

                    V2 = ArrayType(copy(V_init))
                    T2 = ArrayType(zeros(T, k, k))
                    NextLA.larft!('F', 'C', V2, tau, T2)
                    synchronize(T2)

                    @test Array(T1) ≈ Array(T2) rtol=rtol
                end
        end
    end
end

@testset "LARFT Edge cases" begin
    T_mat = ones(2, 2)
    @test_nowarn NextLA.larft!('F', 'C', 0, 2, zeros(0, 2), zeros(2), T_mat)
end
