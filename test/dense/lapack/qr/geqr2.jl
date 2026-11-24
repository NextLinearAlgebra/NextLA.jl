for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "GEQR2 [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
            rtol = test_rtol(T)

            @testset "Unblocked QR m=$m, n=$n" for (m, n) in [
                    (0, 0), (1, 1), (10, 8), (20, 15), (32, 32), (50, 30),
                ]
                    k = min(m, n)
                    A_init = rand(T, m, n)
                    A_orig = copy(A_init)
                    A = ArrayType(copy(A_init))
                    tau = ArrayType(zeros(T, k))
                    work = ArrayType(zeros(T, max(1, n)))
                    NextLA.geqr2!(m, n, A, tau, work)
                    synchronize(A)

                    if m == 0 || n == 0
                        @test size(A) == size(A_orig)
                    else
                        A_cpu = Array(A)
                        tau_cpu = Array(tau)
                        R = triu(A_cpu[1:k, 1:n])
                        Q = Matrix{T}(I, m, m)
                        LAPACK.ormqr!('L', 'N', A_cpu, tau_cpu, Q)
                        res = opnorm(A_orig - Q[:, 1:k] * R, 1) /
                              (opnorm(A_orig, 1) * n * eps(real(T)))
                        @test res < 10
                        orth = opnorm(Q' * Q - I, 1) / (m * eps(real(T)))
                        @test orth < 10
                    end
            end

            @testset "Helper matches kernel" begin
                m, n = 20, 15
                k = min(m, n)
                A_init = rand(T, m, n)
                A1 = ArrayType(copy(A_init))
                tau1 = ArrayType(zeros(T, k))
                work = ArrayType(zeros(T, n))
                NextLA.geqr2!(m, n, A1, tau1, work)
                synchronize(A1)

                A2 = ArrayType(copy(A_init))
                tau2 = ArrayType(zeros(T, k))
                NextLA.geqr2!(A2, tau2)
                synchronize(A2)

                @test Array(A1) ≈ Array(A2) rtol=rtol
                @test Array(tau1) ≈ Array(tau2) rtol=rtol
            end
        end
    end
end

@testset "GEQR2 Error handling" begin
    @test_throws ArgumentError NextLA.geqr2!(-1, 5, zeros(5, 5), zeros(5), zeros(5))
    @test_throws ArgumentError NextLA.geqr2!(5, -1, zeros(5, 5), zeros(5), zeros(5))
end
