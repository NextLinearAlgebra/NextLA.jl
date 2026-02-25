for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "GEQRT [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
                rtol = test_rtol(T)

                @testset "Blocked QR m=$m, n=$n, ib=$ib" for (m, n, ib) in [
                    (20, 15, 4),
                    (32, 32, 8),
                    (64, 32, 16),
                    (32, 64, 8),
                ]
                    k = min(m, n)
                    A_init = rand(T, m, n)
                    A_orig = copy(A_init)
                    A = ArrayType(copy(A_init))
                    Tm = ArrayType(zeros(T, ib, k))
                    tau = ArrayType(zeros(T, k))
                    work = ArrayType(zeros(T, ib * n))
                    NextLA.geqrt!(m, n, ib, A, Tm, tau, work)
                    synchronize(A)

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

                @testset "Helper matches kernel" begin
                    m, n, ib = 32, 24, 8
                    k = min(m, n)
                    A_init = rand(T, m, n)
                    A1 = ArrayType(copy(A_init))
                    Tm1 = ArrayType(zeros(T, ib, k))
                    NextLA.geqrt!(A1, Tm1)
                    synchronize(A1)

                    A2 = ArrayType(copy(A_init))
                    Tm2 = ArrayType(zeros(T, ib, k))
                    tau = ArrayType(zeros(T, k))
                    work = ArrayType(zeros(T, ib * n))
                    NextLA.geqrt!(m, n, ib, A2, Tm2, tau, work)
                    synchronize(A2)

                    @test Array(A1) ≈ Array(A2) rtol=rtol
                    @test Array(Tm1) ≈ Array(Tm2) rtol=rtol
                end

                @testset "ib=1 matches unblocked" begin
                    m, n = 16, 12
                    A_init = rand(T, m, n)
                    A_blocked = ArrayType(copy(A_init))
                    Tm = ArrayType(zeros(T, 1, min(m, n)))
                    tau_b = ArrayType(zeros(T, min(m, n)))
                    work_b = ArrayType(zeros(T, n))
                    NextLA.geqrt!(m, n, 1, A_blocked, Tm, tau_b, work_b)
                    synchronize(A_blocked)

                    A_unblocked = ArrayType(copy(A_init))
                    tau_u = ArrayType(zeros(T, min(m, n)))
                    work_u = ArrayType(zeros(T, n))
                    NextLA.geqr2!(m, n, A_unblocked, tau_u, work_u)
                    synchronize(A_unblocked)

                    @test Array(A_blocked) ≈ Array(A_unblocked) rtol=rtol
                end
        end
    end
end

@testset "GEQRT Error handling" begin
    @test_throws ArgumentError NextLA.geqrt!(-1, 5, 2, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
    @test_throws ArgumentError NextLA.geqrt!(5, -1, 2, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
    @test_throws ArgumentError NextLA.geqrt!(5, 5, -1, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
    @test_throws ArgumentError NextLA.geqrt!(5, 5, 0, zeros(5, 5), zeros(2, 5), zeros(5), zeros(10))
end

for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "GEQRT Deterministic [$backend_name]" begin
        m, n, ib = 20, 15, 4
        k = min(m, n)
        A_init = rand(ComplexF64, m, n)
        A1 = ArrayType(copy(A_init))
        Tm1 = ArrayType(zeros(ComplexF64, ib, k))
        tau1 = ArrayType(zeros(ComplexF64, k))
        work1 = ArrayType(zeros(ComplexF64, ib * n))
        NextLA.geqrt!(m, n, ib, A1, Tm1, tau1, work1)
        synchronize(A1)

        A2 = ArrayType(copy(A_init))
        Tm2 = ArrayType(zeros(ComplexF64, ib, k))
        tau2 = ArrayType(zeros(ComplexF64, k))
        work2 = ArrayType(zeros(ComplexF64, ib * n))
        NextLA.geqrt!(m, n, ib, A2, Tm2, tau2, work2)
        synchronize(A2)

        @test Array(A1) ≈ Array(A2)
        @test Array(Tm1) ≈ Array(Tm2)
    end
end
