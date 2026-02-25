for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "UNMQR [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
            rtol = test_rtol(T)
            m, n, k, ib = 64, 48, 32, 8

            # Build a valid QR factorization first
            A_qr = ArrayType(copy(rand(T, m, k)))
            Tm = ArrayType(zeros(T, ib, k))
            tau = ArrayType(zeros(T, k))
            work_qr = ArrayType(zeros(T, ib * k))
            NextLA.geqrt!(m, k, ib, A_qr, Tm, tau, work_qr)
            synchronize(A_qr)

            @testset "Left, No-Transpose" begin
                C = ArrayType(copy(rand(T, m, n)))
                C_orig = copy(C)

                work = ArrayType(zeros(T, n, ib))
                NextLA.unmqr!('L', 'N', m, n, k, ib, A_qr, m, Tm, C, work)
                synchronize(C)

                @test norm(C) ≈ norm(C_orig) rtol=rtol

                C_h = copy(C_orig)
                NextLA.unmqr!('L', 'N', A_qr, Tm, C_h)
                synchronize(C_h)
                @test C_h ≈ C rtol=rtol
            end

            @testset "Left, Conjugate-Transpose" begin
                C = ArrayType(copy(rand(T, m, n)))
                C_orig = copy(C)

                work = ArrayType(zeros(T, n, ib))
                NextLA.unmqr!('L', 'C', m, n, k, ib, A_qr, m, Tm, C, work)
                synchronize(C)

                @test norm(C) ≈ norm(C_orig) rtol=rtol

                C_h = copy(C_orig)
                NextLA.unmqr!('L', 'C', A_qr, Tm, C_h)
                synchronize(C_h)
                @test C_h ≈ C rtol=rtol
            end

            @testset "Right, No-Transpose" begin
                C = ArrayType(copy(rand(T, n, m)))
                C_orig = copy(C)

                work = ArrayType(zeros(T, n, ib))
                NextLA.unmqr!('R', 'N', n, m, k, ib, A_qr, m, Tm, C, work)
                synchronize(C)

                @test norm(C) ≈ norm(C_orig) rtol=rtol
            end

            @testset "Right, Conjugate-Transpose" begin
                C = ArrayType(copy(rand(T, n, m)))
                C_orig = copy(C)

                work = ArrayType(zeros(T, n, ib))
                NextLA.unmqr!('R', 'C', n, m, k, ib, A_qr, m, Tm, C, work)
                synchronize(C)

                @test norm(C) ≈ norm(C_orig) rtol=rtol
            end

            @testset "Q * Qᴴ = I (orthogonality)" begin
                C = ArrayType(collect(Matrix{T}(I, m, m)))
                work = ArrayType(zeros(T, m, ib))

                NextLA.unmqr!('L', 'N', m, m, k, ib, A_qr, m, Tm, C, work)
                NextLA.unmqr!('L', 'C', m, m, k, ib, A_qr, m, Tm, C, work)
                synchronize(C)

                @test Array(C) ≈ Matrix{T}(I, m, m) rtol=rtol
            end
        end

        @testset "Rectangular tile: k clamped to nq" begin
            for T in (Float64, ComplexF64)
                m_tile, n_tile = 16, 32  # wide tile: only 16 reflectors possible
                ib = 4
                k_actual = m_tile
                A_qr = ArrayType(copy(rand(T, m_tile, k_actual)))
                Tm = ArrayType(zeros(T, ib, k_actual))
                tau = ArrayType(zeros(T, k_actual))
                work_qr = ArrayType(zeros(T, ib * k_actual))
                NextLA.geqrt!(m_tile, k_actual, ib, A_qr, Tm, tau, work_qr)

                C = ArrayType(copy(rand(T, m_tile, 24)))
                C_orig = copy(C)
                work = ArrayType(zeros(T, 24, ib))
                @test_nowarn NextLA.unmqr!('L', 'N', m_tile, 24, n_tile, ib,
                                           A_qr, m_tile, Tm, C, work)
                synchronize(C)
                @test norm(C) ≈ norm(C_orig) rtol=test_rtol(T)

                Tm_wide = ArrayType(zeros(T, ib, n_tile))
                Tm_wide[:, 1:k_actual] .= Tm
                C2 = copy(C_orig)
                @test_nowarn NextLA.unmqr!('L', 'N', A_qr, Tm_wide, C2)
                synchronize(C2)
                @test norm(C2) ≈ norm(C_orig) rtol=test_rtol(T)
            end
        end
    end
end

@testset "UNMQR Error handling" begin
    m, n, k, ib = 32, 24, 16, 4
    A = zeros(ComplexF64, m, k)
    Tm = zeros(ComplexF64, ib, k)
    C = zeros(ComplexF64, m, n)
    work = zeros(ComplexF64, n, ib)

    @test_throws ArgumentError NextLA.unmqr!('X', 'N', m, n, k, ib, A, m, Tm, C, work)
    @test_throws ArgumentError NextLA.unmqr!('L', 'X', m, n, k, ib, A, m, Tm, C, work)
    @test_throws ArgumentError NextLA.unmqr!('L', 'N', -1, n, k, ib, A, m, Tm, C, work)
    C_test = zeros(ComplexF64, m, n)
    @test_nowarn NextLA.unmqr!('L', 'N', m, n, m+1, ib, A, m, Tm, C_test, work)
end

@testset "UNMQR k=0 is a no-op" begin
    C = rand(ComplexF64, 32, 24)
    C_orig = copy(C)
    A = zeros(ComplexF64, 32, 1)
    Tm = zeros(ComplexF64, 4, 1)
    work = zeros(ComplexF64, 24, 4)

    NextLA.unmqr!('L', 'N', 32, 24, 0, 4, A, 32, Tm, C, work)
    @test C ≈ C_orig
end
