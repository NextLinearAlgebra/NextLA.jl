@testset "PAMM" begin
    @testset "W op — left, column, forward" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, k, l = 40, 30, 20, 10

            A1 = rand(T, k, n)
            A2 = rand(T, m, n)
            V  = rand(T, m, k)
            W  = zeros(T, k, n)

            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)

            @test all(isfinite.(W))
            @test !isapprox(W, zeros(T, k, n), rtol=rtol)
        end
    end

    @testset "A op — left, column, forward" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, k, l = 40, 30, 20, 10

            A1 = rand(T, k, n)
            A2 = rand(T, m, n)
            V  = rand(T, m, k)
            W  = rand(T, k, n)
            A2_orig = copy(A2)

            NextLA.pamm!('A', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)

            @test all(isfinite.(A2))
            @test !isapprox(A2, A2_orig, rtol=rtol)
        end
    end

    @testset "W op — right, column, forward" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            m, n, k, l = 30, 40, 20, 10

            A1 = rand(T, m, k)
            A2 = rand(T, m, n)
            V  = rand(T, n, k)
            W  = rand(T, m, k)

            NextLA.pamm!('W', 'R', 'C', 'F', m, n, k, l, A1, A2, V, W)

            @test all(isfinite.(W))
        end
    end

    @testset "W op — left, column, backward" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            m, n, k, l = 40, 30, 20, 10

            A1 = rand(T, k, n)
            A2 = rand(T, m, n)
            V  = rand(T, m, k)
            W  = rand(T, k, n)

            NextLA.pamm!('W', 'L', 'C', 'B', m, n, k, l, A1, A2, V, W)

            @test all(isfinite.(W))
        end
    end

    @testset "W op — left, row, forward" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            m, n, k, l = 40, 30, 20, 10

            A1 = rand(T, k, n)
            A2 = rand(T, m, n)
            V  = rand(T, k, m)
            W  = rand(T, k, n)

            NextLA.pamm!('W', 'L', 'R', 'F', m, n, k, l, A1, A2, V, W)

            @test all(isfinite.(W))
        end
    end

    @testset "Quick return for zero dims" begin
        T = ComplexF64
        A1 = rand(T, 5, 5)
        A2 = rand(T, 10, 5)
        V  = rand(T, 10, 5)
        W  = rand(T, 5, 5)
        W_orig = copy(W)

        NextLA.pamm!('W', 'L', 'C', 'F', 0, 5, 5, 3, A1, A2, V, W)
        @test W == W_orig

        NextLA.pamm!('W', 'L', 'C', 'F', 10, 0, 5, 3, A1, A2, V, W)
        @test W == W_orig

        NextLA.pamm!('W', 'L', 'C', 'F', 10, 5, 0, 3, A1, A2, V, W)
        @test W == W_orig
    end

    @testset "Error handling" begin
        T = ComplexF64
        A1 = zeros(T, 5, 5)
        A2 = zeros(T, 10, 5)
        V  = zeros(T, 10, 5)
        W  = zeros(T, 5, 5)

        @test_throws ArgumentError NextLA.pamm!('X', 'L', 'C', 'F', 10, 5, 5, 3, A1, A2, V, W)
        @test_throws ArgumentError NextLA.pamm!('W', 'X', 'C', 'F', 10, 5, 5, 3, A1, A2, V, W)
        @test_throws ArgumentError NextLA.pamm!('W', 'L', 'X', 'F', 10, 5, 5, 3, A1, A2, V, W)
        @test_throws ArgumentError NextLA.pamm!('W', 'L', 'C', 'X', 10, 5, 5, 3, A1, A2, V, W)
        @test_throws ArgumentError NextLA.pamm!('W', 'L', 'C', 'F', -1, 5, 5, 3, A1, A2, V, W)
        @test_throws ArgumentError NextLA.pamm!('W', 'L', 'C', 'F', 10, -1, 5, 3, A1, A2, V, W)
    end

    @testset "W then A consistency" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, k, l = 40, 30, 20, 10

            A1 = rand(T, k, n)
            A2 = rand(T, m, n)
            V  = rand(T, m, k)
            W  = zeros(T, k, n)

            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)
            @test all(isfinite.(W))

            A2_before = copy(A2)
            NextLA.pamm!('A', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)
            @test all(isfinite.(A2))
            @test !isapprox(A2, A2_before, rtol=rtol)
        end
    end

    @testset "Different sizes" begin
        T = ComplexF64
        for (m, n, k, l) in [(10, 8, 5, 3), (25, 20, 12, 8), (50, 40, 25, 15)]
            A1 = rand(T, k, n)
            A2 = rand(T, m, n)
            V  = rand(T, m, k)
            W  = zeros(T, k, n)

            NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W)
            @test all(isfinite.(W))
            @test size(W) == (k, n)
        end
    end

    @testset "Deterministic" begin
        Random.seed!(42)
        T = ComplexF64
        m, n, k, l = 20, 15, 10, 5
        A1 = rand(T, k, n); A2 = rand(T, m, n)
        V  = rand(T, m, k); W1 = zeros(T, k, n)
        NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W1)

        W2 = zeros(T, k, n)
        NextLA.pamm!('W', 'L', 'C', 'F', m, n, k, l, A1, A2, V, W2)
        @test W1 == W2
    end
end
