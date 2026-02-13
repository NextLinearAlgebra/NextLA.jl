@testset "GERC" begin
    # gerc! performs A := A + α·x·conj(y)ᵀ  (complex rank‑1 update)
    @testset "$T" for T in (ComplexF32, ComplexF64)
        rtol = test_rtol(T)

        @testset "m=$m, n=$n" for (m, n) in [
            (0, 0), (1, 1), (2, 3), (5, 4), (10, 8), (50, 40),
        ]
            α  = T(0.7, 0.3)
            x  = rand(T, m)
            y  = rand(T, n)
            A  = rand(T, m, n)

            A_test = copy(A)

            NextLA.gerc!(α, x, y, A_test)

            if m == 0 || n == 0
                @test A_test == A
            else
                # gerc! computes A := A + α·x·yᴴ  (y is conjugated)
                @test A_test ≈ A .+ α .* x * y' rtol=rtol
            end
        end

        @testset "α=0 leaves A unchanged" begin
            A = rand(T, 5, 4);  A0 = copy(A)
            NextLA.gerc!(zero(T), rand(T, 5), rand(T, 4), A)
            @test A == A0
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError NextLA.gerc!(1.0+0im, ComplexF64[1], ComplexF64[1,2], zeros(ComplexF64, 2, 2))
    end
end
