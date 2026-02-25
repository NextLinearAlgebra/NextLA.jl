for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "GERC [$backend_name]" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)

                @testset "m=$m, n=$n" for (m, n) in [
                    (0, 0), (1, 1), (2, 3), (5, 4), (10, 8), (50, 40),
                ]
                    α  = T(0.7, 0.3)
                    x  = ArrayType(rand(T, m))
                    y  = ArrayType(rand(T, n))
                    A  = ArrayType(rand(T, m, n))
                    A_init = Array(A)

                    NextLA.gerc!(α, x, y, A)
                    synchronize(A)

                    if m == 0 || n == 0
                        @test Array(A) == A_init
                    else
                        @test Array(A) ≈ A_init .+ α .* Array(x) * Array(y)' rtol=rtol
                    end
                end

                @testset "α=0 leaves A unchanged" begin
                    A = ArrayType(rand(T, 5, 4))
                    A0 = Array(A)
                    NextLA.gerc!(zero(T), ArrayType(rand(T, 5)), ArrayType(rand(T, 4)), A)
                    synchronize(A)
                    @test Array(A) == A0
                end
            end
    end
end

@testset "GERC Error handling" begin
    @test_throws ArgumentError NextLA.gerc!(1.0+0im, ComplexF64[1], ComplexF64[1,2], zeros(ComplexF64, 2, 2))
end
