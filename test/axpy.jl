for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "AXPY [$backend_name]" begin
        @testset "$T" for T in TEST_TYPES
            rtol = test_rtol(T)

            @testset "n=$n, α=$α" for n in [0, 1, 10, 100, 1023],
                                   α in [zero(T), one(T), T(0.7)]
                x = ArrayType(rand(T, n))
                y = ArrayType(rand(T, n))
                y_ref = Array(copy(y))
                n > 0 && BLAS.axpy!(α, Array(x), y_ref)

                NextLA.axpy!(α, x, y)
                synchronize(y)

                if n == 0 || α == zero(T)
                    @test Array(y) == y_ref
                else
                    @test Array(y) ≈ y_ref rtol=rtol
                end
            end
        end
    end
end

@testset "AXPY Edge cases" begin
    @test_nowarn NextLA.axpy!(0.0, Float64[], Float64[])
    x = rand(10); y = rand(10)
    @test_nowarn NextLA.axpy!(0.0, x, copy(y))
end
