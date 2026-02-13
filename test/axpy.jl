@testset "AXPY" begin
    @testset "$T" for T in TEST_TYPES
        rtol = test_rtol(T)

        @testset "n=$n, α=$α" for n in [0, 1, 10, 100, 1023],
                                   α in [zero(T), one(T), T(0.7)]
            x = rand(T, n)
            y = rand(T, n)
            y_ref = copy(y)
            y_test = copy(y)

            # Reference: BLAS
            n > 0 && BLAS.axpy!(α, x, y_ref)

            # NextLA
            NextLA.axpy!(α, x, y_test)

            if n == 0 || α == zero(T)
                @test y_test == y   # unchanged
            else
                @test y_test ≈ y_ref rtol=rtol
                # Direct formula check
                @test y_test ≈ y .+ α .* x rtol=rtol
            end
        end
    end

    @testset "Edge cases" begin
        @test_nowarn NextLA.axpy!(0.0, Float64[], Float64[])
        x = rand(10); y = rand(10)
        @test_nowarn NextLA.axpy!(0.0, x, copy(y))
    end
end
