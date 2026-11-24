# quantize and dequantize, src/mxp/core/quantize.jl.
# Helpers and error bounds: test/setup/mxp.jl.

@testset "quantize and dequantize [CPU]" begin
    Random.seed!(2047)

    @testset "to $T" for T in (Float16, NextLA.BFloat16, Float32)
        A = mxp_rand(Float64, 32, 32) .* 1000
        q, s = quantize(A, T)
        @test eltype(q) === T && s === 1f0
        back = dequantize(q, s, Float64)
        @test mxp_storage_ok(back, A, fill(T, size(A)))
    end

    @testset "the default type is Float16" begin
        @test eltype(first(quantize(rand(Float32, 4, 4)))) === Float16
    end

    @testset "values past floatmax come back scaled, not saturated" begin
        big = fill(1f5, 8, 8)
        qb, sb = quantize(big)
        @test sb > 1f0
        @test all(isfinite, Float32.(qb))
        @test dequantize(qb, sb, Float32) ≈ big rtol=2mxp_u(Float16)
    end

    @testset "a zero matrix and an empty matrix" begin
        z, sz = quantize(zeros(Float32, 4, 4))
        @test sz == 1f0 && all(iszero, z)
        qe, se = quantize(zeros(Float64, 0, 3), Float16)
        @test size(qe) == (0, 3) && eltype(qe) === Float16
        @test se === 1f0
    end

    @testset "a non-finite matrix throws" begin
        @test_throws ArgumentError quantize([1.0 NaN])
    end

    @testset "dequantize converts before multiplying" begin
        q3 = reshape(Float16[3], 1, 1)
        @test dequantize(q3, 1.1f0, Float64)[1] == 3.0 * Float64(1.1f0)
        @test (@inferred dequantize(q3, 1.1f0, Float64)) isa Matrix{Float64}
    end
end

# On each GPU backend: the result stays on the device.
@testset "quantize and dequantize [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2048)
    n = 13

    @testset "quantize and dequantize stay on the device" begin
        A = mxp_rand(Float64, n, n)
        q, s = quantize(AT(A), Float16)
        @test q isa AT && eltype(q) === Float16
        back = dequantize(q, s, Float64)
        @test back isa AT
        @test mxp_storage_ok(Array(back), A, fill(Float16, n, n))
    end
end
