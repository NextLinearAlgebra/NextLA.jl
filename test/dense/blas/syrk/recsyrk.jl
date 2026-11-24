# Recursive blocked SYRK over dense matrices.
#
# recsyrk! writes the lower triangle only, and the recursion's off-diagonal
# GEMM fills C21 while the diagonal blocks recurse -- so the comparison is
# against tril(alpha*A*A' + beta*C).
#
# The Float16 path of _syrk_dispatch! routes to gemmEx!, which now has a CPU
# method: the accumulation happens in the compute type and is rounded back to
# Float16 on store, so the tolerance below is the storage's, not the sum's.

@testset "RECSYRK [CPU]" begin
    # syrk is A*transpose(A), not A*adjoint(A), so complex is a real case here
    # and BLAS.syrk! has methods for it.
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        # ComplexF32 is single precision too -- `T === Float32` missed that and
        # held the complex single case to a double-precision tolerance.
        rtol = T <: Union{Float32, ComplexF32} ? 1e-4 : 1e-10

        @testset "dense n=$n k=$k threshold=$th" for
                n  in [32, 64, 128],
                k  in [16, 64],
                th in [16, 64, 256]

            A0 = rand(T, n, k)
            C0 = rand(T, n, n)
            alpha = T(1.5)
            beta  = T(0.5)

            C = copy(C0)
            NextLA.recsyrk!(alpha, A0, beta, C, th)

            expected = alpha * (A0 * transpose(A0)) + beta * C0
            @test norm(tril(C) - tril(expected)) / norm(tril(expected)) < rtol
        end
    end

    # threshold >= n takes the base case directly, without ever recursing
    @testset "base case only" begin
        n = 32
        A0 = rand(Float64, n, 8); C0 = rand(Float64, n, n)
        C = copy(C0)
        NextLA.recsyrk!(1.0, A0, 0.0, C, 4096)
        @test norm(tril(C) - tril(A0 * transpose(A0))) / norm(A0 * transpose(A0)) < 1e-10
    end

    # This used to assert an ArgumentError: gemmEx! had no CPU method, so a
    # Float16 container could not be updated on the host at all. It now
    # computes, and the reference is taken in double precision so that a wrong
    # accumulation order cannot hide inside Float16 rounding.
    @testset "Float16" begin
        n = 32
        A0 = rand(Float16, n, 8)
        C0 = rand(Float16, n, n)
        C = copy(C0)
        NextLA.recsyrk!(Float16(1), A0, Float16(0), C, 4096)
        expected = Float64.(A0) * transpose(Float64.(A0))
        @test norm(tril(Float64.(C)) - tril(expected)) / norm(tril(expected)) < 5e-3
    end
end

# A and C of different types are computed in C's type: A widened exactly into a
# wider C, rounded once into a narrower one, and C itself never rounded. The
# Float32 fallback this replaced rounded a Float64 C to Float32 (single-precision
# accuracy) and threw InexactError on mismatched complex types.
@testset "RECSYRK A and C of different types [$backend_name]" for (backend_name, ArrayType, synchronize) in
        filter(b -> b[1] in ("CPU", "CUDA", "AMDGPU"), available_backends())
    @testset "$TA -> $TC" for (TA, TC, rtol) in ((Float32, Float64, 1e-12),
                                                  (Float16, Float64, 1e-12),
                                                  (Float64, Float32, 1e-5),
                                                  (ComplexF32, ComplexF64, 1e-12),
                                                  (ComplexF64, ComplexF32, 1e-5))
        n, k = 96, 40
        A, C0 = rand(TA, n, k), rand(TC, n, n)
        C = ArrayType(copy(C0))
        NextLA.recsyrk!(TC(2), ArrayType(A), TC(3), C, 32)
        synchronize(C)
        expected = 2 .* (TC.(A) * transpose(TC.(A))) .+ 3 .* C0
        @test norm(tril(Array(C)) - tril(expected)) / norm(tril(expected)) < rtol
    end
end

@testset "RECSYRK uplo [CPU]" begin
    @testset "uplo='U', $T n=$n threshold=$th" for T in (Float32, Float64), n in (32, 50, 64), th in (16, 256)
        A0, C0 = rand(T, n, 12), rand(T, n, n)
        C = copy(C0)
        NextLA.recsyrk!(T(1.5), A0, T(0.5), C, th; uplo='U')
        expected = T(1.5) * (A0 * transpose(A0)) + T(0.5) * C0
        @test norm(triu(C) - triu(expected)) / norm(triu(expected)) < (T === Float32 ? 1e-4 : 1e-10)
        @test tril(C, -1) == tril(C0, -1)
    end
    @testset "uplo='L' leaves the upper triangle" begin
        A0, C0 = rand(40, 8), rand(40, 40)
        C = copy(C0)
        NextLA.recsyrk!(1.0, A0, 1.0, C, 16)
        @test triu(C, 1) == triu(C0, 1)
    end
    @testset "an unknown uplo throws" begin
        @test_throws ArgumentError NextLA.recsyrk!(1.0, rand(8, 2), 1.0, rand(8, 8); uplo='X')
    end
    # The half-precision leaf is a GEMM; it must still leave the other triangle alone.
    @testset "Float16 leaves the other triangle, uplo=$uplo" for uplo in ('L', 'U')
        A0, C0 = rand(Float16, 32, 8), rand(Float16, 32, 32)
        C = copy(C0)
        NextLA.recsyrk!(1, A0, 1, C, 16; uplo=uplo)
        if uplo == 'L'
            @test triu(C, 1) == triu(C0, 1)
        else
            @test tril(C, -1) == tril(C0, -1)
        end
    end
end

# Float16 storage on the GPU: gemmEx! with Float32 accumulation on CUDA and
# AMDGPU, gemm! on oneAPI, which has no gemmEx! (ext/oneapi/syrk.jl).
@testset "RECSYRK Float16 on the GPU [$backend_name]" for (backend_name, ArrayType, synchronize) in
        filter(b -> b[1] in ("CUDA", "AMDGPU", "oneAPI"), available_backends())
    @testset "C $TC" for TC in (Float16, Float32)
        n, k = 64, 16
        A0, C0 = rand(Float16, n, k), rand(TC, n, n)
        C = ArrayType(copy(C0))
        NextLA.recsyrk!(1, ArrayType(A0), 1, C, 16)
        synchronize(C)
        expected = Float64.(A0) * transpose(Float64.(A0)) .+ Float64.(C0)
        @test norm(tril(Float64.(Array(C))) - tril(expected)) / norm(tril(expected)) < 5e-3
    end
end
