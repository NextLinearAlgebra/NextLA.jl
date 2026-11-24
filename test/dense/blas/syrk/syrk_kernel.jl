# SYRK_KERNEL!, the KernelAbstractions symmetric rank-k update.
#
# Vicki's own test/syrk/syrk.jl is not usable here: it hardcodes CuArray, pulls in
# an include("benchmark.jl") that is not on the branch, and runs to n=1024 with
# printing per case. This one covers the same ground over available_backends().
#
# CPU coverage is the point of the exercise. The kernel as it stood on the
# branch could not run there at all -- I and J were assigned before a
# @synchronize and read after it, and N and M were computed inside the body. A
# CUDA-only green run would have hidden every one of those. See KNOWN_ISSUES.md.

for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "SYRK_KERNEL [$backend_name]" begin
        @testset "$T" for T in (Float32, Float64)
            rtol = test_rtol(T)

            @testset "uplo=$uplo trans=$trans n=$n k=$k" for
                    uplo  in ('L', 'U'),
                    trans in ('N', 'T'),
                    n     in [16, 32, 64, 50],
                    k     in [1, 8, 40]

                A0 = trans == 'N' ? rand(T, n, k) : rand(T, k, n)
                C0 = rand(T, n, n)
                alpha = T(1.5)
                beta  = T(0.5)

                Ad = ArrayType(copy(A0))
                Cd = ArrayType(copy(C0))
                NextLA.SYRK_KERNEL!(uplo, trans, alpha, Ad, beta, Cd)
                synchronize(Cd)

                got = Array(Cd)
                prod = trans == 'N' ? A0 * transpose(A0) : transpose(A0) * A0
                expected = alpha * prod + beta * C0

                keep = uplo == 'L' ? tril : triu
                @test norm(keep(got) - keep(expected)) / max(norm(keep(expected)), eps(T)) < rtol

                # the other triangle must be byte-identical: that is what makes
                # this a SYRK rather than a GEMM
                drop = uplo == 'L' ? triu : tril
                @test drop(got, uplo == 'L' ? 1 : -1) == drop(C0, uplo == 'L' ? 1 : -1)
            end
        end

        @testset "rejects bad arguments" begin
            A = ArrayType(rand(Float32, 8, 4))
            C = ArrayType(rand(Float32, 8, 8))
            @test_throws ArgumentError NextLA.SYRK_KERNEL!('X', 'N', 1f0, A, 0f0, C)
            @test_throws ArgumentError NextLA.SYRK_KERNEL!('L', 'X', 1f0, A, 0f0, C)
            @test_throws DimensionMismatch NextLA.SYRK_KERNEL!('L', 'N', 1f0, A, 0f0,
                                                               ArrayType(rand(Float32, 8, 4)))
            @test_throws DimensionMismatch NextLA.SYRK_KERNEL!('L', 'N', 1f0, A, 0f0,
                                                               ArrayType(rand(Float32, 4, 4)))
        end
    end
end
