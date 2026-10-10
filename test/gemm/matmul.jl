# GEMM_ADD! and GEMM_SUB! had no test of their own. Both were broken on the CPU
# backend: the matmul! kernel assigned I and J once and used them again after a
# @synchronize, and on CPU that splits the body into separate ndrange loops
# where a plain local no longer exists — so `I` resolved to LinearAlgebra.I and
# the call died with `isless(::UniformScaling{Bool}, ::Int64)`. Anything that
# recursed through unified_rectrxm! inherited it.
#
# Note the argument orders differ:
#   GEMM_ADD!(A, B, C)  ->  C .+= A * B
#   GEMM_SUB!(A, B, C)  ->  A .-= B * C

for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "MATMUL [$backend_name]" begin
        @testset "$T" for T in (Float32, Float64)
            rtol = test_rtol(T)

            @testset "GEMM_ADD! n=$n, r=$r, m=$m" for (n, r, m) in [
                (16, 16, 16), (32, 32, 32), (64, 32, 16), (33, 17, 9),
            ]
                A = rand(T, n, r)
                B = rand(T, r, m)
                C = rand(T, n, m)
                expected = C + A * B

                Ad, Bd, Cd = ArrayType(copy(A)), ArrayType(copy(B)), ArrayType(copy(C))
                NextLA.GEMM_ADD!(Ad, Bd, Cd)
                synchronize(Cd)
                @test Array(Cd) ≈ expected rtol=rtol
            end

            @testset "GEMM_SUB! n=$n, r=$r, m=$m" for (n, r, m) in [
                (16, 16, 16), (32, 32, 32), (64, 32, 16),
            ]
                B = rand(T, r, m)
                C = rand(T, m, n)
                A = rand(T, r, n)
                expected = A - B * C

                Ad, Bd, Cd = ArrayType(copy(A)), ArrayType(copy(B)), ArrayType(copy(C))
                NextLA.GEMM_SUB!(Ad, Bd, Cd)
                synchronize(Ad)
                @test Array(Ad) ≈ expected rtol=rtol
            end
        end
    end
end

# The wrappers also expose the kernel's alpha and transpose flags, accumulate a
# Float16 product in Float32, and check shapes. src/trsm/rectrxm.jl's mixed-precision
# recursion needs all three.
for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "MATMUL scaled/transposed [$backend_name]" begin
        @testset "$T" for T in (Float32, Float64)
            rtol = test_rtol(T)
            n, r, m = 32, 24, 16

            @testset "GEMM_ADD! scale=$sc" for sc in (1, 2.5, -0.5)
                A, B, C = rand(T, n, r), rand(T, r, m), rand(T, n, m)
                expected = C + T(sc) * A * B
                Ad, Bd, Cd = ArrayType(copy(A)), ArrayType(copy(B)), ArrayType(copy(C))
                NextLA.GEMM_ADD!(Ad, Bd, Cd, sc)
                synchronize(Cd)
                @test Array(Cd) ≈ expected rtol=rtol
            end

            @testset "GEMM_SUB! scale=$sc" for sc in (1, 0.5)
                A, B, C = rand(T, n, m), rand(T, n, r), rand(T, r, m)
                expected = A - T(sc) * B * C
                Ad, Bd, Cd = ArrayType(copy(A)), ArrayType(copy(B)), ArrayType(copy(C))
                NextLA.GEMM_SUB!(Ad, Bd, Cd, sc)
                synchronize(Ad)
                @test Array(Ad) ≈ expected rtol=rtol
            end

            @testset "transposed operands" begin
                A, B, C = rand(T, n, r), rand(T, r, m), rand(T, n, m)
                expected = C + A * B
                # pass Aᵀ wrapped, so the wrapper must route it through transA
                Atd = ArrayType(permutedims(copy(A)))
                Bd, Cd = ArrayType(copy(B)), ArrayType(copy(C))
                NextLA.GEMM_ADD!(transpose(Atd), Bd, Cd)
                synchronize(Cd)
                @test Array(Cd) ≈ expected rtol=rtol
            end
        end

        # Each term is 200 * 200 = 40000 and the signs alternate, so the sum is 0
        # but a running Float16 sum would pass floatmax(Float16) = 65504 on the way.
        @testset "Float16 destination accumulates in Float32" begin
            A = ArrayType(fill(Float16(200), 32, 32))
            B = ArrayType(Float16[isodd(k) ? 200 : -200 for k in 1:32, j in 1:32])
            C = ArrayType(zeros(Float16, 32, 32))
            NextLA.GEMM_ADD!(A, B, C)
            synchronize(C)
            @test all(iszero, Array(C))
        end

        # 32 * 200 * 200 is beyond floatmax(Float16): it is Inf, not clamped.
        @testset "a Float16 result past floatmax is Inf" begin
            A = ArrayType(fill(Float16(200), 32, 32))
            B = ArrayType(fill(Float16(200), 32, 32))
            C = ArrayType(zeros(Float16, 32, 32))
            NextLA.GEMM_ADD!(A, B, C)
            synchronize(C)
            @test all(isinf, Array(C))
        end

        @testset "adjoint operands" begin
            T = ComplexF64
            A, B, C = rand(T, 8, 6), rand(T, 8, 5), rand(T, 6, 5)
            Ad, Bd, Cd = ArrayType(copy(A)), ArrayType(copy(B)), ArrayType(copy(C))
            NextLA.GEMM_ADD!(adjoint(Ad), Bd, Cd)
            synchronize(Cd)
            @test Array(Cd) ≈ C + A' * B rtol=test_rtol(Float64)
        end

        @testset "shapes that do not fit throw" begin
            @test_throws DimensionMismatch NextLA.GEMM_ADD!(ArrayType(rand(Float32, 8, 3)),
                                                            ArrayType(rand(Float32, 5, 8)),
                                                            ArrayType(zeros(Float32, 8, 8)))
            @test_throws DimensionMismatch NextLA.GEMM_ADD!(ArrayType(rand(Float32, 8, 5)),
                                                            ArrayType(rand(Float32, 5, 8)),
                                                            ArrayType(zeros(Float32, 4, 4)))
            @test_throws DimensionMismatch NextLA.GEMM_SUB!(ArrayType(zeros(Float32, 4, 4)),
                                                            ArrayType(rand(Float32, 8, 5)),
                                                            ArrayType(rand(Float32, 5, 8)))
        end
    end
end
