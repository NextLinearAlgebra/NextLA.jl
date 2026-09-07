using NextLA.MixedPrecision

const MXP_TYPES = (Float32, Float64)

@testset "MXP [CPU]" begin
    Random.seed!(2047)


    @testset "FullMixedPrec" begin
        @testset "$T" for T in MXP_TYPES
            @testset "n=$n" for n in (4, 5, 8, 13, 16)
                A = rand(T, n, n)

                @testset "single precision level round-trips exactly" begin
                    M = FullMixedPrec(A; precisions=DataType[T])
                    @test size(M) == (n, n)
                    @test reconstruct_matrix(M) == A
                end

                @testset "two levels round-trip within the coarser precision" begin
                    M = FullMixedPrec(A; precisions=DataType[Float32, T])
                    @test size(M) == (n, n)
                    @test eltype(reconstruct_matrix(M)) === T
                    @test reconstruct_matrix(M) ≈ A rtol=test_rtol(Float32)
                end

                @testset "indexing agrees with the dense original" begin
                    M = FullMixedPrec(A; precisions=DataType[Float32, T])
                    @test all(isapprox(M[i, j], A[i, j]; rtol=test_rtol(Float32))
                              for i in 1:n, j in 1:n)
                    @test M[1, 1] isa T
                end
            end
        end

        @testset "three levels with Float16 off-diagonal blocks" begin
            A = rand(Float64, 16, 16)
            M = FullMixedPrec(A; precisions=DataType[Float16, Float32, Float64])
            @test eltype(M.A21) === Float16
            @test eltype(M.A11.A21) === Float32
            @test reconstruct_matrix(M) ≈ A rtol=1e-3
        end

        @testset "scale_type sets the type of every scale" begin
            M = FullMixedPrec(rand(8, 8); precisions=DataType[Float32, Float64], scale_type=Float64)
            @test M.A21_scale isa Float64
            @test M.A11.Base_scale isa Float64
        end

        @testset "arguments are checked" begin
            @test_throws DimensionMismatch FullMixedPrec(rand(Float64, 4, 8);
                                                      precisions=DataType[Float64])
            @test_throws ArgumentError FullMixedPrec(zeros(0, 0); precisions=DataType[Float64])
            @test_throws ArgumentError FullMixedPrec(rand(4, 4); precisions=DataType[])
            @test_throws ArgumentError FullMixedPrec(rand(4, 4); precisions=DataType[Int])
            @test_throws ArgumentError FullMixedPrec(rand(4, 4); precisions=DataType[Float64],
                                                     scale_type=BigFloat)
            @test_throws ArgumentError FullMixedPrec([1.0 NaN; 0.0 1.0]; precisions=DataType[Float64])
            @test_throws ArgumentError FullMixedPrec([1.0 Inf; 0.0 1.0]; precisions=DataType[Float64])
        end

        @testset "a block too large even for the largest scale throws" begin
            @test_throws ArgumentError FullMixedPrec(fill(1.0e300, 4, 4);
                                                     precisions=DataType[Float16], scale_type=Float16)
        end

        @testset "out-of-bounds indexing throws" begin
            M = FullMixedPrec(rand(8, 8); precisions=DataType[Float32, Float64])
            @test_throws BoundsError M[0, 1]
            @test_throws BoundsError M[1, 9]
        end
    end

    @testset "transpose is lazy" begin
        A = rand(Float64, 8, 8)
        M = FullMixedPrec(A; precisions=DataType[Float32, Float64])
        Mᵀ = transpose(M)
        @test parent(Mᵀ) === M
        @test transpose(Mᵀ) === M
        @test size(Mᵀ) == reverse(size(M))
        @test all(Mᵀ[i, j] == M[j, i] for i in 1:8, j in 1:8)
        @test_throws BoundsError Mᵀ[9, 1]
    end

    @testset "SymmMixedPrec" begin
        @testset "$T, uplo=$uplo, n=$n" for T in MXP_TYPES, uplo in ('L', 'U'), n in (7, 8)
            B = rand(T, n, n)
            A = B + transpose(B)

            M = SymmMixedPrec(A, uplo; precisions=DataType[Float32, T])
            @test size(M) == (n, n)

            @testset "reads back symmetric, matching the stored triangle" begin
                @test all(isapprox(M[i, j], A[i, j]; rtol=test_rtol(Float32))
                          for i in 1:n, j in 1:n)
                @test all(M[i, j] == M[j, i] for i in 1:n, j in 1:n)
            end

            @testset "reconstruct_matrix fills both triangles" begin
                R = reconstruct_matrix(M)
                @test R ≈ A rtol=test_rtol(Float32)
                @test R == transpose(R)
            end

            @testset "transpose is the identity" begin
                @test transpose(M) === M
            end
        end

        @testset "the other triangle is not read, uplo=$uplo" for uplo in ('L', 'U')
            B = rand(8, 8)
            A = B + transpose(B)
            G = copy(A)
            uplo == 'L' ? (G[1, 8] = NaN) : (G[8, 1] = NaN)
            M = SymmMixedPrec(G, uplo; precisions=DataType[Float32, Float64])
            @test reconstruct_matrix(M) ≈ A rtol=test_rtol(Float32)
        end

        @testset "a scaled off-diagonal block reads back unscaled, uplo=$uplo" for uplo in ('L', 'U')
            A = ones(4, 4)
            A[3:4, 1:2] .= 1.0e5
            A[1:2, 3:4] .= 1.0e5
            M = SymmMixedPrec(A, uplo; precisions=DataType[Float16, Float64])
            @test M.OffDiag_scale != 1
            @test all(isapprox.(reconstruct_matrix(M), A; rtol=1e-2))
            @test isapprox(M[3, 1], 1.0e5; rtol=1e-2)
            @test isapprox(M[1, 3], 1.0e5; rtol=1e-2)
        end

        @testset "arguments are checked" begin
            @test_throws ArgumentError SymmMixedPrec(rand(4, 4), 'X'; precisions=DataType[Float64])
            @test_throws DimensionMismatch SymmMixedPrec(rand(4, 8), 'L'; precisions=DataType[Float64])
            @test_throws BoundsError SymmMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float64])[5, 1]
        end
    end

    @testset "TriMixedPrec" begin
        @testset "$T, uplo=$uplo, n=$n" for T in MXP_TYPES, uplo in ('L', 'U'), n in (7, 8)
            A = uplo == 'L' ? tril(rand(T, n, n)) : triu(rand(T, n, n))

            M = TriMixedPrec(A, uplo; precisions=DataType[Float32, T])
            @test size(M) == (n, n)

            @testset "stored triangle matches, other triangle reads zero" begin
                @test all(isapprox(M[i, j], A[i, j]; rtol=test_rtol(Float32))
                          for i in 1:n, j in 1:n)
                @test reconstruct_matrix(M) ≈ A rtol=test_rtol(Float32)
            end
        end

        @testset "only the uplo triangle of a full input is kept, uplo=$uplo" for uplo in ('L', 'U')
            A = rand(8, 8)
            M = TriMixedPrec(A, uplo; precisions=DataType[Float32, Float64])
            @test reconstruct_matrix(M) ≈ (uplo == 'L' ? tril(A) : triu(A)) rtol=test_rtol(Float32)
        end

        @testset "converts from SymmMixedPrec preserving the triangle, uplo=$uplo" for uplo in ('L', 'U')
            n = 8
            B = rand(Float64, n, n)
            A = B + transpose(B)
            S = SymmMixedPrec(A, uplo; precisions=DataType[Float32, Float64])
            M = TriMixedPrec(S)

            @test size(M) == size(S)
            @test M.uplo == uplo
            @test all(M[i, j] == S[i, j] for i in 1:n, j in 1:n if (uplo == 'L' ? i >= j : i <= j))
            @test all(iszero(M[i, j]) for i in 1:n, j in 1:n if (uplo == 'L' ? i < j : i > j))
        end

        @testset "a scaled off-diagonal block reads back unscaled" begin
            A = tril(ones(4, 4))
            A[3:4, 1:2] .= 1.0e5
            M = TriMixedPrec(A, 'L'; precisions=DataType[Float16, Float64])
            @test M.OffDiag_scale != 1
            @test all(isapprox.(reconstruct_matrix(M), A; rtol=1e-2))
        end

        @testset "arguments are checked" begin
            @test_throws ArgumentError TriMixedPrec(rand(4, 4), 'X'; precisions=DataType[Float64])
            @test_throws BoundsError TriMixedPrec(rand(4, 4), 'L'; precisions=DataType[Float64])[1, 0]
        end
    end

    @testset "TiledTriMixedPrec" begin
        @testset "uplo=$uplo" for uplo in ('L', 'U')
            n = 8
            A = uplo == 'L' ? tril(rand(Float64, n, n)) : triu(rand(Float64, n, n))

            M = TiledTriMixedPrec(A, uplo; precisions=DataType[Float32, Float64],
                                  threshold=4)

            @test size(M) == (n, n)
            @test size(M, 1) == n
            @test all(isapprox(M[i, j], A[i, j]; rtol=test_rtol(Float32))
                      for i in 1:n, j in 1:n)
            @test M[1, 1] isa Float64
            @test reconstruct_matrix(M) ≈ A rtol=test_rtol(Float32)
            @test all(transpose(M)[i, j] == M[j, i] for i in 1:n, j in 1:n)
        end

        @testset "multi-level tiling keeps the whole off-diagonal block" begin
            n = 8
            A = tril(rand(Float64, n, n))
            M = TiledTriMixedPrec(A, 'L'; precisions=DataType[Float32, Float64],
                                  threshold=2)

            @test isapprox(M[7, 1], A[7, 1]; rtol=test_rtol(Float32))
            @test reconstruct_matrix(M) ≈ A rtol=test_rtol(Float32)
        end

        @testset "uneven tiles, n=$n, threshold=$t" for (n, t) in ((7, 3), (13, 4), (5, 1))
            A = tril(rand(Float64, n, n))
            M = TiledTriMixedPrec(A, 'L'; precisions=DataType[Float32, Float64], threshold=t)
            @test reconstruct_matrix(M) ≈ A rtol=test_rtol(Float32)
            @test sum(size.(M.diag, 1)) == n
        end

        @testset "a scaled tile reads back unscaled" begin
            A = tril(ones(8, 8))
            A[5:8, 1:4] .= 1.0e5
            M = TiledTriMixedPrec(A, 'L'; precisions=DataType[Float16, Float64], threshold=4)
            @test M.off_scales[2, 1] != 1
            @test all(isapprox.(reconstruct_matrix(M), A; rtol=1e-2))
        end

        @testset "arguments are checked" begin
            A = tril(rand(8, 8))
            @test_throws ArgumentError TiledTriMixedPrec(A, 'X'; precisions=DataType[Float64], threshold=2)
            @test_throws ArgumentError TiledTriMixedPrec(A, 'L'; precisions=DataType[Float64], threshold=0)
            @test_throws ArgumentError TiledTriMixedPrec(A, 'L';
                                                        precisions=DataType[Float16, Float32, Float64],
                                                        threshold=2)
        end

        @testset "out-of-bounds indexing throws" begin
            A = tril(rand(Float64, 8, 8))
            M = TiledTriMixedPrec(A, 'L'; precisions=DataType[Float32, Float64],
                                  threshold=2)
            @test_throws BoundsError M[0, 1]
            @test_throws BoundsError M[9, 1]
        end
    end

    @testset "adaptive_precisions" begin
        n = 16
        A = tril(rand(Float64, n, n)) + n * I

        @testset "returns one precision per level, then the leaf precision" begin
            u = adaptive_precisions(A)
            @test u isa Vector{DataType}
            @test all(T -> T in (Float32, Float64), u)
            @test u[end] === Float64
            @test size(FullMixedPrec(A; precisions=u)) == (n, n)
        end

        @testset "n_min controls the number of levels" begin
            @test length(adaptive_precisions(A, DataType[Float32, Float64], 2)) >
                  length(adaptive_precisions(A, DataType[Float32, Float64], 8))
        end

        @testset "a matrix too small to split gets only the leaf precision" begin
            @test adaptive_precisions(A[1:3, 1:3]) == DataType[Float64]
        end

        @testset "U may be given in any order" begin
            @test adaptive_precisions(A, DataType[Float64, Float32]) == adaptive_precisions(A)
        end

        @testset "arguments are checked" begin
            @test_throws DimensionMismatch adaptive_precisions(rand(Float64, 4, 8))
            @test_throws ArgumentError adaptive_precisions(zeros(0, 0))
            @test_throws ArgumentError adaptive_precisions(A, DataType[])
            @test_throws ArgumentError adaptive_precisions(A, DataType[Int])
            @test_throws ArgumentError adaptive_precisions(A, DataType[Float32, Float64], 0)
            @test_throws ArgumentError adaptive_precisions(A, DataType[Float32, Float64], 4, 0.0)
            @test_throws ArgumentError adaptive_precisions(A; uplo='X')
            @test_throws ArgumentError adaptive_precisions(zeros(n, n))
            @test_throws ArgumentError adaptive_precisions(fill(NaN, n, n))
        end

        @testset "uplo='U' reads the upper triangle" begin
            @test adaptive_precisions(copy(transpose(A)); uplo='U') == adaptive_precisions(A)
        end

        @testset "uplo='F' measures both off-diagonal blocks" begin
            F = [i == j ? 1.0 : i > j ? 1.0e-12 : 1.0 for i in 1:n, j in 1:n]
            @test adaptive_precisions(F; uplo='L')[1] === Float32
            @test adaptive_precisions(F; uplo='F')[1] === Float64
            @test size(FullMixedPrec(F; precisions=adaptive_precisions(F; uplo='F'))) == (n, n)
        end

        @testset "the other triangle is not read" begin
            G = copy(A)
            G[1, n] = NaN
            @test adaptive_precisions(G) == adaptive_precisions(A)
        end

        @testset "adaptive_precision_LT is the same function" begin
            @test adaptive_precision_LT === adaptive_precisions
        end
    end

    @testset "values beyond floatmax(T_Base) cannot be read back as T_Base" begin
        A = fill(1.0e5, 4, 4)
        M = FullMixedPrec(A; precisions=DataType[Float16])
        @test M.Base_scale != 1
        @test_broken all(isapprox.(Float64.(reconstruct_matrix(M)), A; rtol=1e-2))
        @test_broken isapprox(Float64(M[1, 1]), A[1, 1]; rtol=1e-2)
    end

    @testset "a scaled off-diagonal block reads back unscaled" begin
        A = fill(1.0, 4, 4)
        A[3:4, 1:2] .= 1.0e5
        M = FullMixedPrec(A; precisions=DataType[Float16, Float64])
        @test M.A21_scale != 1
        @test all(isapprox.(reconstruct_matrix(M), A; rtol=1e-2))
        @test isapprox(M[3, 1], 1.0e5; rtol=1e-2)
    end
end
