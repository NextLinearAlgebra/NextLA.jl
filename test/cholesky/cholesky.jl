# Cholesky: the unblocked single-workgroup kernel and the recursive driver.
#
# Written from scratch. The ten Cholesky files on vicki-development are all
# benchmark or plotting scripts with a top-level call at EOF, and the sole
# @testset among them is commented out, so there is nothing to adapt. See
# KNOWN_ISSUES.md.
#
# SPD input is built as R*R' + n*I, matching the branch's own fixtures: well
# conditioned, and positive definite by construction.

@testset "CHOLESKY_LOWER [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "$T" for T in (Float32, Float64)
        rtol = test_rtol(T)

        @testset "n=$n" for n in (4, 8, 16, 32, 64)
            R = rand(T, n, n)
            A0 = R * R' + T(n) * I
            Ad = ArrayType(copy(A0))

            NextLA.cholesky_lower!(Ad)
            synchronize(Ad)
            A = Array(Ad)

            L = LowerTriangular(A)
            @test norm(L * L' - A0) / norm(A0) < rtol
            # this kernel zeroes the upper triangle, unlike potrf!
            @test all(iszero, triu(A, 1))
        end
    end

    @testset "rejects a non-square matrix" begin
        @test_throws DimensionMismatch NextLA.cholesky_lower!(ArrayType(rand(Float64, 4, 5)))
    end
end

@testset "POTRF_RECURSIVE [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "$T" for T in (Float32, Float64)
        rtol = test_rtol(T)

        @testset "n=$n block=$bs" for n in (32, 64, 128), bs in (8, 16, 32)
            R = rand(T, n, n)
            A0 = R * R' + T(n) * I
            Ad = ArrayType(copy(A0))

            NextLA.potrf_recursive!(Ad, bs)
            synchronize(Ad)

            L = LowerTriangular(Array(Ad))
            @test norm(L * L' - A0) / norm(A0) < rtol
        end

        # A block size at or above n must take the potrf! base case directly
        # and agree with it exactly.
        @testset "block_size >= n falls through to potrf!" begin
            n = 32
            R = rand(T, n, n)
            A0 = R * R' + T(n) * I
            a = ArrayType(copy(A0)); b = ArrayType(copy(A0))
            NextLA.potrf_recursive!(a, 64)
            NextLA.potrf!('L', b)
            synchronize(a); synchronize(b)
            @test tril(Array(a)) == tril(Array(b))
        end
    end
end

@testset "POTRF_RECURSIVE uplo and errors [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "uplo='U', $T n=$n block=$bs" for T in (Float32, Float64), n in (32, 50, 64), bs in (8, 32)
        R = rand(T, n, n)
        A0 = R * R' + T(n) * I
        Ad = ArrayType(copy(A0))
        @test NextLA.potrf_recursive!(Ad, bs; uplo='U') === Ad
        synchronize(Ad)
        U = UpperTriangular(Array(Ad))
        @test norm(U' * U - A0) / norm(A0) < test_rtol(T)
    end

    # The leaf whose minor fails is not the first one, so the offset is checked.
    @testset "not positive definite throws PosDefException, uplo=$uplo" for uplo in ('L', 'U')
        A0 = Matrix{Float64}(I, 12, 12)
        A0[10, 10] = -1
        err = try NextLA.potrf_recursive!(ArrayType(A0), 4; uplo=uplo); nothing catch e; e end
        @test err isa PosDefException
        @test err.info == 10
    end

    @testset "rejects a block_size below one" begin
        @test_throws ArgumentError NextLA.potrf_recursive!(ArrayType(Matrix{Float64}(I, 4, 4)), 0)
    end

    @testset "rejects a non-square matrix" begin
        @test_throws DimensionMismatch NextLA.potrf_recursive!(ArrayType(zeros(Float64, 4, 6)), 2)
    end
end

# Half precision. The leaves go through potrf!'s Float16 method and the panel
# solve through the Float16 TRSM base case, which here is handed a transposed
# view of A. The residual is taken in Float64 so that it measures the factor
# rather than Float16 arithmetic in the check. Metal has no potrf! at all.
@testset "POTRF_RECURSIVE Float16 [$backend_name]" for (backend_name, ArrayType, synchronize) in available_backends()
    backend_name in ("CPU", "CUDA", "AMDGPU") || continue
    @testset "n=$n block=$bs" for n in (32, 64, 128), bs in (8, 16, 32)
        R = rand(n, n)
        A0 = Float16.(R * R' + n * I)
        Ad = ArrayType(copy(A0))

        NextLA.potrf_recursive!(Ad, bs)
        synchronize(Ad)

        L = LowerTriangular(Float64.(Array(Ad)))
        @test norm(L * L' - Float64.(A0)) / norm(Float64.(A0)) < 5e-3
    end
end
