using NextLA.MixedPrecision
# potrf_recursive! over a SymmMixedPrec container. CPU only.

function _mxp_spd(T, n)
    R = rand(T, n, n)
    return R * R' + T(n) * I
end

@testset "POTRF_RECURSIVE mixed precision [CPU]" begin
    Random.seed!(2057)

    @testset "n=$n uplo=$uplo" for n in (32, 40, 64), uplo in ('L', 'U')
        T = Float32
        A0 = _mxp_spd(T, n)

        mp = SymmMixedPrec(copy(A0), uplo; precisions = DataType[Float32, Float32])
        @test NextLA.potrf_recursive!(mp) === mp

        F = reconstruct_matrix(TriMixedPrec(mp))
        if uplo == 'L'
            @test istril(F)
            @test norm(F * F' - A0) / norm(A0) < 1e-3
        else
            @test istriu(F)
            @test norm(F' * F - A0) / norm(A0) < 1e-3
        end
    end

    @testset "block_size reaches the leaves" begin
        A0 = _mxp_spd(Float64, 32)
        mp = SymmMixedPrec(copy(A0), 'L'; precisions = DataType[Float64])
        NextLA.potrf_recursive!(mp, 8)
        L = reconstruct_matrix(TriMixedPrec(mp))
        @test norm(L * L' - A0) / norm(A0) < 1e-12
    end

    @testset "Float16 off-diagonal blocks" begin
        n = 32
        A0 = _mxp_spd(Float32, n)
        mp = SymmMixedPrec(copy(A0), 'L'; precisions = DataType[Float16, Float32])
        NextLA.potrf_recursive!(mp)
        L = reconstruct_matrix(TriMixedPrec(mp))
        @test norm(L * L' - A0) / norm(A0) < 1e-2
    end

    # A leaf no larger than block_size is factored by potrf!, whose info turns
    # into PosDefException. A larger leaf goes through the dense potrf_recursive!,
    # which drops that info.
    @testset "a matrix that is not positive definite throws, uplo=$uplo" for uplo in ('L', 'U')
        A0 = -Matrix{Float64}(I, 8, 8)
        mp = SymmMixedPrec(A0, uplo; precisions = DataType[Float32, Float64])
        @test_throws PosDefException NextLA.potrf_recursive!(mp)
    end

    @testset "a leaf larger than block_size, uplo=$uplo" for uplo in ('L', 'U')
        A0 = _mxp_spd(Float64, 32)
        mp = SymmMixedPrec(copy(A0), uplo; precisions = DataType[Float64])
        NextLA.potrf_recursive!(mp, 8)
        F = reconstruct_matrix(TriMixedPrec(mp))
        @test norm((uplo == 'L' ? F * F' : F' * F) - A0) / norm(A0) < 1e-12
    end

    @testset "a leaf stored with a scale factor throws" begin
        mp = SymmMixedPrec(1.0e5 * ones(4, 4) + 1.0e6I, 'L'; precisions = DataType[Float16])
        @test mp.Base_scale != 1
        @test_throws ArgumentError NextLA.potrf_recursive!(mp)
    end

    # reconstruct_matrix round-trips an unfactored container, and is the
    # symmetric counterpart to the FullMixedPrec method.
    @testset "reconstruct_matrix round-trips" begin
        n = 32
        A0 = _mxp_spd(Float32, n)
        mp = SymmMixedPrec(copy(A0), 'L'; precisions = DataType[Float32, Float32])
        @test norm(reconstruct_matrix(mp) - A0) / norm(A0) < 1e-5
    end
end
