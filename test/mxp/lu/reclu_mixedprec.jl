using NextLA.MixedPrecision
# lu_recursive! over a FullMixedPrec matrix, and the unified_rectrxm! entry
# points it reaches the container through. CPU only.

_mxp_lu_err(R, A0) = norm(UnitLowerTriangular(R) * UpperTriangular(R) - A0) / norm(A0)

@testset "RECLU mixed precision [CPU]" begin
    Random.seed!(2053)

    @testset "lu_recursive! n=$n" for n in [32, 50, 64]
        T = Float32
        A0 = rand(T, n, n) + T(2n) * I
        mp = FullMixedPrec(copy(A0); precisions = DataType[Float32, Float32])

        @test lu_recursive!(mp, 16) === mp
        @test _mxp_lu_err(reconstruct_matrix(mp), A0) < 1e-3
    end

    @testset "an uneven split with a split trailing block throws" begin
        # n = 50 splits 32 + 18: the trailing update multiplies an 18×32 block by a 32×18
        # one into A22, which recgemm! accepts only while A22 is a leaf.
        A0 = rand(Float64, 50, 50) + 100I
        mp = FullMixedPrec(copy(A0); precisions = DataType[Float32, Float32, Float64])
        @test_throws DimensionMismatch lu_recursive!(mp, 8)
    end

    @testset "lu_recursive_mixed! is the same function" begin
        @test lu_recursive_mixed! === lu_recursive!
        A0 = rand(Float64, 16, 16) + 32I
        mp = FullMixedPrec(copy(A0); precisions = DataType[Float32, Float64])
        lu_recursive_mixed!(mp, 4)
        @test _mxp_lu_err(reconstruct_matrix(mp), A0) < 1e-5
    end

    @testset "the dense method still takes a dense matrix" begin
        A0 = rand(Float64, 16, 16) + 32I
        A = copy(A0)
        lu_recursive!(A, 4)
        @test _mxp_lu_err(A, A0) < 1e-12
    end

    @testset "Float16 off-diagonal blocks" begin
        n = 32
        A0 = rand(Float32, n, n) + Float32(2n) * I
        mp = FullMixedPrec(copy(A0); precisions = DataType[Float16, Float32])
        lu_recursive!(mp, 8)
        @test _mxp_lu_err(reconstruct_matrix(mp), A0) < 1e-2
    end

    @testset "parallel=false gives the same factors as parallel=true" begin
        A0 = rand(Float64, 64, 64) + 128I
        a = FullMixedPrec(copy(A0); precisions = DataType[Float16, Float32, Float64])
        b = FullMixedPrec(copy(A0); precisions = DataType[Float16, Float32, Float64])
        lu_recursive!(a, 8; parallel=true)
        lu_recursive!(b, 8; parallel=false)
        @test reconstruct_matrix(a) ≈ reconstruct_matrix(b) rtol=1e-12
        @test _mxp_lu_err(reconstruct_matrix(b), A0) < 1e-2
    end

    @testset "two large Float16 scales do not overflow the trailing update" begin
        n = 8
        A0 = rand(n, n) .* 2.0e7 + 1.0e10I
        mp = FullMixedPrec(copy(A0); precisions = DataType[Float16, Float64], scale_type = Float16)
        @test isinf(mp.A21_scale * mp.A12_scale)
        lu_recursive!(mp, 4)
        @test all(isfinite, reconstruct_matrix(mp))
        @test _mxp_lu_err(reconstruct_matrix(mp), A0) < 1e-2
    end

    @testset "a leaf stored with a scale factor throws" begin
        mp = FullMixedPrec(fill(1.0e5, 4, 4) + 1.0e6I; precisions = DataType[Float16])
        @test mp.Base_scale != 1
        @test_throws ArgumentError lu_recursive!(mp)
    end

    # lu_recursive! reaches the container through unified_rectrxm!'s
    # mixed-precision methods. Without them the call dies with
    # "Implement KernelAbstractions.get_backend(::FullMixedPrec)", so assert
    # that entry point exists rather than only exercising it indirectly.
    @testset "mixed-precision entry point dispatches" begin
        n = 32
        A0 = rand(Float32, n, n) + 32f0 * I
        mp = FullMixedPrec(copy(A0); precisions = DataType[Float32, Float32])
        B = rand(Float32, n, 4)
        ref = copy(B)
        LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'U', 1f0, A0, ref)

        Bm = copy(B)
        @test unified_rectrxm!('L', 'L', 'N', 'U', 1f0, 'S', mp, Bm) === Bm
        @test norm(Bm - ref) / norm(ref) < 1e-3
    end

    @testset "entry point: side=$side trans=$trans func=$func" for side in ('L', 'R'), trans in ('N', 'T'), func in ('S', 'M')
        n = 32
        A0 = tril(rand(Float32, n, n)) + 32f0 * I
        mp = TriMixedPrec(A0, 'L'; precisions = DataType[Float32, Float32])
        B = side == 'L' ? rand(Float32, n, 4) : rand(Float32, 4, n)
        ref = Float64.(B)
        if func == 'S'
            LinearAlgebra.BLAS.trsm!(side, 'L', trans, 'N', 2.0, Float64.(A0), ref)
        else
            LinearAlgebra.BLAS.trmm!(side, 'L', trans, 'N', 2.0, Float64.(A0), ref)
        end
        Bm = copy(B)
        unified_rectrxm!(side, 'L', trans, 'N', 2f0, func, mp, Bm)
        @test norm(Bm - ref) / norm(ref) < 1e-4
    end

    @testset "entry point without diag takes 'N'" begin
        n = 16
        A0 = tril(rand(Float32, n, n)) + 16f0 * I
        mp = TriMixedPrec(A0, 'L'; precisions = DataType[Float32, Float32])
        B = rand(Float32, n, 3)
        B1, B2 = copy(B), copy(B)
        unified_rectrxm!('L', 'L', 'N', 1f0, 'S', mp, B1)
        unified_rectrxm!('L', 'L', 'N', 'N', 1f0, 'S', mp, B2)
        @test B1 == B2
    end

    @testset "entry point checks its flags" begin
        mp = TriMixedPrec(tril(rand(Float32, 8, 8)) + 8f0 * I, 'L'; precisions = DataType[Float32])
        B = rand(Float32, 8, 2)
        @test_throws ArgumentError unified_rectrxm!('X', 'L', 'N', 'N', 1f0, 'S', mp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'X', 'N', 'N', 1f0, 'S', mp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'X', 'N', 1f0, 'S', mp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'N', 'X', 1f0, 'S', mp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'N', 'N', 1f0, 'X', mp, B)
    end
end

# A solve that takes a Float16 block past floatmax(Float16) stores it back with
# a raised scale instead of Inf.
@testset "a solve that outgrows a Float16 block raises its scale [CPU]" begin
    blk = fill(Float16(100), 2, 2)
    s = MixedPrecision._solve_block!(W -> (W .*= 1000), blk, 1f0, Float32)
    @test s > 1 && all(isfinite, blk)
    @test Float32.(blk) .* s ≈ fill(1f5, 2, 2) rtol=2e-3
    same = fill(1f0, 2, 2)
    @test MixedPrecision._solve_block!(W -> (W .*= 2), same, 1f0, Float32) == 1f0
    @test same == fill(2f0, 2, 2)
end
