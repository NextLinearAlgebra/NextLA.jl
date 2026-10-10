using NextLA.MixedPrecision
# recsyrk! over a SymmMixedPrec container. CPU only.

_mxp_tri(uplo, X) = uplo == 'L' ? tril(X) : triu(X)

@testset "RECSYRK mixed precision [CPU]" begin
    Random.seed!(2055)

    @testset "SymmMixedPrec n=$n uplo=$uplo" for n in [32, 45, 64], uplo in ('L', 'U')
        T = Float32
        A0 = rand(T, n, 16)
        C0 = rand(T, n, n); C0 = C0 + transpose(C0)
        mp = SymmMixedPrec(copy(C0), uplo; precisions = DataType[Float32, Float32])

        @test NextLA.recsyrk!(T(1.5), A0, T(0.5), mp) === mp

        expected = T(1.5) * (A0 * transpose(A0)) + T(0.5) * C0
        R = reconstruct_matrix(mp)
        @test norm(_mxp_tri(uplo, R) - _mxp_tri(uplo, expected)) / norm(_mxp_tri(uplo, expected)) < 1e-3
        @test R ≈ expected rtol=1e-3
    end

    @testset "Float16 off-diagonal blocks, uplo=$uplo" for uplo in ('L', 'U')
        n = 32
        A0 = rand(Float32, n, 8)
        C0 = rand(Float32, n, n); C0 = C0 + transpose(C0)
        mp = SymmMixedPrec(copy(C0), uplo; precisions = DataType[Float16, Float32])
        NextLA.recsyrk!(1f0, A0, 1f0, mp)
        @test reconstruct_matrix(mp) ≈ A0 * transpose(A0) + C0 rtol=1e-2
    end

    @testset "a scaled block takes alpha divided by its scale" begin
        n = 8
        C0 = ones(Float64, n, n)
        C0[5:8, 1:4] .= 1.0e5
        C0[1:4, 5:8] .= 1.0e5
        mp = SymmMixedPrec(copy(C0), 'L'; precisions = DataType[Float16, Float64])
        @test mp.OffDiag_scale != 1
        A0 = rand(Float64, n, 4)
        NextLA.recsyrk!(-100.0, A0, 1.0, mp)
        @test reconstruct_matrix(mp) ≈ -100 * A0 * transpose(A0) + C0 rtol=1e-2
    end

    @testset "parallel and sequential agree" begin
        n = 32
        A0 = rand(Float64, n, 8)
        C0 = rand(Float64, n, n); C0 = C0 + transpose(C0)
        seq = SymmMixedPrec(copy(C0), 'L'; precisions = DataType[Float32, Float64])
        par = SymmMixedPrec(copy(C0), 'L'; precisions = DataType[Float32, Float64])
        MixedPrecision._recsyrk_mixed!(1.0, A0, 1.0, seq; parallel=false)
        MixedPrecision._recsyrk_mixed!(1.0, A0, 1.0, par; parallel=true)
        @test reconstruct_matrix(seq) ≈ reconstruct_matrix(par) rtol=1e-12
    end

    @testset "A with the wrong number of rows throws" begin
        mp = SymmMixedPrec(rand(8, 8), 'L'; precisions = DataType[Float64])
        @test_throws DimensionMismatch NextLA.recsyrk!(1.0, rand(7, 3), 1.0, mp)
    end
end

# An update past floatmax of the Float16 off-diagonal block raises its scale.
@testset "RECSYRK mixed precision re-scales an overflowing block [CPU]" begin
    C_mp = SymmMixedPrec(zeros(4, 4), 'L'; precisions = DataType[Float16, Float32])
    A = fill(1000.0, 4, 2)
    NextLA.recsyrk!(1.0, A, 0.0, C_mp)
    @test C_mp.OffDiag_scale > 1
    @test reconstruct_matrix(C_mp) ≈ A * A' rtol=1e-2
end
