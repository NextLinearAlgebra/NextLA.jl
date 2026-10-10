using NextLA.MixedPrecision
# recgemm! into a FullMixedPrec destination. CPU only: the container walks
# with scalar indexing on the host, the limitation recorded for src/mxp/.

@testset "RECGEMM mixed precision [CPU]" begin
    Random.seed!(2049)

    @testset "FullMixedPrec destination, n=$n" for n in [5, 8, 13, 16]
        T = Float32
        A = rand(T, n, n)
        B = rand(T, n, n)
        C_dense = rand(T, n, n)
        alpha, beta = T(1.5), T(0.5)
        expected = alpha * A * B + beta * C_dense

        C_mp = FullMixedPrec(C_dense; precisions=DataType[Float32, Float32])
        @test NextLA.recgemm!(alpha, A, B, beta, C_mp) === C_mp
        @test reconstruct_matrix(C_mp) ≈ expected rtol=1e-4
    end

    @testset "a leaf C takes A n×k and B k×n, k=$k" for k in [1, 3, 8, 20]
        n = 8
        A = rand(Float64, n, k)
        B = rand(Float64, k, n)
        C_dense = rand(Float64, n, n)
        C_mp = FullMixedPrec(copy(C_dense); precisions=DataType[Float64])
        NextLA.recgemm!(1.0, A, B, 1.0, C_mp)
        @test reconstruct_matrix(C_mp) ≈ A * B + C_dense rtol=1e-12
    end

    @testset "a split C needs square A and B, k=$k" for k in [3, 20]
        n = 8
        C_mp = FullMixedPrec(rand(Float64, n, n); precisions=DataType[Float32, Float64])
        @test_throws DimensionMismatch NextLA.recgemm!(1.0, rand(n, k), rand(k, n), 1.0, C_mp)
    end

    @testset "an uneven split below the top needs its leaves, n=$n" for n in [5, 13]
        # n = 5 splits 4 + 1: C11's second product A12 * B21 has k = 1. That is fine while
        # C11 is a leaf and a DimensionMismatch once C11 is split again.
        A, B, C_dense = rand(n, n), rand(n, n), rand(n, n)
        leafy = FullMixedPrec(copy(C_dense); precisions=DataType[Float32, Float64])
        NextLA.recgemm!(1.0, A, B, 1.0, leafy)
        @test reconstruct_matrix(leafy) ≈ A * B + C_dense rtol=1e-5
        deep = FullMixedPrec(copy(C_dense); precisions=DataType[Float32, Float32, Float64])
        @test_throws DimensionMismatch NextLA.recgemm!(1.0, A, B, 1.0, deep)
    end

    @testset "Float16 off-diagonal blocks" begin
        n = 16
        A = rand(Float32, n, n)
        B = rand(Float32, n, n)
        C_dense = rand(Float32, n, n)
        C_mp = FullMixedPrec(C_dense; precisions=DataType[Float16, Float32])
        expected = A * B + C_dense
        NextLA.recgemm!(1f0, A, B, 1f0, C_mp)
        @test eltype(C_mp.A21) === Float16
        @test reconstruct_matrix(C_mp) ≈ expected rtol=1e-2
    end

    @testset "a scaled block takes alpha divided by its scale" begin
        n = 8
        C_dense = ones(Float64, n, n)
        C_dense[5:8, 1:4] .= 1.0e5
        C_mp = FullMixedPrec(copy(C_dense); precisions=DataType[Float16, Float64])
        @test C_mp.A21_scale != 1
        A = rand(Float64, n, n)
        B = rand(Float64, n, n)
        expected = -100 * A * B + C_dense
        NextLA.recgemm!(-100.0, A, B, 1.0, C_mp)
        @test reconstruct_matrix(C_mp) ≈ expected rtol=1e-2
    end

    @testset "beta = 0 overwrites C" begin
        n = 8
        A = rand(Float64, n, n)
        B = rand(Float64, n, n)
        C_mp = FullMixedPrec(rand(Float64, n, n); precisions=DataType[Float32, Float64])
        NextLA.recgemm!(1.0, A, B, 0.0, C_mp)
        @test reconstruct_matrix(C_mp) ≈ A * B rtol=1e-5
    end

    @testset "parallel and sequential agree" begin
        n = 16
        T = Float32
        A = rand(T, n, n)
        B = rand(T, n, n)
        C_dense = rand(T, n, n)

        C_seq = FullMixedPrec(copy(C_dense); precisions=DataType[Float32, Float32])
        C_par = FullMixedPrec(copy(C_dense); precisions=DataType[Float32, Float32])
        NextLA.recgemm!(one(T), A, B, one(T), C_seq; parallel=false)
        NextLA.recgemm!(one(T), A, B, one(T), C_par; parallel=true)

        @test reconstruct_matrix(C_seq) ≈ reconstruct_matrix(C_par) rtol=1e-5
    end

    @testset "mismatched shapes throw DimensionMismatch" begin
        C_mp = FullMixedPrec(rand(8, 8); precisions=DataType[Float64])
        @test_throws DimensionMismatch NextLA.recgemm!(1.0, rand(7, 4), rand(4, 8), 1.0, C_mp)
        @test_throws DimensionMismatch NextLA.recgemm!(1.0, rand(8, 4), rand(4, 7), 1.0, C_mp)
        @test_throws DimensionMismatch NextLA.recgemm!(1.0, rand(8, 4), rand(5, 8), 1.0, C_mp)
    end
end

# An update past floatmax of a Float16 off-diagonal block raises the block's
# scale, so the result stays finite where a fixed scale left Inf.
@testset "RECGEMM mixed precision re-scales an overflowing block [CPU]" begin
    C_mp = FullMixedPrec(zeros(4, 4); precisions = DataType[Float16, Float32])
    A, B = fill(1000.0, 4, 4), fill(1000.0, 4, 4)
    NextLA.recgemm!(1.0, A, B, 0.0, C_mp)
    @test C_mp.A12_scale > 1 && C_mp.A21_scale > 1
    @test all(isfinite, reconstruct_matrix(C_mp))
    @test reconstruct_matrix(C_mp) ≈ A * B rtol=1e-2
end
