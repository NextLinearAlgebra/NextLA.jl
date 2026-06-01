for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "TLRMatrix allocation [$backend_name]" begin
        prototype = ArrayType(zeros(Float32, 9, 7))
        A = NextLA.TLRMatrix(prototype, 9, 7; blocksize=4, maxrank=3, compress_diag=false)

        @test A isa NextLA.TLRMatrix
        @test size(A) == (9, 7)
        @test eltype(A) == Float32
        @test A.b == 4
        @test A.mt == 3
        @test A.nt == 2
        @test A.maxrank == 3
        @test !A.compress_diag
        @test A.AUV isa NextLA.TileFactorBuffer
        @test A.AUV.order isa NextLA.TileColMajor

        @test size(A.AUV.data) == (8, 3, 6)
        @test size(A.AUV) == (3, 2)
        @test size(A.AUV[1, 1]) == (8, 3)
        @test Array(A.AUV[2, 1]) == Array(view(A.AUV.data, :, :, 2))
        @test size(A.ranks) == (3, 2)
        @test size(A.diag) == (4, 4, 2)
        @test all(iszero, Array(A.ranks))

        synchronize(A.AUV.data)
        synchronize(A.diag)
    end
end

@testset "TLRMatrix tile order" begin
    Acol = NextLA.TLRMatrix(zeros(Float64, 8, 12); blocksize=4, maxrank=2, tile_order=NextLA.TileColMajor)
    Arow = NextLA.TLRMatrix(zeros(Float64, 8, 12); blocksize=4, maxrank=2, tile_order=NextLA.TileRowMajor)

    @test NextLA.tile_linear_index(Acol, 1, 1) == 1
    @test NextLA.tile_linear_index(Acol, 2, 1) == 2
    @test NextLA.tile_linear_index(Acol, 1, 2) == 3

    @test NextLA.tile_linear_index(Arow, 1, 1) == 1
    @test NextLA.tile_linear_index(Arow, 1, 2) == 2
    @test NextLA.tile_linear_index(Arow, 2, 1) == 4
end

@testset "TLRMatrix compress_diag=true" begin
    A = NextLA.TLRMatrix(zeros(Float64, 5, 5); blocksize=2, maxrank=3, compress_diag=true)

    @test size(A) == (5, 5)
    @test size(A.AUV.data) == (4, 3, 9)
    @test size(A.ranks) == (3, 3)
    @test size(A.diag) == (2, 2, 0)
    @test A.compress_diag
end

@testset "TLRMatrix validation" begin
    @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), -1, 5; blocksize=2, maxrank=2)
    @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 5, 5; blocksize=0, maxrank=2)
    @test_throws ArgumentError NextLA.TLRMatrix(zeros(Float64, 5, 5), 5, 5; blocksize=2, maxrank=-1)

    A = NextLA.TLRMatrix(zeros(Float64, 8, 8); blocksize=4, maxrank=2)
    @test_throws BoundsError NextLA.tile_linear_index(A, 3, 1)
    @test_throws BoundsError NextLA.tile_linear_index(A, 1, 3)
end

@testset "GeneralTLRMatrix allocation" begin
    A = NextLA.GeneralTLRMatrix(
        zeros(Float64, 1),
        [3, 5, 2],
        [3, 5];
        maxrank=2,
        compress_diag=false,
    )

    @test size(A) == (10, 8)
    @test eltype(A) == Float64
    @test A.mt == 3
    @test A.nt == 2
    @test A.maxrank == 2
    @test !A.compress_diag
    @test A.order isa NextLA.TileColMajor

    @test A.rowsizes == [3, 5, 2]
    @test A.colsizes == [3, 5]
    @test A.rowptr == [1, 4, 9, 11]
    @test A.colptr == [1, 4, 9]
    @test size(A.offsets) == (3, 2)
    @test size(A.ranks) == (3, 2)
    @test length(A.diagoffsets) == 2
    @test all(iszero, Array(A.ranks))

    @test length(A.UV) == 88
    @test length(A.diag) == 34
    @test Array(A.offsets) == Int64[1 39; 13 55; 29 75]
    @test Array(A.diagoffsets) == Int64[1, 10]
end

@testset "GeneralTLRMatrix tile order" begin
    Acol = NextLA.GeneralTLRMatrix(zeros(Float64, 1), [4, 4], [4, 4, 4]; maxrank=2, tile_order=NextLA.TileColMajor)
    Arow = NextLA.GeneralTLRMatrix(zeros(Float64, 1), [4, 4], [4, 4, 4]; maxrank=2, tile_order=NextLA.TileRowMajor)

    @test NextLA.tile_linear_index(Acol, 1, 1) == 1
    @test NextLA.tile_linear_index(Acol, 2, 1) == 2
    @test NextLA.tile_linear_index(Acol, 1, 2) == 3

    @test NextLA.tile_linear_index(Arow, 1, 1) == 1
    @test NextLA.tile_linear_index(Arow, 1, 2) == 2
    @test NextLA.tile_linear_index(Arow, 2, 1) == 4
    @test Array(Acol.offsets) == Int64[1 33 65; 17 49 81]
    @test Array(Arow.offsets) == Int64[1 17 33; 49 65 81]
end

@testset "GeneralTLRMatrix validation" begin
    @test_throws ArgumentError NextLA.GeneralTLRMatrix(zeros(Float64, 1), Int[], [4]; maxrank=2)
    @test_throws ArgumentError NextLA.GeneralTLRMatrix(zeros(Float64, 1), [4], Int[]; maxrank=2)
    @test_throws ArgumentError NextLA.GeneralTLRMatrix(zeros(Float64, 1), [4, 0], [4]; maxrank=2)
    @test_throws ArgumentError NextLA.GeneralTLRMatrix(zeros(Float64, 1), [4], [4]; maxrank=-1)
    @test_throws ArgumentError NextLA.GeneralTLRMatrix(zeros(Float64, 1), [3, 5], [4, 5]; maxrank=2)
end

@testset "similar_tlr compatibility alias" begin
    A = NextLA.similar_tlr(zeros(Float64, 8, 12); blocksize=4, maxrank=2)
    @test A isa NextLA.TLRMatrix
    @test size(A) == (8, 12)
end
