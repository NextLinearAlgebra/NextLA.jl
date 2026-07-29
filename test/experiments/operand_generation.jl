using LinearAlgebra
using Test
using NextLA

include(joinpath(@__DIR__, "../../experiments/operand_generation.jl"))
using .ExperimentMatrixGeneration

@testset "experiment TLR matrix generation" begin
    A, B = generate_tlr_operands(
        8, 8, 8, 4, (2, 2), Float64; seed=13, shared_rank=1)
    A_again, B_again = generate_tlr_operands(
        8, 8, 8, 4, (2, 2), Float64; seed=13, shared_rank=1)

    @test size(A) == (8, 8)
    @test size(B) == (8, 8)
    @test all(Int.(A.ranks) .== 2)
    @test all(Int.(B.ranks) .== 2)
    @test A.int_U == A_again.int_U
    @test B.int_V == B_again.int_V

    UA1 = Matrix(NextLA.get_factors(A, 1, 1)[1])
    UA2 = Matrix(NextLA.get_factors(A, 1, 2)[1])
    VB1 = Matrix(NextLA.get_factors(B, 1, 1)[2])
    VB2 = Matrix(NextLA.get_factors(B, 2, 1)[2])
    @test UA1[:, 1] == UA2[:, 1]
    @test VB1[:, 1] == VB2[:, 1]
    @test norm(UA1[:, 1]' * UA2[:, 2]) < 100eps()
    @test norm(VB1[:, 1]' * VB2[:, 2]) < 100eps()
end

@testset "padded operand tile orders" begin
    A, B = generate_ftlr_operands(
        8, 8, 8, 4, (2, 2), Float32;
        seed=14,
        format=:padded,
        padded_orders=(NextLA.TileRowMajor, NextLA.TileColMajor),
    )
    @test NextLA.TLRmodule.tile_order(A) isa NextLA.TileRowMajor
    @test NextLA.TLRmodule.tile_order(B) isa NextLA.TileColMajor
    @test all(Int.(A.ranks) .== 2)
    @test all(Int.(B.ranks) .== 2)
end

@testset "variable-rank FTLR generation" begin
    for distribution in (:constant, :uniform, :skewed)
        A, B = generate_ftlr_operands(16, 16, 16, 4, (3, 3), Float32;
            seed=77, format=:compressed, rank_distribution=distribution,
            min_rank=1, max_rank=3)
        @test A isa NextLA.CompressedFTLRMatrix
        @test B isa NextLA.CompressedFTLRMatrix
        @test all(1 .<= Int.(A.ranks) .<= 3)
        @test all(1 .<= Int.(B.ranks) .<= 3)
    end
    A, B = generate_ftlr_operands(8, 8, 8, 4, (2, 2), Float64;
        seed=5, format=:compressed, rank_distribution=:constant,
        min_rank=2, max_rank=2)
    U1, V1 = NextLA.get_factors(A, 1, 1)
    U2, V2 = NextLA.get_factors(B, 1, 1)
    @test size(U1, 2) == size(V1, 2) == 2
    @test size(U2, 2) == size(V2, 2) == 2
end
