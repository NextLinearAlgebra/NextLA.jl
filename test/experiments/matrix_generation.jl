using LinearAlgebra
using Test
using NextLA

include(joinpath(@__DIR__, "../../experiments/matrix_generation.jl"))
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
