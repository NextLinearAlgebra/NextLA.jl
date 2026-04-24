using KernelAbstractions
using LinearAlgebra
using Test

include("src/trsm.jl")

@testset "CPU TRSM Tests" begin
    n = 128
    m = 128
    A = Matrix(LowerTriangular(rand(Float32, n, n) .+ 1))
    A .+= Diagonal(10.0 * ones(Float32, n))
    B = rand(Float32, n, m) .+ 1

    Bc = copy(B)

    # Call our kernel
    LeftLowerTRSM!(A, B)

    # Baseline with BLAS trsm!
    LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'N', one(eltype(A)), A, Bc)

    result_diff = norm(B .- Bc) / norm(Bc)
    println("LeftLowerTRSM result diff: $result_diff")
    @test result_diff < 1e-5
end
