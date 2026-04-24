using KernelAbstractions
using LinearAlgebra
using Test

include("src/syrk.jl")

@testset "CPU SYRK Tests" begin
    n = 64
    k = 64
    alpha = 2f0
    beta = 1f0

    A = rand(Float32, n, k)
    C = rand(Float32, n, n)
    C = (C + C') / 2 # symmetric

    C_ref = copy(C)
    LinearAlgebra.BLAS.syrk!('L', 'N', alpha, A, beta, C_ref)

    SYRK_KERNEL!('L', 'N', alpha, A, beta, C)

    # Note: SYRK_KERNEL! might only update lower triangle, 
    # we copy the lower part for comparison
    C_lower = LowerTriangular(C)
    C_ref_lower = LowerTriangular(C_ref)
    
    result_diff = norm(C_lower - C_ref_lower) / norm(C_ref_lower)
    println("SYRK result diff: $result_diff")
    @test result_diff < 1e-5
end
