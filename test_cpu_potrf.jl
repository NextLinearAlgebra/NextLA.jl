using KernelAbstractions
using LinearAlgebra
using Test

include("src/potrf.jl")

@testset "CPU POTRF Tests" begin
    n = 64
    A = rand(Float32, n, n)
    A = (A * A') + I * n # positive definite

    A_ref = copy(A)
    L_ref = cholesky(A_ref).L

    backend = KernelAbstractions.get_backend(A)
    kernel = chol_kernel_lower!(backend, MAX_THREADS)
    kernel(A, Val(n); ndrange=MAX_THREADS)
    KernelAbstractions.synchronize(backend)

    A_lower = LowerTriangular(A)
    
    result_diff = norm(A_lower - L_ref) / norm(L_ref)
    println("POTRF result diff: $result_diff")
    @test result_diff < 1e-4
end
