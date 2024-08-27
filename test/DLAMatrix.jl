@testset "DLAMatrix" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n = 10
        A = rand(T, 10, 10)
        DLA_A = DLAMatrix{T}(A)
        @test A == DLA_A.data
        @test size(DLA_A) == (10, 10)
        @test size(DLA_A, 1) == 10
        @test size(DLA_A, 2) == 10
        @test length(DLA_A) == 100
        @test size(A) == (10, 10)
        @test size(A, 1) == 10
        @test size(A, 2) == 10
        @test length(A) == 100
    end
end