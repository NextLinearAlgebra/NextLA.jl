using Test
using NextLA
using LinearAlgebra


@testset "slas2! test random matrices against Base Implementation" begin
    for T in [Float16, Float32, Float64]
        starting = -(floatmax(T)/T(1e10))
        ending = (floatmax(T)/T(1e10))
        for i in 1:200
            A = LinearAlgebra.UpperTriangular(starting .+ (ending - starting) .* rand(T,2,2))
            A_prime = copy(A)


            svs = LinearAlgebra.svdvals(A)

            NextLA.slas2!(A_prime)


            @test A_prime[1,1] ≈ svs[1]   
            @test A_prime[2,2] ≈ svs[2]   

        end
    end
end

@testset "slas2! test overflow" begin

    A = zeros(2, 2)
    A[1,1] = 0
    A[1,2] = Float64(1.7976931348623157e+308)
    A[2,2] = Float64(1.7976931348623157e+208)
    A = LinearAlgebra.UpperTriangular(A)
    A_prime = copy(A)
    
    svs = LinearAlgebra.svdvals(A)
    NextLA.slas2!(A_prime)

    @test A_prime[1,1] ≈ svs[1]   
    @test A_prime[2,2] ≈ svs[2] 

end

@testset "slas2! test underflow" begin
    A = zeros(2, 2)
    A[1,1] = 0
    A[1,2] = Float64(4.9406564584124654e-170)
    A[2,2] = Float64(1.2251e-165)
    A = LinearAlgebra.UpperTriangular(A)
    A_prime = copy(A)

    
    svs = LinearAlgebra.svdvals(A)
    NextLA.slas2!(A_prime)

    @test A_prime[1,1] ≈ svs[1]   
    @test A_prime[2,2] ≈ svs[2] 

end
