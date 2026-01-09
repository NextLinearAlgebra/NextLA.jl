using Test
using NextLA
using LinearAlgebra


@testset "slas2! test random matrices against Base Implementation" begin
    for T in [Float32, Float64]
        starting = -(floatmax(T)/T(1e10))
        ending = (floatmax(T)/T(1e10))
        for i in 1:200
            A = LinearAlgebra.UpperTriangular(starting .+ (ending - starting) .* rand(T,2,2))

            f = A[1,1]
            g = A[1,2]
            h = A[2,2]

            ssmin = Ref{T}()
            ssmax = Ref{T}()

            NextLA.slas2!(f, g, h, ssmin, ssmax)

            svs = LinearAlgebra.svdvals(A)

            @test ssmax[] ≈ svs[1]   
            @test ssmin[] ≈ svs[2]   

        end
    end
end

@testset "slas2! test overflow" begin
    A = zeros(2, 2)
    A[1,1] = 0
    A[1,2] = Float64(1.7976931348623157e+308)
    A[2,2] = Float64(1.7976931348623157e+208)
    ssmin = Ref{Float64}()
    ssmax = Ref{Float64}()

    NextLA.slas2!(A[1,1], A[1,2], A[2,2], ssmin, ssmax)

    svs = LinearAlgebra.svdvals(A)

    @test ssmax[] ≈ svs[1]   
    @test ssmin[] ≈ svs[2] 

end

@testset "slas2! test underflow" begin
    A = zeros(2, 2)
    A[1,1] = 0
    A[1,2] = Float64(4.9406564584124654e-170)
    A[2,2] = Float64(1.2251e-165)
    ssmin = Ref{Float64}()
    ssmax = Ref{Float64}()

    NextLA.slas2!(A[1,1], A[1,2], A[2,2], ssmin, ssmax)

    svs = LinearAlgebra.svdvals(A)

    @test ssmax[] ≈ svs[1]   
    @test ssmin[] ≈ svs[2] 

end
