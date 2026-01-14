using LinearAlgebra
using NextLA
using Test

@testset "slamrg! test random input against rotation matrix" begin
    for T in [Float16, Float32, Float64]
        for i in 50:10:100

            if (i /10)%2 == 0
                arr = [sort(rand(T, i)); sort(rand(T, i))]
                index = zeros(Int64, 2*i)

                NextLA.slamrg!(i, i, arr, 1, 1, index)
                arr = [arr[j] for j in index]

                @test issorted(arr)

            else
                arr = [sort(rand(T, i), rev=true); sort(rand(T, i))]
                index = zeros(Int64, 2*i)

                NextLA.slamrg!(i, i, arr, -1, 1, index)
                arr = [arr[j] for j in index]

                @test issorted(arr)
            end


        end
        for i in 100:100:1000
            if (i /100)%2 == 0
                arr = [sort(rand(T, i)); sort(rand(T, i))]
                index = zeros(Int64, 2*i)

                NextLA.slamrg!(i, i, arr, 1, 1, index)
                arr = [arr[j] for j in index]

                @test issorted(arr)

            else
                arr = [sort(rand(T, i)); sort(rand(T, i), rev=true)]
                index = zeros(Int64, 2*i)

                NextLA.slamrg!(i, i, arr, 1, -1, index)
                arr = [arr[j] for j in index]

                @test issorted(arr)
            end

        end
    end
end
