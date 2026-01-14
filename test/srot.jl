using LinearAlgebra
using NextLA
using Test

@testset "srot! test random input against rotation matrix large input" begin
    for T in [Float16, Float32, Float64]
        starting = -(floatmax(T)/T(1e10))
        ending = (floatmax(T)/T(1e10))
        for i in 1:100
            v = (starting .+ (ending - starting) .* rand(T,2))
            rand_angle = 2pi*rand(T)
            c = T(cos(rand_angle))
            s = T(sin(rand_angle))

            rotation_matrix = [c s; -s c]


            matrix_output = rotation_matrix * v

            NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


            @test matrix_output[1] ≈ v[1]
            @test matrix_output[2] ≈ v[2]   

        end
    end
end
@testset "srot! test random input against rotation matrix small input" begin
    for T in [Float16, Float32, Float64]
        starting = -T(0)
        ending = T(1)
        for i in 1:100
            v = (starting .+ (ending - starting) .* rand(T,2))
            rand_angle = 2pi*rand(T)
            c = T(cos(rand_angle))
            s = T(sin(rand_angle))

            rotation_matrix = [c s; -s c]


            matrix_output = rotation_matrix * v

            NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


            @test matrix_output[1] ≈ v[1] 
            @test matrix_output[2] ≈ v[2]   

        end
    end
end
@testset "srot! test random input against rotation matrix medium input" begin
    for T in [Float16, Float32, Float64]
        starting = -T(1e3)
        ending = T(1e3)
        for i in 1:100
            v = (starting .+ (ending - starting) .* rand(T,2))
            rand_angle = 2pi*rand(T)
            c = T(cos(rand_angle))
            s = T(sin(rand_angle))

            rotation_matrix = [c s; -s c]


            matrix_output = rotation_matrix * v

            NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


            @test matrix_output[1] ≈ v[1] 
            @test matrix_output[2] ≈ v[2]   

        end
    end
end
@testset "srot! test random input against rotation matrix large medium input" begin
    for T in [Float32, Float64]
        starting = -T(1e4)
        ending = T(1e7)
        for i in 1:100
            v = (starting .+ (ending - starting) .* rand(T,2))
            rand_angle = 2pi*rand(T)
            c = T(cos(rand_angle))
            s = T(sin(rand_angle))

            rotation_matrix = [c s; -s c]


            matrix_output = rotation_matrix * v

            NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


            @test matrix_output[1] ≈ v[1] 
            @test matrix_output[2] ≈ v[2]   

        end
    end
end
@testset "srot! test angle out of range" begin
    for T in [Float16, Float32, Float64]
        starting = -T(1e3)
        ending = T(1e3)
        for i in 1:100
            v = (starting .+ (ending - starting) .* rand(T,2))
            rand_angle = 2pi + (2pi + (20 - 2pi)) * rand(T)
            c = T(cos(rand_angle))
            s = T(sin(rand_angle))

            rotation_matrix = [c s; -s c]


            matrix_output = rotation_matrix * v

            NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


            @test matrix_output[1] ≈ v[1]   
            @test matrix_output[2] ≈ v[2]   

        end
    end
end
@testset "srot! test zero and 2pi" begin
    for T in [Float16, Float32, Float64]
        starting = -(floatmax(T)/T(1e10))
        ending = (floatmax(T)/T(1e10))
        v = (starting .+ (ending - starting) .* rand(T,2))
        rand_angle = 0
        c = T(cos(rand_angle))
        s = T(sin(rand_angle))

        rotation_matrix = [c s; -s c]


        matrix_output = rotation_matrix * v

        NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


        @test matrix_output[1] ≈ v[1]   
        @test matrix_output[2] ≈ v[2]   

        rand_angle = 2pi
        c = T(cos(rand_angle))
        s = T(sin(rand_angle))

        rotation_matrix = [c s; -s c]


        matrix_output = rotation_matrix * v

        NextLA.srot!(1, view(v, 1:1), 1, view(v, 2:2), 1, c, s)


        @test matrix_output[1] ≈ v[1]   
        @test matrix_output[2] ≈ v[2]   

    end
end
