using Test, LinearAlgebra, SparseArrays
include("matrix_io.jl")

@testset "Catalog MatrixMarket loading" begin
    mktempdir() do dir
        for header in ("general", "symmetric")
            path = joinpath(dir, header * ".mtx")
            write(path, "%%MatrixMarket matrix coordinate real $header\n2 2 3\n1 1 4\n2 1 -1\n2 2 3\n")
            @test Matrix(load_input(path, 2)) == [4 -1; -1 3]
            @test_throws ErrorException load_input(path, 3)
        end
        path = joinpath(dir, "bad.mtx")
        write(path, "%%MatrixMarket matrix coordinate real general\n2 2 4\n1 1 4\n2 1 -1\n1 2 -2\n2 2 3\n")
        @test_throws ErrorException load_input(path, 2)
    end
end
