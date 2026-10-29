using Test
include("download.jl")

@testset "Downloaded file validation" begin
    entry = Dict("name"=>"tiny", "n"=>"2", "nnz_lower"=>"3")
    mktempdir() do dir
        path = joinpath(dir,"tiny.mtx")
        write(path,"%%MatrixMarket matrix coordinate real symmetric\n% test\n2 2 3\n1 1 4\n2 1 -1\n2 2 3\n")
        @test check_matrix_file(path, entry)
        write(path,"%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 4\n")
        @test_throws ErrorException check_matrix_file(path, entry)
        write(path,"<html>Not found</html>")
        @test_throws ErrorException check_matrix_file(path, entry)
        write(path,"%%MatrixMarket matrix coordinate real symmetric\n3 3 3\n1 1 4\n2 1 -1\n2 2 3\n")
        @test_throws ErrorException check_matrix_file(path, entry)
    end
end
