using Test
include("report.jl")
import AMD, Scotch

@testset "Ordering and recursion invariants" begin
    A = demo(17)
    @test issymmetric(A)
    @test isposdef(Symmetric(A))
    @test first(regions(6,2)) == (lo=1,cut=4,hi=6,level=1)
    @test first(regions(8,2)) == (lo=1,cut=4,hi=8,level=1)
    # Isolated vertices and disconnected components must appear exactly once.
    D = sparse([1,2,3,4,2,1],[1,2,3,4,1,2],[3.,3.,3.,3.,-1.,-1.],4,4)
    @test sort(rcm(D,17)) == collect(1:4)
    @test sort(scotch_order(D,17)) == collect(1:4)
    mktempdir() do out
        path = joinpath(out,"triangle.mtx")
        write(path,"%%MatrixMarket matrix coordinate real general\n3 3 5\n1 1 4\n2 1 -1\n2 2 4\n3 2 -1\n3 3 4\n")
        input = loadmatrix(path)
        @test issymmetric(input)
        @test nnz(input) == 7
        @test input[1,2] == input[2,1] == -1.0
        write(path,"%%MatrixMarket matrix coordinate real general\n3 3 5\n1 1 4\n2 1 -1\n2 2 4\n2 3 -2\n3 3 4\n")
        @test_throws ErrorException loadmatrix(path)
        for method in ("natural","random","amd","rcm","scotch","cholmod")
            r = measure(A,method,17,1,2,out,3)
            @test get(r,:status,"") == "OK"
            @test r[:forward_error] < 1e-10
            @test isfile(joinpath(out,method*"-factor.svg"))
            @test occursin("<svg",read(joinpath(out,method*"-matrix.svg"),String))
        end
        @test scotch_order(A,17) == scotch_order(A,17)
        bad = measure(spdiagm(0=>[-1.,1.]),"natural",17,1,2,out,1)
        @test bad[:status] == "FAILED"
        bad[:matrix] = "indefinite"
        render(out,[bad],"test <metadata>")
        @test occursin("test &lt;metadata&gt;",read(joinpath(out,"index.html"),String))
        @test isfile(joinpath(out,"metrics.csv"))
    end
end
