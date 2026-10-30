# Requires the benchmark dependencies. Source loading and matrix I/O need no GPU.
include("matrix_io_tests.jl")
include("accuracy.jl")

using CUDA
include("solver.jl")
const S = CatalogSolver

@testset "CUDA source loading and dispatch" begin
    @test hasmethod(S.unified_rec, Tuple{Char,Char,Char,Char,CUDA.CuMatrix{Float32},CUDA.CuMatrix{Float32},Int})
    @test hasmethod(S.unified_rec_mixed, Tuple{Char,Char,Char,Char,S.TriMixedPrec{Float32},CUDA.CuMatrix{Float32},Int})
    A = Matrix{Float64}(I,4,4) .* 1e6
    A[3:4,1:2] .= [131008.0 0.75; -0.75 -131008.0]
    A[1:2,3:4] .= transpose(A[3:4,1:2])
    mp = S.SymmMixedPrec(A,'L'; precisions=DataType[Float16,Float64])
    @test decode_blocks(mp) == A
end

if CUDA.functional()
    @testset "GPU factorization with scaling active" begin
        L = Matrix{Float64}(I, 8, 8) .* 1000
        L[5:8,1:4] .= [(-1.0)^(i+j) * (150+25i+10j) for i in 1:4, j in 1:4]
        A = L * L'
        for precisions in (DataType[Float16,Float32], DataType[Float16,Float16,Float32])
            mp = S.SymmMixedPrec(CuArray(A),'L'; precisions=precisions)
            @test mp.offDiag_scale > 1
            S.potrf_recursive!(mp)
            factor = Array(tril!(decode_blocks(mp)))
            @test norm(factor*factor'-A)/norm(A) < 1e-3
        end
    end
elseif get(ENV, "REQUIRE_CUDA", "0") == "1"
    error("CUDA device unavailable: refusing to start the GPU catalog sweep")
else
    @info "GPU numerical checks skipped: CUDA device unavailable"
end
