using Test
using KernelAbstractions

function _compressed_ftlr_to_backend(A::NextLA.CompressedFTLRMatrix, AT)
    qm, qn = NextLA.grid_size(A)
    rank_grid = [Int(NextLA.ranks(A)[_TLRM._rank_index(A, i, j)]) for i in 1:qm, j in 1:qn]
    B = NextLA.CompressedFTLRMatrix(KernelAbstractions.get_backend(AT(zeros(eltype(A), 1))),
                           eltype(A), size(A, 1), size(A, 2),
                           NextLA.nominal_tile_size(A),
                           rank_grid;
                           outer_order=typeof(A.outer.order), inner_order=typeof(A.inner.order))
    copyto!(B.outer.data, AT(Array(A.outer.data)))
    copyto!(B.inner.data, AT(Array(A.inner.data)))
    return B
end

@testset "CompressedFTLR CUDA logical transpose combinations" begin
    Ahost = _compressed_ftlr_fixture(Float32, Int[1 2; 3 1])
    Bhost = _compressed_ftlr_fixture(Float32, Int[2 1; 1 3])
    referenceA = zeros(Float32, size(Ahost)); referenceB = zeros(Float32, size(Bhost))
    NextLA.uncompress!(referenceA, Ahost); NextLA.uncompress!(referenceB, Bhost)
    for (name, AT, sync) in backends
        name == "CUDA" || continue
        A = _compressed_ftlr_to_backend(Ahost, AT)
        B = _compressed_ftlr_to_backend(Bhost, AT)
        for transA in ('N', 'T'), transB in ('N', 'T')
            opA = transA == 'N' ? referenceA : transpose(referenceA)
            opB = transB == 'N' ? referenceB : transpose(referenceB)
            minbytes = NextLA.gemm_minimum_workspace_bytes(A, B; transA, transB)
            maxbytes = NextLA.gemm_maximum_workspace_bytes(A, B; transA, transB)
            for bytes in unique((minbytes, maxbytes))
                C0 = rand(Float32, size(opA, 1), size(opB, 2))
                C = AT(copy(C0))
                _TLRM.gemm!(C, A, B; workspace=bytes, alpha=1.25f0, beta=-0.5f0,
                             transA, transB)
                sync(C)
                @test Array(C) ≈ 1.25f0 .* (opA * opB) .- 0.5f0 .* C0 rtol=2f-4 atol=2f-4
            end
        end
    end
end

@testset "CompressedFTLR CUDA dense GEMM" begin
    Ahost = _compressed_ftlr_fixture(Float32)
    Bhost = _compressed_ftlr_fixture(Float32, Int[1 0; 2 1; 3 1])
    @test_throws ArgumentError _TLRM.gemm!(zeros(Float32, 8, 8), Ahost, Bhost;
                                             workspace=NextLA.gemm_minimum_workspace_bytes(Ahost, Bhost))
    for (name, AT, sync) in backends
        name == "CUDA" || continue
        A = _compressed_ftlr_to_backend(Ahost, AT)
        B = _compressed_ftlr_to_backend(Bhost, AT)
        referenceA = zeros(Float32, size(A)); referenceB = zeros(Float32, size(B))
        NextLA.uncompress!(referenceA, Ahost); NextLA.uncompress!(referenceB, Bhost)
        reconstructed = AT(zeros(Float32, size(A)))
        NextLA.uncompress!(reconstructed, A)
        sync(reconstructed)
        @test Array(reconstructed) ≈ referenceA
        minbytes = NextLA.gemm_minimum_workspace_bytes(A, B)
        maxbytes = NextLA.gemm_maximum_workspace_bytes(A, B)
        for bytes in unique((minbytes, maxbytes))
            C0 = rand(Float32, size(A, 1), size(B, 2))
            C = AT(copy(C0))
            _TLRM.gemm!(C, A, B; workspace=bytes, alpha=1.5f0, beta=-0.25f0)
            sync(C)
            @test Array(C) ≈ 1.5f0 .* (referenceA * referenceB) .- 0.25f0 .* C0 rtol=2f-4 atol=2f-4
        end
    end
end
