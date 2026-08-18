using KernelAbstractions

function _compressed_ftlr_fixture(
    ::Type{T}=Float64,
    ranks::AbstractMatrix{<:Integer}=Int[1 2 0; 3 1 2];
    rank_multiple::Int=0) where {T}
    qm, qn = size(ranks)
    b = max(4, NextLA.gemm_alignment_quantum(T))
    A = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), T, b*qm, b*qn, b, ranks; rank_multiple)
    fill_random_tlr!(A, Array; seed=123)
    return A
end

@testset "CompressedFTLR complementary packing and reconstruction" begin
    rank_grid = Int[1 2 0; 2 1 1; 1 1 1]
    A = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 9, 10, (4, 4), rank_grid;
        rank_multiple=8)
    fill_random_tlr!(A, Array; seed=88)

    @test A.outer.offsets[1] == 1
    @test A.inner.offsets[1] == 1
    @test issorted(A.outer.offsets)
    @test issorted(A.inner.offsets)
    @test A.outer.offsets[end] - 1 == length(A.outer.data)
    @test A.inner.offsets[end] - 1 == length(A.inner.data)

    dense = zeros(Float64, size(A))
    NextLA.uncompress!(dense, A)
    for j in 1:3, i in 1:3
        p0, q0 = NextLA.tile_origin_coords(A, i, j)
        tm, tn = NextLA.tile_size(A, i, j)
        U, V = NextLA.get_factors(A, i, j)
        reference = isempty(U) ? zeros(tm, tn) : Matrix(U) * Matrix(V)'
        @test dense[p0:p0+tm-1, q0:q0+tn-1] ≈ reference
    end
end

@testset "rank_multiple replaces symbolic execution policies" begin
    ranks = Int[1 9; 3 8]
    exact = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks)
    q8 = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks;
        rank_multiple=8)
    q16 = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 32, 32, 16, ranks;
        rank_multiple=16)
    @test [_TLRM._compressed_ftlr_storage_rank(exact, i, j)
           for i in 1:2, j in 1:2] == ranks
    @test [_TLRM._compressed_ftlr_storage_rank(q8, i, j)
           for i in 1:2, j in 1:2] == Int[8 16; 8 8]
    @test [_TLRM._compressed_ftlr_storage_rank(q16, i, j)
           for i in 1:2, j in 1:2] == fill(16, 2, 2)
end
