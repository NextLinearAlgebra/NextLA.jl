using Test
using KernelAbstractions

function _compressed_ftlr_fixture(::Type{T}=Float64, ranks::AbstractMatrix{<:Integer}=Int[1 2 0; 3 1 2]) where {T}
    qm, qn = size(ranks)
    A = NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), T, 4 * qm, 4 * qn, (4, 4), ranks;
                          outer_order=NextLA.TileRowMajor,
                          inner_order=NextLA.TileColMajor)
    rng = MersenneTwister(123)
    for j in 1:qn, i in 1:qm
        U, V = NextLA.get_factors(A, i, j)
        U .= randn(rng, T, size(U))
        V .= randn(rng, T, size(V))
    end
    return A
end

@testset "CompressedFTLR exact-rank container" begin
    defaults = NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float64, 4, 4, 4, ones(Int, 1, 1))
    @test defaults.outer.order isa NextLA.TileRowMajor
    @test defaults.inner.order isa NextLA.TileColMajor

    A = _compressed_ftlr_fixture()
    @test NextLA.grid_size(A) == (2, 3)
    @test NextLA.tail_tile_size(A) == (0, 0)
    @test NextLA.maxrank(A) == 3
    @test size(NextLA.get_factors(A, 1, 2)[1]) == (4, 2)
    @test size(NextLA.get_factors(A, 1, 2)[2]) == (4, 2)
    @test size(NextLA.get_factors(A, 1, 3)[1]) == (4, 0)
    @test A.outer.offsets[end] - 1 == 4 * sum(Int.(NextLA.ranks(A)))
    @test A.inner.offsets[end] - 1 == 4 * sum(Int.(NextLA.ranks(A)))

    dense = zeros(Float64, size(A))
    NextLA.uncompress!(dense, A)
    expected = zeros(Float64, size(A))
    for j in 1:3, i in 1:2
        U, V = NextLA.get_factors(A, i, j)
        expected[(i - 1) * 4 + 1:i * 4, (j - 1) * 4 + 1:j * 4] .= U * V'
    end
    @test dense ≈ expected

    padded = NextLA.PaddedFTLRMatrix(KernelAbstractions.CPU(), Float64, 8, 12, (4, 4), 3)
    padded.ranks .= NextLA.ranks(A)
    for j in 1:3, i in 1:2
        r = Int(NextLA.ranks(A)[_TLRM._rank_index(A, i, j)])
        r == 0 && continue
        U, V = NextLA.get_factors(A, i, j)
        Up, Vp = NextLA.get_factors(padded, i, j)
        copyto!(Up, U); copyto!(Vp, V)
    end
    packed = NextLA.pack_compressed_ftlr(padded;
                               outer_order=NextLA.TileRowMajor,
                               inner_order=NextLA.TileColMajor)
    dense_packed = zeros(Float64, size(A))
    NextLA.uncompress!(dense_packed, packed)
    @test dense_packed ≈ dense
    @test_throws ArgumentError NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float64, 9, 8, 4, ones(Int, 2, 2))
end

@testset "CompressedFTLR ragged workspace profile" begin
    A = _compressed_ftlr_fixture()
    B = _compressed_ftlr_fixture(Float64, Int[1 0; 2 1; 3 1])
    profile = _TLRM._compressed_ftlr_workspace_profile(A, B)
    @test profile.minimum == maximum(profile.row_bytes)
    @test profile.maximum == min(sum(profile.right_row_bytes), sum(profile.left_row_bytes))
    @test NextLA.gemm_minimum_workspace_bytes(A, B) == profile.minimum
    @test NextLA.gemm_maximum_workspace_bytes(A, B) == profile.maximum
    @test length(_TLRM._compressed_ftlr_row_runs(profile, profile.minimum)) >= 1
    @test length(_TLRM._compressed_ftlr_row_runs(profile, profile.maximum)) == 1
    @test_throws ArgumentError _TLRM._compressed_ftlr_row_runs(profile, profile.minimum - 1)

    Aleft = NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float64, 8, 12, 4,
                               Int[1 2 0; 3 1 2];
                               outer_order=NextLA.TileColMajor,
                               inner_order=NextLA.TileColMajor)
    left_profile = _TLRM._compressed_ftlr_workspace_profile(Aleft, B)
    @test left_profile.right_row_bytes === nothing
    @test all(run.fold === :left for run in _TLRM._compressed_ftlr_row_runs(left_profile, left_profile.maximum))
end
