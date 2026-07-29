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
    @test NextLA.execution_maxrank(A) == 8
    @test Int.(NextLA.execution_ranks(A)) == map(r -> iszero(r) ? 0 : 8, Int.(NextLA.ranks(A)))
    @test size(NextLA.get_factors(A, 1, 2)[1]) == (4, 2)
    @test size(NextLA.get_factors(A, 1, 2)[2]) == (4, 2)
    @test size(NextLA.get_factors(A, 1, 3)[1]) == (4, 0)
    @test A.outer.offsets[end] - 1 == 8 * sum(Int.(NextLA.execution_ranks(A)))
    @test A.inner.offsets[end] - 1 == 8 * sum(Int.(NextLA.execution_ranks(A)))
    @test all(==(0), mod.((A.outer.offsets .- 1) .* sizeof(eltype(A)), 16))
    @test all(==(0), mod.((A.inner.offsets .- 1) .* sizeof(eltype(A)), 16))
    @test all(==(0), mod.(A.outer.leading_dimensions, 8))
    @test all(==(0), mod.(A.inner.leading_dimensions, 8))

    U, V = NextLA.get_factors(A, 1, 2)
    @test _TLRM.logical_tile_factors(_TLRM.logical_operand(A), 1, 2) == (U, V)
    @test _TLRM.logical_tile_factors(_TLRM.logical_operand(A, 'T'), 2, 1) == (V, U)

    dense = zeros(Float64, size(A))
    NextLA.uncompress!(dense, A)
    expected = zeros(Float64, size(A))
    for j in 1:3, i in 1:2
        U, V = NextLA.get_factors(A, i, j)
        expected[(i - 1) * 4 + 1:i * 4, (j - 1) * 4 + 1:j * 4] .= U * V'
    end
    @test dense ≈ expected

    mapped = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float32, 16, 80, (16, 16),
        reshape(Int[0, 1, 7, 8, 9], 1, 5))
    @test Int.(NextLA.ranks(mapped)) == [0, 1, 7, 8, 9]
    @test Int.(NextLA.execution_ranks(mapped)) == [0, 8, 8, 8, 16]
    @test NextLA.maxrank(mapped) == 9
    @test NextLA.execution_maxrank(mapped) == 16
    @test size(NextLA.get_factors(mapped, 1, 5)[1]) == (16, 9)
    execution_U = _TLRM.compressed_ftlr_execution_outer(mapped, 1, 5)
    @test size(execution_U) == (16, 16)
    @test all(iszero, execution_U[:, 10:16])

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
    tailed = NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float64, 9, 10, (4, 4),
                                         Int[2 1 1; 1 2 1; 1 1 1])
    @test NextLA.grid_size(tailed) == (3, 3)
    @test NextLA.tail_tile_size(tailed) == (1, 2)
    @test size(NextLA.get_factors(tailed, 3, 3)[1]) == (1, 1)
    @test size(NextLA.get_factors(tailed, 3, 3)[2]) == (2, 1)
    padded_tail = NextLA.PaddedFTLRMatrix(KernelAbstractions.CPU(), Float64, 9, 10, (4, 4), 2)
    padded_tail.ranks .= NextLA.ranks(tailed)
    for j in 1:3, i in 1:3
        U, V = NextLA.get_factors(tailed, i, j)
        Up, Vp = NextLA.get_factors(padded_tail, i, j)
        copyto!(Up, U); copyto!(Vp, V)
    end
    packed_tail = NextLA.pack_compressed_ftlr(padded_tail)
    @test size(NextLA.get_factors(packed_tail, 3, 3)[1]) == (1, 1)
    @test size(NextLA.get_factors(packed_tail, 3, 3)[2]) == (2, 1)
    @test_throws ArgumentError NextLA.CompressedFTLRMatrix(KernelAbstractions.CPU(), Float64, 9, 8, 4, Int[1 1; 1 1; 1 2])
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
