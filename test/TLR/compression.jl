include("helpers.jl")

function _check_packed_tiles(layout, packed, diag11, offdiag12, offdiag21, diag22)
    @test Array(packed.diag[:, :, 1]) ≈ diag11
    @test Array(packed.diag[:, :, 2]) ≈ diag22
    @test Array(packed.offdiag[:, :, NextLA.TLRmodule.offdiag_batch_index(layout, 1, 2)]) ≈ offdiag12
    @test Array(packed.offdiag[:, :, NextLA.TLRmodule.offdiag_batch_index(layout, 2, 1)]) ≈ offdiag21
end

@testset "TLR dense compression" begin
    fixture = canonical_dense_fixture(Float32; compress_diag=false)
    b = fixture.b
    max_rank = 16
    eps_tol = 1f-4

    for (backend_name, AT, sync) in backends
        @testset "$backend_name" begin
            A_dev = _to_backend(AT, fixture.A)
            layout = NextLA.TileMap(NextLA.TileColMajor(2, 2), b, b, size(fixture.A, 1), size(fixture.A, 2))

            packed = NextLA.PackedDenseTileStorage(
                _to_backend(AT, zeros(Float32, b, b, 2)),
                _to_backend(AT, zeros(Float32, b, b, 2)),
                layout,
            )
            NextLA.to_tiles!(packed, A_dev)
            sync(packed.offdiag)
            sync(packed.diag)
            _check_packed_tiles(layout, packed, fixture.diag11, fixture.offdiag12, fixture.offdiag21, fixture.diag22)

            A_tlr = NextLA.TLRMatrix(A_dev, b, max_rank; compress_diag=false)
            Random.seed!(0x1234)
            NextLA.compress!(A_tlr, A_dev; block_size=8, eps=eps_tol)

            sync(A_tlr.storage.U)
            sync(A_tlr.storage.V)
            sync(A_tlr.storage.D)

            @test Array(A_tlr.storage.D[:, :, 1]) ≈ fixture.diag11
            @test Array(A_tlr.storage.D[:, :, 2]) ≈ fixture.diag22
            assert_tile_rank_and_error(A_tlr, 1, 2, 8, fixture.offdiag12; rtol_error=5f-4)
            assert_tile_rank_and_error(A_tlr, 2, 1, 16, fixture.offdiag21; rtol_error=5f-4)

            A_reconstructed = reconstruct_tlr(A_tlr)
                @test norm(A_reconstructed - fixture.A) / max(norm(fixture.A), eps(Float32)) <= 5f-4
        end
    end

    @testset "compress_diag=true" begin
        b = 16
        fixture_all = (
            b=b,
            diag11=make_lowrank_tile(Float32, b, 4; seed=501),
            offdiag12=make_lowrank_tile(Float32, b, 4; seed=502),
            offdiag21=make_lowrank_tile(Float32, b, 4; seed=503),
            diag22=make_lowrank_tile(Float32, b, 4; seed=504),
        )
        fixture_all = merge(fixture_all, (A=assemble_block_matrix(
            fixture_all.diag11,
            fixture_all.offdiag12,
            fixture_all.offdiag21,
            fixture_all.diag22,
        ),))

        for (backend_name, AT, sync) in backends
            @testset "$backend_name" begin
                A_dev = _to_backend(AT, fixture_all.A)
                A_tlr = NextLA.TLRMatrix(A_dev, fixture_all.b, 16; compress_diag=true)
                Random.seed!(0x2345)
                NextLA.compress!(A_tlr, A_dev; block_size=4, eps=1f-4)

                sync(A_tlr.storage.U)
                sync(A_tlr.storage.V)

                assert_tile_rank_and_error(A_tlr, 1, 1, 4, fixture_all.diag11; rtol_error=5f-4)
                assert_tile_rank_and_error(A_tlr, 1, 2, 4, fixture_all.offdiag12; rtol_error=5f-4)
                assert_tile_rank_and_error(A_tlr, 2, 1, 4, fixture_all.offdiag21; rtol_error=5f-4)
                assert_tile_rank_and_error(A_tlr, 2, 2, 4, fixture_all.diag22; rtol_error=5f-4)

                A_reconstructed = reconstruct_tlr(A_tlr)
                @test norm(A_reconstructed - fixture_all.A) / max(norm(fixture_all.A), eps(Float32)) <= 5f-4
            end
        end
    end
end

@testset "TLR operator compression" begin
    fixture = canonical_dense_fixture(Float32; compress_diag=false)
    A = fixture.A
    op = NextLA.TLRLinearOperator(
        Float32,
        size(A, 1),
        size(A, 2),
        (Y, X) -> mul!(Y, A, X),
        (Y, X) -> mul!(Y, adjoint(A), X),
    )

    for tile_order in (NextLA.TileColMajor, NextLA.TileRowMajor)
        @testset "tile_order=$(tile_order)" begin
            A_dense = NextLA.TLRMatrix(A, fixture.b, 16; compress_diag=false, tile_order)
            A_op = NextLA.TLRMatrix(A, fixture.b, 16; compress_diag=false, tile_order)

            Random.seed!(0x3456)
            NextLA.compress!(A_dense, A; block_size=8, eps=1f-4)
            Random.seed!(0x3456)
            NextLA.compress!(A_op, op; block_size=8, eps=1f-4)

            @test Array(A_op.storage.D[:, :, 1]) ≈ fixture.diag11
            @test Array(A_op.storage.D[:, :, 2]) ≈ fixture.diag22
            @test Array(A_dense.storage.ranks) == Array(A_op.storage.ranks)

            assert_tile_rank_and_error(A_op, 1, 2, 8, fixture.offdiag12; rtol_error=5f-4)
            assert_tile_rank_and_error(A_op, 2, 1, 16, fixture.offdiag21; rtol_error=5f-4)

            A_reconstructed = reconstruct_tlr(A_op)
            @test norm(A_reconstructed - A) / max(norm(A), eps(Float32)) <= 5f-4
        end
    end

    @testset "operator validation" begin
        A_tlr = NextLA.TLRMatrix(zeros(Float32, 32, 32), 16, 16)

        wrong_size = NextLA.TLRLinearOperator(
            Float32, 16, 16,
            (Y, X) -> fill!(Y, 0f0),
            (Y, X) -> fill!(Y, 0f0),
        )
        @test_throws DimensionMismatch NextLA.compress!(A_tlr, wrong_size; block_size=8, eps=1f-4)

        wrong_type = NextLA.TLRLinearOperator(
            Float64, 32, 32,
            (Y, X) -> fill!(Y, 0.0),
            (Y, X) -> fill!(Y, 0.0),
        )
        @test_throws ArgumentError NextLA.compress!(A_tlr, wrong_type; block_size=8, eps=1f-4)

        struct NoAdjointOp{T}
            m::Int
            n::Int
        end
        Base.size(op::NoAdjointOp) = (op.m, op.n)
        Base.eltype(::NoAdjointOp{T}) where {T} = T
        function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::NoAdjointOp{T}, X::AbstractVecOrMat) where {T}
            fill!(Y, zero(T))
            return Y
        end
        @test_throws ArgumentError NextLA.compress!(A_tlr, NoAdjointOp{Float32}(32, 32); block_size=8, eps=1f-4)
    end
end
