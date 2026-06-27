function make_rect_lowrank_tile(::Type{T}, m::Int, n::Int, r::Int; seed::Integer) where {T}
    0 <= r <= min(m, n) || throw(ArgumentError("rank must satisfy 0 <= r <= min(m, n)"))
    rng = MersenneTwister(seed)
    if r == 0
        return zeros(T, m, n)
    end
    qleft = Matrix(qr(randn(rng, T, m, r)).Q)
    qright = Matrix(qr(randn(rng, T, n, r)).Q)
    sigma = collect(range(T(2), T(1), length=r))
    return qleft[:, 1:r] * Diagonal(sigma) * qright[:, 1:r]'
end

function boundary_dense_fixture(::Type{T}) where {T}
    d11 = make_dense_tile(T, 4; seed=11)
    d22 = make_dense_tile(T, 4; seed=22)
    d33 = make_dense_tile(T, 2; seed=33)

    a12 = make_lowrank_tile(T, 4, 2; seed=101)
    a21 = make_lowrank_tile(T, 4, 3; seed=102)
    a13 = make_rect_lowrank_tile(T, 4, 2, 2; seed=103)
    a23 = make_rect_lowrank_tile(T, 4, 2, 2; seed=104)
    a31 = make_rect_lowrank_tile(T, 2, 4, 2; seed=105)
    a32 = make_rect_lowrank_tile(T, 2, 4, 2; seed=106)

    top = hcat(d11, a12, a13)
    mid = hcat(a21, d22, a23)
    bot = hcat(a31, a32, d33)
    A = vcat(top, mid, bot)
    return (; A, d11, d22, d33, a12, a21, a13, a23, a31, a32)
end

@testset "TLR compress! on CPU" begin
    fixture = canonical_dense_fixture(Float64)
    A_uniform = NextLA.TLRMatrix(fixture.A, fixture.b, 16)
    ws_uniform = NextLA.TLRmodule.alloc_workspace(A_uniform)
    NextLA.compress!(A_uniform, fixture.A, ws_uniform; tol=1e-6, alg=:cholqr2)

    relerr = norm(reconstruct_tlr(A_uniform) - fixture.A) / norm(fixture.A)
    @test relerr <= 1e-6
    assert_tile_rank_and_error(A_uniform, 1, 2, 10, fixture.offdiag12; atol_rank=4, rtol_error=1e-6)
    assert_tile_rank_and_error(A_uniform, 2, 1, 16, fixture.offdiag21; rtol_error=1e-6)

    boundary = boundary_dense_fixture(Float64)
    A_panel = NextLA.TLRMatrix(boundary.A, 4, 3)
    ws_panel = NextLA.TLRmodule.alloc_workspace(A_panel)
    NextLA.compress!(A_panel, boundary.A, ws_panel; tol=1e-6, alg=:cholqr2)

    relerr_panel = norm(reconstruct_tlr(A_panel) - boundary.A) / norm(boundary.A)
    @test relerr_panel <= 1e-6
    @test size(A_panel.int_U) == (4, 3, 2)
    @test size(A_panel.right_U) == (4, 3, 2)
    @test size(A_panel.bottom_U) == (2, 3, 2)
    assert_tile_rank_and_error(A_panel, 1, 2, 2, boundary.a12; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 2, 1, 3, boundary.a21; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 1, 3, 2, boundary.a13; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 2, 3, 2, boundary.a23; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 3, 1, 2, boundary.a31; atol_rank=1, rtol_error=1e-6)
    assert_tile_rank_and_error(A_panel, 3, 2, 2, boundary.a32; atol_rank=1, rtol_error=1e-6)
end

@testset "TLR compress! on GPU" begin
    boundary = boundary_dense_fixture(Float32)
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name in ("CUDA", "AMDGPU") || continue
        @testset "$backend_name" begin
            dense = ArrayType(boundary.A)
            A_tlr = NextLA.TLRMatrix(dense, 4, 3)
            ws = NextLA.TLRmodule.alloc_workspace(A_tlr)
            NextLA.compress!(A_tlr, dense, ws; tol=1f-4, alg=:cholqr2)

            synchronize(A_tlr.int_U)
            relerr = norm(reconstruct_tlr(A_tlr) - boundary.A) / norm(boundary.A)
            @test relerr <= 5f-3
            @test Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._offdiag_index(A_tlr, 1, 3)]) == 2
            @test Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._offdiag_index(A_tlr, 3, 1)]) == 2
        end
    end
end
