using LinearAlgebra
using Random

# Shared by the TLR test files, which `runtests.jl` includes after this file.
# `const` so closures in allocation-sensitive tests (`gemm_budget.jl`) resolve
# calls statically rather than boxing a global.
const _TLRM = NextLA.TLRmodule

# ── Tile fixtures ─────────────────────────────────────────────────────────────

function make_rect_lowrank_tile(::Type{T}, m::Int, n::Int, r::Int; seed::Integer) where {T}
    0 <= r <= min(m, n) || throw(ArgumentError("rank must satisfy 0 <= r <= min(m, n)"))
    rng = MersenneTwister(seed)
    if r == 0
        return zeros(T, m, n)
    end
    qleft = Matrix(qr(randn(rng, T, m, r)).Q)
    qright = Matrix(qr(randn(rng, T, n, r)).Q)
    sigma = r == 1 ? [T(1)] : collect(range(T(2), T(1), length=r))
    return qleft[:, 1:r] * Diagonal(sigma) * qright[:, 1:r]'
end

make_lowrank_tile(::Type{T}, b::Int, r::Int; seed::Integer) where {T} =
    make_rect_lowrank_tile(T, b, b, r; seed)

function make_dense_tile(::Type{T}, b::Int; seed::Integer) where {T}
    rng = MersenneTwister(seed)
    tile = randn(rng, T, b, b)
    tile .+= T(0.5) * Matrix{T}(I, b, b)
    return tile
end

function assemble_block_matrix(tile11, tile12, tile21, tile22)
    top = hcat(tile11, tile12)
    bottom = hcat(tile21, tile22)
    return vcat(top, bottom)
end

# 10×10 dense-diagonal fixture tiled 4|4|2: right/bottom/corner boundary tiles are
# all populated and every off-diagonal tile has known rank.
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

# ── TLR reconstruction and storage assertions ─────────────────────────────────

function reconstruct_tlr(A_tlr::NextLA.TLRMatrix)
    T = eltype(A_tlr)
    A = zeros(T, size(A_tlr))
    D = Array(NextLA.dense_diag(A_tlr))
    D_corner = Array(NextLA.dense_diag_corner(A_tlr))

    for linear in 1:prod(NextLA.grid_size(A_tlr))
        tile_i, tile_j = NextLA.TLRmodule.inverse_tile_index(A_tlr.order, NextLA.grid_size(A_tlr)..., linear)
        p0, q0 = NextLA.tile_origin_coords(A_tlr, tile_i, tile_j)
        tile_m, tile_n = NextLA.tile_size(A_tlr, tile_i, tile_j)
        rows = p0:(p0 + tile_m - 1)
        cols = q0:(q0 + tile_n - 1)

        tile = if tile_i == tile_j
            if tile_i <= size(D, 3)
                @view D[1:tile_m, 1:tile_n, tile_i]
            else
                @view D_corner[1:tile_m, 1:tile_n, 1]
            end
        else
            r = Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._rank_index(A_tlr, tile_i, tile_j)])
            U, V = NextLA.get_factors(A_tlr, tile_i, tile_j)
            r == 0 ? zeros(T, tile_m, tile_n) :
                Matrix(U) * Matrix(adjoint(V))
        end

        A[rows, cols] .= tile
    end

    return A
end

function reconstruct_tlr(A_tlr::NextLA.PaddedFTLRMatrix)
    T = eltype(A_tlr)
    A = zeros(T, size(A_tlr))

    for linear in 1:prod(NextLA.grid_size(A_tlr))
        tile_i, tile_j = NextLA.TLRmodule.inverse_tile_index(A_tlr.order, NextLA.grid_size(A_tlr)..., linear)
        p0, q0 = NextLA.tile_origin_coords(A_tlr, tile_i, tile_j)
        tile_m, tile_n = NextLA.tile_size(A_tlr, tile_i, tile_j)
        rows = p0:(p0 + tile_m - 1)
        cols = q0:(q0 + tile_n - 1)

        r = Int(NextLA.ranks(A_tlr)[NextLA.TLRmodule._rank_index(A_tlr, tile_i, tile_j)])
        U, V = NextLA.get_factors(A_tlr, tile_i, tile_j)
        A[rows, cols] .= r == 0 ? zeros(T, tile_m, tile_n) :
            Matrix(U) * Matrix(adjoint(V))
    end

    return A
end

expected_storage_slot(A_tlr::NextLA.AbstractTLRMatrix, i::Int, j::Int) =
    NextLA.TLRmodule._rank_index(A_tlr, i, j)

function assert_tile_rank_and_error(
    A_tlr::NextLA.AbstractTLRMatrix,
    tile_i::Int,
    tile_j::Int,
    expected_rank::Int,
    tile_ref::AbstractMatrix;
    atol_rank::Int=0,
    rtol_error=1f-4,
)
    batch = expected_storage_slot(A_tlr, tile_i, tile_j)
    rank = Int(NextLA.ranks(A_tlr)[batch])
    @test abs(rank - expected_rank) <= atol_rank

    tile_m, tile_n = size(tile_ref)
    U, V = NextLA.get_factors(A_tlr, tile_i, tile_j)
    approx = rank == 0 ? zeros(eltype(tile_ref), tile_m, tile_n) :
        Matrix(U) * Matrix(adjoint(V))
    relerr = norm(tile_ref - approx) / max(norm(tile_ref), eps(real(eltype(tile_ref))))
    @test relerr <= rtol_error
end

# ── Random TLR operands for GEMM tests ────────────────────────────────────────

function fill_random_tlr!(A_tlr::NextLA.TLRMatrix, ArrayType::Type; seed::Integer)
    rng = MersenneTwister(seed)
    T = eltype(A_tlr)
    A_tlr.D .= ArrayType(randn(rng, T, size(A_tlr.D)))
    A_tlr.D_corner .= ArrayType(randn(rng, T, size(A_tlr.D_corner)))
    qm, qn = NextLA.grid_size(A_tlr)
    rank_grid = [i == j ? 0 : min(A_tlr.maxrank, NextLA.tile_size(A_tlr, i, j)...)
                 for i in 1:qm, j in 1:qn]
    packed = NextLA.CompressedFTLRMatrix(
        KernelAbstractions.get_backend(A_tlr), T, size(A_tlr)...,
        NextLA.nominal_tile_size(A_tlr), rank_grid;
        outer_order=NextLA.TileRowMajor, inner_order=NextLA.TileColMajor,
        execution_rank_policy=:exact,
        rank_type=eltype(NextLA.ranks(A_tlr)))
    for j in 1:qn, i in 1:qm
        i == j && continue
        U, V = NextLA.get_factors(packed, i, j)
        U .= ArrayType(randn(rng, T, size(U)))
        V .= ArrayType(randn(rng, T, size(V)))
    end
    A_tlr.offdiag = packed
    return A_tlr
end

function fill_random_tlr!(A_tlr::NextLA.PaddedFTLRMatrix, ArrayType::Type; seed::Integer)
    rng = MersenneTwister(seed)
    T = eltype(A_tlr)
    for f in (A_tlr.int_U, A_tlr.int_V, A_tlr.right_U, A_tlr.right_V,
              A_tlr.bottom_U, A_tlr.bottom_V, A_tlr.corner_U, A_tlr.corner_V)
        length(f) == 0 && continue
        f .= ArrayType(randn(rng, T, size(f)))
    end
    A_tlr.ranks .= A_tlr.maxrank
    return A_tlr
end

# ── Dense-reference GEMM drivers ──────────────────────────────────────────────

# Core driver: `gemm!` on prefilled TLR operands against the tile-by-tile dense
# reference `alpha * A * B + beta * C0`.
function assert_gemm_matches_dense(ArrayType::Type, A_tlr, B_tlr, synchronize;
                                   budget::Int, alpha, beta, atol, rtol)
    T = eltype(A_tlr)
    rng = MersenneTwister(303)
    C0_cpu = randn(rng, T, size(A_tlr, 1), size(B_tlr, 2))
    C = ArrayType(C0_cpu)
    workspace = max(budget, NextLA.gemm_minimum_workspace_bytes(A_tlr, B_tlr))
    NextLA.TLRmodule.gemm!(C, A_tlr, B_tlr; alpha=alpha, beta=beta, workspace)
    synchronize(C)

    C_ref = alpha * reconstruct_tlr(A_tlr) * reconstruct_tlr(B_tlr) + beta * C0_cpu
    @test isapprox(Array(C), C_ref; atol=atol, rtol=rtol)
end

# Dense-diagonal container pair (square, tile size `b`, max rank `r`).
function assert_tlr_gemm_matches_dense(ArrayType::Type, T::Type, n::Int, b::Int, r::Int,
                                       orderA, orderB, synchronize; budget::Int,
                                       alpha=T(1.3), beta=T(-0.4), atol=1e-10, rtol=1e-10)
    A_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, r)
    B_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, r)
    fill_random_tlr!(A_tlr, ArrayType; seed=101)
    fill_random_tlr!(B_tlr, ArrayType; seed=202)
    assert_gemm_matches_dense(ArrayType, A_tlr, B_tlr, synchronize;
                              budget, alpha, beta, atol, rtol)
end

function assert_dense_fulllr_gemm(ArrayType::Type, ::Type{T}, side::Symbol,
                                  m, k, n, tiles, order, transA, transB, synchronize;
                                  budget, atol=1e-9, rtol=1e-9) where {T}
    stored(rows, cols, op) = uppercase(op) == 'T' ? (cols, rows) : (rows, cols)
    optiles(ts, op) = uppercase(op) == 'T' ? reverse(ts) : ts
    opd(X, op) = uppercase(op) == 'T' ? transpose(X) : X
    α, β = T(1.3), T(-0.4)
    C0 = randn(MersenneTwister(303), T, m, n)
    C = ArrayType(C0)

    if side === :tlr_dense
        A = NextLA.PaddedFTLRMatrix(ArrayType(zeros(T, stored(m, k, transA)...)),
                             optiles(tiles, transA), 3; tile_order=order)
        fill_random_tlr!(A, ArrayType; seed=101)
        B = ArrayType(randn(MersenneTwister(202), T, stored(k, n, transB)...))
        workspace = max(budget, 3 * sizeof(T))
        NextLA.TLRmodule.gemm!(C, A, B; alpha=α, beta=β, transA, transB,
                               workspace)
        ref = α * opd(reconstruct_tlr(A), transA) * opd(Array(B), transB) + β * C0
    else
        A = ArrayType(randn(MersenneTwister(101), T, stored(m, k, transA)...))
        B = NextLA.PaddedFTLRMatrix(ArrayType(zeros(T, stored(k, n, transB)...)),
                             optiles(tiles, transB), 3; tile_order=order)
        fill_random_tlr!(B, ArrayType; seed=202)
        workspace = max(budget, 3 * sizeof(T))
        NextLA.TLRmodule.gemm!(C, A, B; alpha=α, beta=β, transA, transB,
                               workspace)
        ref = α * opd(Array(A), transA) * opd(reconstruct_tlr(B), transB) + β * C0
    end
    synchronize(C)
    @test isapprox(Array(C), ref; atol, rtol)
end

function assert_dense_diag_transpose_matches(n, b, r, oA, oB, transA, transB;
                                             budget, alpha=1.3, beta=-0.4,
                                             ArrayType=Array, synchronize=_ -> nothing,
                                             atol=1e-9, rtol=1e-9)
    A = NextLA.TLRMatrix(ArrayType(zeros(Float64, n, n)), b, r)
    B = NextLA.TLRMatrix(ArrayType(zeros(Float64, n, n)), b, r)
    fill_random_tlr!(A, ArrayType; seed=11)
    fill_random_tlr!(B, ArrayType; seed=22)
    C0 = randn(MersenneTwister(33), Float64, n, n)
    C = ArrayType(C0)
    workspace = max(
        budget,
        NextLA.gemm_minimum_workspace_bytes(A, B; transA, transB),
    )
    NextLA.TLRmodule.gemm!(C, A, B; alpha, beta, transA, transB, workspace)
    synchronize(C)
    opd(X, t) = uppercase(t) == 'T' ? transpose(X) : X
    ref = alpha .* (opd(reconstruct_tlr(A), transA) * opd(reconstruct_tlr(B), transB)) .+ beta .* C0
    @test isapprox(Array(C), ref; atol, rtol)
end
