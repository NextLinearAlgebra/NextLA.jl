using LinearAlgebra
using Random

function make_lowrank_tile(::Type{T}, b::Int, r::Int; seed::Integer) where {T}
    0 <= r <= b || throw(ArgumentError("rank must satisfy 0 <= r <= b"))
    rng = MersenneTwister(seed)
    if r == 0
        return zeros(T, b, b)
    end
    qleft = Matrix(qr(randn(rng, T, b, r)).Q)
    qright = Matrix(qr(randn(rng, T, b, r)).Q)
    sigma = r == 1 ? [T(1)] : collect(range(T(2), T(1), length=r))
    return qleft[:, 1:r] * Diagonal(sigma) * qright[:, 1:r]'
end

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

function reconstruct_tlr(A_tlr::NextLA.TLRDenseDiagMatrix)
    T = eltype(A_tlr)
    A = zeros(T, size(A_tlr))
    D = Array(NextLA.dense_diag(A_tlr))
    D_corner = Array(NextLA.dense_diag_corner(A_tlr))

    for linear in 1:prod(NextLA.tilegrid_size(A_tlr))
        tile_i, tile_j = NextLA.TLRmodule.inverse_tile_index(A_tlr.order, NextLA.tilegrid_size(A_tlr)..., linear)
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

function reconstruct_tlr(A_tlr::NextLA.TLRMatrix)
    T = eltype(A_tlr)
    A = zeros(T, size(A_tlr))

    for linear in 1:prod(NextLA.tilegrid_size(A_tlr))
        tile_i, tile_j = NextLA.TLRmodule.inverse_tile_index(A_tlr.order, NextLA.tilegrid_size(A_tlr)..., linear)
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

function canonical_dense_fixture(::Type{T}) where {T}
    b = 16
    offdiag12 = make_lowrank_tile(T, b, 8; seed=101)
    offdiag21 = make_lowrank_tile(T, b, 16; seed=202)
    diag11 = make_dense_tile(T, b; seed=303)
    diag22 = make_dense_tile(T, b; seed=404)

    A = assemble_block_matrix(diag11, offdiag12, offdiag21, diag22)
    return (; b, A, diag11, offdiag12, offdiag21, diag22)
end
