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
    sigma = collect(range(T(2), T(1), length=r))
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

function reconstruct_tlr(A_tlr::NextLA.TLRMatrix)
    T = eltype(A_tlr)
    A = zeros(T, size(A_tlr))
    storage = A_tlr.storage

    for linear in 1:prod(size(A_tlr.layout))
        tile_i, tile_j = NextLA.inverse_tile_index(A_tlr.layout, linear)
        p0, q0 = NextLA.tile_origin_coords(A_tlr.layout, tile_i, tile_j)
        tile_m, tile_n = NextLA.tile_sizes(A_tlr.layout, tile_i, tile_j)
        rows = p0:(p0 + tile_m - 1)
        cols = q0:(q0 + tile_n - 1)

        tile = if !NextLA.compress_diag(A_tlr) && tile_i == tile_j
            @view storage.D[1:tile_m, 1:tile_n, tile_i]
        else
            batch = NextLA.compress_diag(A_tlr) ? linear : NextLA.tile_storage_index(A_tlr, tile_i, tile_j)
            r = Int(NextLA.ranks(A_tlr)[batch])
            r == 0 ? zeros(T, tile_m, tile_n) :
                Matrix(NextLA.tile_u(A_tlr, batch)) *
                Matrix(adjoint(NextLA.tile_v(A_tlr, batch)))
        end

        A[rows, cols] .= tile
    end

    return A
end

expected_storage_slot(A_tlr::NextLA.TLRMatrix, i::Int, j::Int) = NextLA.tile_storage_index(A_tlr, i, j)

function assert_tile_rank_and_error(
    A_tlr::NextLA.TLRMatrix,
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
    approx = rank == 0 ? zeros(eltype(tile_ref), tile_m, tile_n) :
        Matrix(NextLA.tile_u(A_tlr, batch)) *
        Matrix(adjoint(NextLA.tile_v(A_tlr, batch)))
    relerr = norm(tile_ref - approx) / max(norm(tile_ref), eps(real(eltype(tile_ref))))
    @test relerr <= rtol_error
end

function canonical_dense_fixture(::Type{T}; compress_diag::Bool=false) where {T}
    b = 16
    offdiag21 = make_lowrank_tile(T, b, 16; seed=202)

    if compress_diag
        offdiag12 = make_lowrank_tile(T, b, 12; seed=101)
        diag11 = make_lowrank_tile(T, b, 12; seed=303)
        diag22 = make_lowrank_tile(T, b, 4; seed=404)
    else
        offdiag12 = make_lowrank_tile(T, b, 8; seed=101)
        diag11 = make_dense_tile(T, b; seed=303)
        diag22 = make_dense_tile(T, b; seed=404)
    end

    A = assemble_block_matrix(diag11, offdiag12, offdiag21, diag22)
    return (; b, A, diag11, offdiag12, offdiag21, diag22)
end
