"""Synthetic TLR operand generation for the benchmark experiments."""
module ExperimentMatrixGeneration

using LinearAlgebra
using Random
using KernelAbstractions
using NextLA.TLRmodule: PaddedFTLRMatrix, CompressedFTLRMatrix, TileRowMajor,
                        TileColMajor, grid_size, get_factors, tile_size

export generate_tlr_operands, generate_ftlr_operands, generate_tlr_matrix

"""
    generate_tlr_operands(m, k, n, tile_size, ranks, ::Type{T};
                          seed=0, shared_rank=0, backend=CPU()) -> (A, B)

Generate independent synthetic operands for `A[m,k] * B[k,n]`.  Every tile of
`A` has rank `ranks[1]`, every tile of `B` has rank `ranks[2]`, and the factors
are independent Gaussian orthonormal bases by default.  When `shared_rank > 0`,
the first `shared_rank` columns of the `A` outer factors are shared across each
fixed tile row, and the first `shared_rank` columns of the `B` inner factors are
shared across each fixed tile column.  The shared bases remain local to those
families; there is no global basis.

Factor generation is deterministic for a fixed `(seed, m, k, n, tile_size,
ranks, T)`.  Basis construction happens on the CPU, while the returned TLR
factors are stored on `backend`.
"""
function generate_tlr_operands(
    m::Integer,
    k::Integer,
    n::Integer,
    tile_size::Integer,
    ranks::NTuple{2,<:Integer},
    ::Type{T};
    seed::Integer=0,
    shared_rank::Integer=0,
    backend=CPU(),
) where {T}
    return generate_ftlr_operands(m, k, n, tile_size, ranks, T;
        seed, shared_rank, backend, format=:padded)
end

"""
    generate_ftlr_operands(...; format=:padded, rank_distribution=:constant,
                           min_rank, max_rank)

Generate synthetic padded or compressed FTLR operands.  `:constant` uses the
two ranks in `ranks`; `:uniform` samples every tile uniformly in the inclusive
`min_rank:max_rank` interval; `:skewed` uses the same interval with a
low-rank-heavy quadratic distribution.  The rank grid is generated on the CPU
from the supplied seed, independently for A and B.
"""
function generate_ftlr_operands(
    m::Integer, k::Integer, n::Integer, tile_size::Integer,
    ranks::NTuple{2,<:Integer}, ::Type{T};
    seed::Integer=0, shared_rank::Integer=0, backend=CPU(),
    format::Symbol=:padded, rank_distribution::Symbol=:constant,
    min_rank=nothing, max_rank=nothing,
) where {T}
    m, k, n, b = Int.((m, k, n, tile_size))
    rA, rB = Int.(ranks)
    shared = Int(shared_rank)

    m > 0 && k > 0 && n > 0 ||
        throw(ArgumentError("matrix dimensions must be positive"))
    b > 0 || throw(ArgumentError("tile_size must be positive"))
    rA >= 0 && rB >= 0 ||
        throw(ArgumentError("ranks must be nonnegative"))
    0 <= shared <= min(rA, rB) ||
        throw(ArgumentError("shared_rank must satisfy 0 <= shared_rank <= min(ranks)"))
    m % b == 0 && k % b == 0 && n % b == 0 ||
        throw(ArgumentError("the first experiments require dimensions divisible by tile_size"))
    rA <= b && rB <= b ||
        throw(ArgumentError("tile ranks must not exceed tile_size"))
    T <: Number || throw(ArgumentError("dtype must be numeric"))

    rng = MersenneTwister(seed)
    lo = isnothing(min_rank) ? min(rA, rB) : Int(min_rank)
    hi = isnothing(max_rank) ? max(rA, rB) : Int(max_rank)
    lo >= 0 && lo <= hi <= b ||
        throw(ArgumentError("rank interval must satisfy 0 <= min_rank <= max_rank <= tile_size"))
    rank_distribution in (:constant, :uniform, :skewed) ||
        throw(ArgumentError("unknown rank distribution: $rank_distribution"))
    rankA = _rank_grid(cld(m, b), cld(k, b), rA, lo, hi, rank_distribution, rng)
    rankB = _rank_grid(cld(k, b), cld(n, b), rB, lo, hi, rank_distribution, rng)
    A, B = _allocate_pair(backend, T, m, k, n, b, rankA, rankB, format)
    shared <= min(minimum(rankA), minimum(rankB)) ||
        throw(ArgumentError("shared_rank exceeds a generated tile rank"))
    _fill_factors(A, B, rng, shared)
    return A, B
end

function generate_tlr_matrix(
    m::Integer, n::Integer, tile_size::Integer, rank::Integer, ::Type{T};
    seed::Integer=0, backend=CPU(),
) where {T}
    A, unused = generate_tlr_operands(
        m, n, n, tile_size, (rank, rank), T; seed, backend)
    unused = nothing
    return A
end

function _rank_grid(qm, qn, constant_rank, lo, hi, distribution, rng)
    if distribution === :constant
        return fill(Int(constant_rank), qm, qn)
    end
    u = rand(rng, qm, qn)
    mapped = distribution === :uniform ? u : u .^ 2
    ranks = lo .+ floor.(Int, mapped .* (hi - lo + 1))
    return min.(ranks, hi)
end

function _allocate_pair(backend, ::Type{T}, m, k, n, b, rankA, rankB, format) where {T}
    if format === :padded
        A = PaddedFTLRMatrix(backend, T, m, k, b, maximum(rankA); tile_order=TileRowMajor)
        B = PaddedFTLRMatrix(backend, T, k, n, b, maximum(rankB); tile_order=TileRowMajor)
        A.ranks .= vec(permutedims(rankA))
        B.ranks .= vec(permutedims(rankB))
        return A, B
    elseif format === :compressed
        return (CompressedFTLRMatrix(backend, T, m, k, b, rankA;
                    outer_order=TileRowMajor, inner_order=TileColMajor),
                CompressedFTLRMatrix(backend, T, k, n, b, rankB;
                    outer_order=TileRowMajor, inner_order=TileColMajor))
    end
    throw(ArgumentError("format must be :padded or :compressed"))
end

function _fill_factors(
    A, B,
    rng,
    shared_rank::Int,
) where {T}
    mt_A, nt_A = grid_size(A)
    mt_B, nt_B = grid_size(B)

    shared_u = [
        _orthonormal_basis(rng, T, tile_size(A, i, 1)[1], shared_rank)
        for i in 1:mt_A
    ]
    shared_v = [
        _orthonormal_basis(rng, T, tile_size(B, 1, j)[2], shared_rank)
        for j in 1:nt_B
    ]

    for i in 1:mt_A, j in 1:nt_A
        U, V = get_factors(A, i, j)
        tm, tn = tile_size(A, i, j)
        U .= _family_basis(rng, T, tm, size(U, 2), shared_u[i])
        V .= _orthonormal_basis(rng, T, tn, size(V, 2))
    end
    for i in 1:mt_B, j in 1:nt_B
        U, V = get_factors(B, i, j)
        tm, tn = tile_size(B, i, j)
        U .= _orthonormal_basis(rng, T, tm, size(U, 2))
        V .= _family_basis(rng, T, tn, size(V, 2), shared_v[j])
    end
    return nothing
end

function _orthonormal_basis(rng, ::Type{T}, dimension::Int, rank::Int) where {T}
    rank == 0 && return Matrix{T}(undef, dimension, 0)
    return Matrix(qr(randn(rng, T, dimension, rank)).Q)[:, 1:rank]
end

function _family_basis(
    rng,
    ::Type{T},
    dimension::Int,
    rank::Int,
    shared::AbstractMatrix,
) where {T}
    private_rank = rank - size(shared, 2)
    private_rank == 0 && return shared

    G = randn(rng, T, dimension, private_rank)
    isempty(shared) || (G .-= shared * (adjoint(shared) * G))
    private = Matrix(qr(G).Q)[:, 1:private_rank]
    return hcat(shared, private)
end

end
