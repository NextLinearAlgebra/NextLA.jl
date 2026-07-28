"""Synthetic TLR operand generation for the benchmark experiments."""
module ExperimentMatrixGeneration

using LinearAlgebra
using Random
using KernelAbstractions
using NextLA.TLRmodule: PaddedFTLRMatrix, TileRowMajor, grid_size, maxrank,
                        get_factors, tile_size

export generate_tlr_operands, generate_tlr_matrix

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

    A = PaddedFTLRMatrix(backend, T, m, k, b, rA; tile_order=TileRowMajor)
    B = PaddedFTLRMatrix(backend, T, k, n, b, rB; tile_order=TileRowMajor)
    A.ranks .= rA
    B.ranks .= rB

    rng = MersenneTwister(seed)
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

function _fill_factors(
    A::PaddedFTLRMatrix{<:Any,T},
    B::PaddedFTLRMatrix{<:Any,T},
    rng,
    shared_rank::Int,
) where {T}
    mt_A, nt_A = grid_size(A)
    mt_B, nt_B = grid_size(B)
    rA, rB = maxrank(A), maxrank(B)

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
        U .= _family_basis(rng, T, tm, rA, shared_u[i])
        V .= _orthonormal_basis(rng, T, tn, rA)
    end
    for i in 1:mt_B, j in 1:nt_B
        U, V = get_factors(B, i, j)
        tm, tn = tile_size(B, i, j)
        U .= _orthonormal_basis(rng, T, tm, rB)
        V .= _family_basis(rng, T, tn, rB, shared_v[j])
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
