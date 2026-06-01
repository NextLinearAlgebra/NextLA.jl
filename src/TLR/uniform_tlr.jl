struct TLRMatrix{
    T,
    RankT<:Integer,
    AUVStore<:AbstractArray{T,3},
    RanksT<:AbstractMatrix{RankT},
    DiagT<:AbstractArray{T,3},
    O<:AbstractTileOrder,
}
    AUV::TileFactorBuffer{T,AUVStore,O}
    ranks::RanksT
    diag::DiagT
    m::Int
    n::Int
    b::Int
    mt::Int
    nt::Int
    maxrank::Int
    compress_diag::Bool
end

tile_linear_index(A::TLRMatrix, i::Integer, j::Integer) = tile_linear_index(A.AUV.order, i, j)

Base.eltype(::Type{<:TLRMatrix{T}}) where {T} = T
Base.eltype(A::TLRMatrix{T}) where {T} = T
Base.size(A::TLRMatrix) = (A.m, A.n)
Base.size(A::TLRMatrix, d::Integer) = size(A)[d]

function TLRMatrix(
    prototype::AbstractArray{T},
    m::Integer,
    n::Integer;
    blocksize::Integer,
    maxrank::Integer,
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:AbstractTileOrder}=TileColMajor,
) where {T}
    m_int = Int(m)
    n_int = Int(n)
    b_int = Int(blocksize)
    maxrank_int = Int(maxrank)

    m_int > 0 || throw(ArgumentError("m must be positive"))
    n_int > 0 || throw(ArgumentError("n must be positive"))
    b_int > 0 || throw(ArgumentError("blocksize must be positive"))
    maxrank_int >= 0 || throw(ArgumentError("maxrank must be nonnegative"))

    mt = cld(m_int, b_int)
    nt = cld(n_int, b_int)
    ndiag = compress_diag ? 0 : min(mt, nt)
    order = tile_order(mt, nt)

    auv = similar(prototype, T, 2 * b_int, maxrank_int, mt * nt)
    ranks = similar(prototype, rank_type, mt, nt)
    diag = similar(prototype, T, b_int, b_int, ndiag)

    fill!(ranks, zero(rank_type))

    return TLRMatrix(
        TileFactorBuffer(auv, order),
        ranks,
        diag,
        m_int,
        n_int,
        b_int,
        mt,
        nt,
        maxrank_int,
        compress_diag,
    )
end

function TLRMatrix(
    A::AbstractMatrix{T};
    blocksize::Integer,
    maxrank::Integer,
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:AbstractTileOrder}=TileColMajor,
) where {T}
    return TLRMatrix(
        A,
        size(A, 1),
        size(A, 2);
        blocksize=blocksize,
        maxrank=maxrank,
        compress_diag=compress_diag,
        rank_type=rank_type,
        tile_order=tile_order,
    )
end

function similar_tlr(
    A::AbstractMatrix{T};
    blocksize::Integer,
    maxrank::Integer,
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:AbstractTileOrder}=TileColMajor,
) where {T}
    return TLRMatrix(
        A;
        blocksize=blocksize,
        maxrank=maxrank,
        compress_diag=compress_diag,
        rank_type=rank_type,
        tile_order=tile_order,
    )
end
