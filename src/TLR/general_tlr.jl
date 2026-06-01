struct GeneralTLRMatrix{
    T,
    OffsetT<:Integer,
    RankT<:Integer,
    UVT<:AbstractVector{T},
    DiagT<:AbstractVector{T},
    OffsetsT<:AbstractMatrix{OffsetT},
    DiagOffsetsT<:AbstractVector{OffsetT},
    RanksT<:AbstractMatrix{RankT},
    SizesT<:AbstractVector{OffsetT},
    PtrsT<:AbstractVector{OffsetT},
    O<:AbstractTileOrder,
}
    UV::UVT
    diag::DiagT
    offsets::OffsetsT
    diagoffsets::DiagOffsetsT
    ranks::RanksT
    rowsizes::SizesT
    colsizes::SizesT
    rowptr::PtrsT
    colptr::PtrsT
    m::Int
    n::Int
    mt::Int
    nt::Int
    maxrank::Int
    compress_diag::Bool
    order::O
end

Base.eltype(::Type{<:GeneralTLRMatrix{T}}) where {T} = T
Base.eltype(A::GeneralTLRMatrix{T}) where {T} = T
Base.size(A::GeneralTLRMatrix) = (A.m, A.n)
Base.size(A::GeneralTLRMatrix, d::Integer) = size(A)[d]

function _block_ptrs(sizes::AbstractVector{<:Integer}, ::Type{OffsetT}) where {OffsetT<:Integer}
    ptr = Vector{OffsetT}(undef, length(sizes) + 1)
    ptr[1] = one(OffsetT)
    @inbounds for i in eachindex(sizes)
        ptr[i + 1] = ptr[i] + OffsetT(sizes[i])
    end
    return ptr
end

function _fill_tile_offsets!(
    offsets::AbstractMatrix{OffsetT},
    rowsizes::AbstractVector{<:Integer},
    colsizes::AbstractVector{<:Integer},
    maxrank::Int,
    order::TileColMajor,
) where {OffsetT<:Integer}
    offset = one(OffsetT)
    @inbounds for j in 1:order.nt
        for i in 1:order.mt
            offsets[i, j] = offset
            offset += OffsetT((rowsizes[i] + colsizes[j]) * maxrank)
        end
    end
    return offset
end

function _fill_tile_offsets!(
    offsets::AbstractMatrix{OffsetT},
    rowsizes::AbstractVector{<:Integer},
    colsizes::AbstractVector{<:Integer},
    maxrank::Int,
    order::TileRowMajor,
) where {OffsetT<:Integer}
    offset = one(OffsetT)
    @inbounds for i in 1:order.mt
        for j in 1:order.nt
            offsets[i, j] = offset
            offset += OffsetT((rowsizes[i] + colsizes[j]) * maxrank)
        end
    end
    return offset
end

function GeneralTLRMatrix(
    prototype::AbstractArray{T},
    rowsizes::AbstractVector{<:Integer},
    colsizes::AbstractVector{<:Integer};
    maxrank::Integer,
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    offset_type::Type{<:Integer}=Int64,
    tile_order::Type{<:AbstractTileOrder}=TileColMajor,
) where {T}
    isempty(rowsizes) && throw(ArgumentError("rowsizes must be nonempty"))
    isempty(colsizes) && throw(ArgumentError("colsizes must be nonempty"))
    all(>(0), rowsizes) || throw(ArgumentError("rowsizes must be positive"))
    all(>(0), colsizes) || throw(ArgumentError("colsizes must be positive"))

    maxrank_int = Int(maxrank)
    maxrank_int >= 0 || throw(ArgumentError("maxrank must be nonnegative"))

    mt = length(rowsizes)
    nt = length(colsizes)
    ndiag = min(mt, nt)

    @inbounds for k in 1:ndiag
        rowsizes[k] == colsizes[k] || throw(ArgumentError("diagonal tiles must be square"))
    end

    m_int = sum(Int, rowsizes)
    n_int = sum(Int, colsizes)
    order = tile_order(mt, nt)

    host_rowsizes = offset_type.(rowsizes)
    host_colsizes = offset_type.(colsizes)
    host_rowptr = _block_ptrs(rowsizes, offset_type)
    host_colptr = _block_ptrs(colsizes, offset_type)

    offsets_host = Matrix{offset_type}(undef, mt, nt)
    diagoffsets_host = Vector{offset_type}(undef, ndiag)

    uv_offset = _fill_tile_offsets!(offsets_host, rowsizes, colsizes, maxrank_int, order)
    diag_offset = one(offset_type)

    if !compress_diag
        @inbounds for k in 1:ndiag
            diagoffsets_host[k] = diag_offset
            diag_offset += offset_type(rowsizes[k] * colsizes[k])
        end
    else
        fill!(diagoffsets_host, zero(offset_type))
        diag_offset = one(offset_type)
    end

    uv_length = Int(uv_offset - one(offset_type))
    diag_length = compress_diag ? 0 : Int(diag_offset - one(offset_type))

    uv = similar(prototype, T, uv_length)
    diag = similar(prototype, T, diag_length)
    offsets = similar(prototype, offset_type, mt, nt)
    diagoffsets = similar(prototype, offset_type, ndiag)
    ranks = similar(prototype, rank_type, mt, nt)
    rows = similar(prototype, offset_type, mt)
    cols = similar(prototype, offset_type, nt)
    rowptr = similar(prototype, offset_type, mt + 1)
    colptr = similar(prototype, offset_type, nt + 1)

    copyto!(offsets, offsets_host)
    copyto!(diagoffsets, diagoffsets_host)
    copyto!(rows, host_rowsizes)
    copyto!(cols, host_colsizes)
    copyto!(rowptr, host_rowptr)
    copyto!(colptr, host_colptr)
    fill!(ranks, zero(rank_type))

    return GeneralTLRMatrix(
        uv,
        diag,
        offsets,
        diagoffsets,
        ranks,
        rows,
        cols,
        rowptr,
        colptr,
        m_int,
        n_int,
        mt,
        nt,
        maxrank_int,
        compress_diag,
        order,
    )
end

tile_linear_index(A::GeneralTLRMatrix, i::Integer, j::Integer) = tile_linear_index(A.order, i, j)
