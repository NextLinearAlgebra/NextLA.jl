"""
Fixed-width factors produced by ARA for one homogeneous tile-shape batch.

`LowRankFactorBatch` is numerical staging, not a matrix representation.  Its
`U`/`V` panels retain `maxrank` columns until every batch has discovered its
logical ranks; finalization then scatters only the active columns into the two
complementarily ordered packed factor vectors of a `CompressedFTLRMatrix`.
"""
struct LowRankFactorBatch{IDs,RI,U,V,R,E}
    tile_shape::NTuple{2,Int}
    tile_ids::IDs
    rank_indices::RI
    U::U
    V::V
    ranks::R
    errors_sq::E
end

"""Reusable ARA scratch attached to one `LowRankFactorBatch`."""
struct CompressCategoryWorkspace{F,ZT,TileVT,I32V,ARAT,QTileVT,OmegaT,
                                 OmegaTileVT,YTileVT}
    factors::F
    S::Int
    R_keep::Int
    Z::ZT
    V_tiles::TileVT
    p0s::I32V
    q0s::I32V
    ara::ARAT
    Q_tiles::QTileVT
    omega::OmegaT
    omega_tiles::OmegaTileVT
    Y_tiles::YTileVT
end

"""
Reusable dense-to-FTLR compression storage.

There is one category per distinct logical tile shape. `diagonal=:compressed`
includes every tile; `diagonal=:dense` excludes diagonal tiles so the same
workspace builds the compressed off-diagonal part of a `TLRMatrix`.
"""
struct FTLRCompressionWorkspace{CatsT,StreamV,KeyT}
    cats::CatsT
    streams::StreamV
    key::KeyT
end

@inline compress_ara_block(S::Int) = max(min(32, S), 1)

function _make_category_workspace(tile_shape::NTuple{2,Int}, tile_ids,
                                  rank_indices, U::AbstractArray{T,3}, V;
                                  rank_type::Type=Int32,
                                  block::Int=compress_ara_block(
                                      min(size(U, 2), tile_shape...)),
                                  p0s=nothing, q0s=nothing) where {T}
    tm, tn = tile_shape
    kout = size(U, 2)
    ntiles = size(U, 3)
    size(U) == (tm, kout, ntiles) || throw(DimensionMismatch("invalid U batch shape"))
    size(V) == (tn, kout, ntiles) || throw(DimensionMismatch("invalid V batch shape"))
    length(tile_ids) == ntiles || throw(DimensionMismatch("tile_ids must match batch size"))
    length(rank_indices) == ntiles || throw(DimensionMismatch("rank_indices must match batch size"))
    backend = get_backend(U)
    S = min(kout, tm, tn)
    blk = S == 0 ? 0 : max(min(block, S), 1)
    Z = view(V, :, 1:S, :)
    Q = zeros(backend, T, tm, S, ntiles)
    ara = S == 0 ? nothing : ARAWorkspace(Q; block=blk)
    omega = zeros(backend, T, tn, blk, ntiles)
    empty_i32 = allocate(backend, Int32, 0)
    pdev = p0s === nothing ? empty_i32 : p0s
    qdev = q0s === nothing ? empty_i32 : q0s
    Y = S == 0 ? zeros(backend, T, tm, 0, ntiles) : ara.Yblk
    factors = LowRankFactorBatch(
        tile_shape, tile_ids, rank_indices, U, V,
        zeros(backend, rank_type, ntiles), zeros(backend, Float64, ntiles))
    return CompressCategoryWorkspace(
        factors, S, min(kout, S), Z, _batch_views(Z, S), pdev, qdev,
        ara, _batch_views(Q, S), omega, _batch_views(omega, blk),
        _batch_views(Y, blk))
end

function _compression_batch_specs(m::Int, n::Int,
                                  tile_size::NTuple{2,Int}, diagonal::Symbol)
    diagonal in (:compressed, :dense) || throw(ArgumentError(
        "diagonal must be :compressed or :dense"))
    bm, bn = tile_size
    qm, qn = cld(m, bm), cld(n, bn)
    shapes = NTuple{2,Int}[]
    ids_by_shape = Dict{NTuple{2,Int},Vector{NTuple{2,Int}}}()
    @inbounds for j in 1:qn, i in 1:qm
        diagonal === :dense && i == j && continue
        tm = min(bm, m - (i - 1) * bm)
        tn = min(bn, n - (j - 1) * bn)
        shape = (tm, tn)
        if !haskey(ids_by_shape, shape)
            ids_by_shape[shape] = NTuple{2,Int}[]
            push!(shapes, shape)
        end
        push!(ids_by_shape[shape], (i, j))
    end
    return [(shape=shape, tile_ids=ids_by_shape[shape]) for shape in shapes]
end

function FTLRCompressionWorkspace(A::AbstractMatrix{T},
                                  tile_size::NTuple{2,Int};
                                  maxrank::Int,
                                  diagonal::Symbol=:compressed,
                                  rank_type::Type{<:Integer}=Int32,
                                  block::Int=compress_ara_block(maxrank)) where {T}
    m, n = size(A)
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 && maxrank >= 0 || throw(ArgumentError(
        "matrix and tile dimensions must be positive and maxrank nonnegative"))
    backend = get_backend(A)
    qm, qn = cld(m, bm), cld(n, bn)
    specs = _compression_batch_specs(m, n, tile_size, diagonal)
    cats = map(specs) do spec
        ids = spec.tile_ids
        count = length(ids)
        tm, tn = spec.shape
        p0_host = Int32[(i - 1) * bm + 1 for (i, _) in ids]
        q0_host = Int32[(j - 1) * bn + 1 for (_, j) in ids]
        p0s = copyto!(allocate(backend, Int32, count), p0_host)
        q0s = copyto!(allocate(backend, Int32, count), q0_host)
        rank_indices = [tile_linear_index(TileRowMajor(), qm, qn, i, j)
                        for (i, j) in ids]
        U = zeros(backend, T, tm, maxrank, count)
        V = zeros(backend, T, tn, maxrank, count)
        _make_category_workspace(
            spec.shape, ids, rank_indices, U, V;
            rank_type, block, p0s, q0s)
    end
    key = (; backend=typeof(backend), device=backend, T, m, n, tile_size, maxrank,
           diagonal, rank_type, block)
    return FTLRCompressionWorkspace(cats, create_streams(backend, length(cats)), key)
end

FTLRCompressionWorkspace(A::AbstractMatrix, b::Int; kwargs...) =
    FTLRCompressionWorkspace(A, (b, b); kwargs...)

function _validate_compression_workspace(ws::FTLRCompressionWorkspace,
                                         A::AbstractMatrix,
                                         tile_size::NTuple{2,Int},
                                         maxrank::Int,
                                         diagonal::Symbol,
                                         rank_type::Type{<:Integer})
    key = ws.key
    expected = (typeof(get_backend(A)), eltype(A), size(A, 1), size(A, 2),
                tile_size, maxrank, diagonal, rank_type)
    actual = (key.backend, key.T, key.m, key.n, key.tile_size, key.maxrank,
              key.diagonal, key.rank_type)
    actual == expected || throw(ArgumentError(
        "FTLRCompressionWorkspace does not match source, geometry, maxrank, " *
        "diagonal policy, or rank type"))
    return ws
end

"""Standalone ARA workspace around caller-owned homogeneous factor panels."""
function carve_tile_workspace(U::AbstractArray{T,3}, V,
                              tm::Int, tn::Int, kout::Int, ntiles::Int;
                              rank_type::Type=Int32,
                              block::Int=compress_ara_block(
                                  min(kout, tm, tn))) where {T}
    size(U) == (tm, kout, ntiles) || throw(DimensionMismatch("invalid U shape"))
    size(V) == (tn, kout, ntiles) || throw(DimensionMismatch("invalid V shape"))
    ids = [(k, 1) for k in 1:ntiles]
    return _make_category_workspace(
        (tm, tn), ids, collect(1:ntiles), U, V; rank_type, block)
end

alloc_tile_workspace(U::AbstractArray{T,3}, V, tm::Int, tn::Int,
                     kout::Int, ntiles::Int; kwargs...) where {T} =
    carve_tile_workspace(U, V, tm, tn, kout, ntiles; kwargs...)
