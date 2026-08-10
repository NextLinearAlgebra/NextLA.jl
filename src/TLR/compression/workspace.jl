"""
Reusable workspace for one homogeneous tile category.

Only data live on the current ARA compression path is retained:

* `ara.Q` is the growing basis;
* `ara` owns block projection/Cholesky-QR scratch;
* `omega` is the fresh sketch block;
* `Z` aliases the first `S` columns of output `V` for the co-range;
* ranks and exact tile energies/errors are persistent diagnostics.

The old one-shot compressor's full-width `Y_hi`, `G_hi`, `R_tiles`,
`shift_mult`, and `Q_T` aliases were removed. They duplicated ARA's block-width
scratch without a production caller.
"""
struct CompressCategoryWorkspace{RegionT,PanelT,ZT,TileVT,RankVT,ErrV,I32V,
                                 RankIndexT,ARAT,QTileVT,OmegaT,OmegaTileVT,YTileVT}
    region::RegionT
    rank_indices::RankIndexT
    S::Int
    R_keep::Int
    U::PanelT
    V::PanelT
    Z::ZT
    V_tiles::TileVT
    ranks_local::RankVT
    norm_err_sq::ErrV
    p0s::I32V
    q0s::I32V
    ara::ARAT
    Q_tiles::QTileVT
    omega::OmegaT
    omega_tiles::OmegaTileVT
    Y_tiles::YTileVT
end

struct CompressWorkspace{CatsT,StreamV}
    cats::CatsT
    streams::StreamV
end

"""Reusable padded sampling storage used to discover exact TLR off-diagonal ranks."""
struct TLRCompressWorkspace{ScratchT,WorkspaceT}
    scratch::ScratchT
    workspace::WorkspaceT
end

@inline function _region_specs(A_tlr::PaddedFTLRMatrix)
    map(lowrank_regions(A_tlr)) do region
        U = outer_factors(A_tlr, region)
        V = inner_factors(A_tlr, region)
        (; region, n=region_tile_count(A_tlr, region), U, V,
            tm=size(U, 1), tn=size(V, 1))
    end
end

"""
    compress_ara_block(S) -> Int

Default sampling block width. It is a performance knob only and is clamped to
the available sketch capacity.
"""
@inline compress_ara_block(S::Int) = max(min(32, S), 1)

function _make_category_workspace(region, rank_indices,
                                  U::AbstractArray{T,3}, V,
                                  tm::Int, tn::Int, kout::Int, ntiles::Int;
                                  rank_type::Type=Int32,
                                  block::Int=compress_ara_block(min(kout, tm, tn)),
                                  p0s=nothing, q0s=nothing) where {T}
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
    return CompressCategoryWorkspace(
        region, rank_indices, S, min(kout, S), U, V, Z,
        _batch_views(Z, S),
        zeros(backend, rank_type, ntiles),
        zeros(backend, Float64, ntiles),
        pdev, qdev, ara, _batch_views(Q, S), omega,
        _batch_views(omega, blk), _batch_views(Y, blk),
    )
end

function _alloc_category_workspace(A_tlr::PaddedFTLRMatrix{<:Any,T}, spec,
                                   r::Int) where {T}
    backend = get_backend(A_tlr)
    n = spec.n
    p0_host = Vector{Int32}(undef, n)
    q0_host = Vector{Int32}(undef, n)
    rank_indices = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        p0, q0 = tile_origin_coords(
            A_tlr, region_tile_coords(A_tlr, spec.region, k)...)
        p0_host[k] = Int32(p0)
        q0_host[k] = Int32(q0)
        rank_indices[k] = _rank_index(A_tlr, spec.region, k)
    end
    p0s = copyto!(allocate(backend, Int32, n), p0_host)
    q0s = copyto!(allocate(backend, Int32, n), q0_host)
    return _make_category_workspace(
        spec.region, rank_indices, spec.U, spec.V,
        spec.tm, spec.tn, r, n;
        rank_type=eltype(A_tlr.ranks), p0s, q0s,
    )
end

"""
    alloc_workspace(A_tlr) → CompressWorkspace

Allocate every category workspace once for repeated `compress!` calls.
Sampling, orthogonalization, co-range storage, and diagnostics allocate no
device buffers after this call. Backend SVD libraries may still own internal
solver workspace during final truncation.
"""
function alloc_workspace(A_tlr::PaddedFTLRMatrix)
    specs = _region_specs(A_tlr)
    cats = map(spec -> _alloc_category_workspace(A_tlr, spec, A_tlr.maxrank),
               specs)
    return CompressWorkspace(cats,
                             create_streams(A_tlr.backend, length(cats)))
end

function alloc_workspace(A_tlr::TLRMatrix{<:Any,T}) where {T}
    scratch = PaddedFTLRMatrix(
        get_backend(A_tlr), T, size(A_tlr)..., nominal_tile_size(A_tlr),
        maxrank(A_tlr); tile_order=compressed_ftlr_outer_order(offdiagonal(A_tlr)),
        rank_type=eltype(ranks(A_tlr)))
    return TLRCompressWorkspace(scratch, alloc_workspace(scratch))
end

"""
    carve_tile_workspace(U, V, tm, tn, kout, ntiles; ...)

Construct a standalone category workspace around caller-owned output factors.
The name is retained for the dense-slab fallback; there is no longer a
full-width high-precision arena to carve.
"""
function carve_tile_workspace(U::AbstractArray{T,3}, V,
                              tm::Int, tn::Int, kout::Int, ntiles::Int;
                              rank_type::Type=Int32,
                              block::Int=compress_ara_block(
                                  min(kout, tm, tn))) where {T}
    return _make_category_workspace(
        nothing, Int[], U, V, tm, tn, kout, ntiles; rank_type, block)
end

"""
    alloc_tile_workspace(U, V, tm, tn, kout, ntiles; rank_type=Int32, block)

Standalone reusable ARA compression workspace for a homogeneous tile batch.
"""
function alloc_tile_workspace(U::AbstractArray{T,3}, V,
                              tm::Int, tn::Int, kout::Int, ntiles::Int;
                              rank_type::Type=Int32,
                              block::Int=compress_ara_block(
                                  min(kout, tm, tn))) where {T}
    return carve_tile_workspace(
        U, V, tm, tn, kout, ntiles; rank_type, block)
end
