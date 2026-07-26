struct CompressCategoryWorkspace{RegionT,PanelT,ScratchT,ScratchHiT,TileVT,RTileVT,
    RankVT,ErrV,ShiftV,I32V,RankIndexT,ARAT,QBufT,OmegaT}
    region::RegionT        # TLR region, or `nothing` for a standalone tile batch
    rank_indices::RankIndexT # category-local tile slot -> A_tlr.ranks / A_tlr.resid slot
    S::Int                 # sketch width    = min(maxrank, tm, tn)
    R_keep::Int            # max stored rank  = min(maxrank, S)
    U::PanelT              # output left factors  (aliases A_tlr, maxrank-wide)
    V::PanelT              # output right factors (aliases A_tlr, maxrank-wide)
    Q_T::ScratchT          # S-wide view into output U; holds Y, then Q
    V_T::ScratchT          # S-wide view into output V; holds Ω, then AᴴQ
    Q_tiles::TileVT        # per-tile GEMM operand views into Q_T / V_T
    V_tiles::TileVT
    R_tiles::RTileVT       # leading S×S views into expired Ω storage
    Y_hi::ScratchHiT       # accumulation-precision Q copy for cholqr (tm × S × n)
    G_hi::ScratchHiT       # accumulation-precision Gram matrix        (S × S × n)
    ranks_local::RankVT    # per-tile detected rank
    norm_err_sq::ErrV      # per-tile ‖A_tile‖²_F, overwritten by squared error
    shift_mult::ShiftV     # view into expired V[1,1,:] for POTRF escalation
    p0s::I32V              # per-tile dense-source row origin (1-based)
    q0s::I32V              # per-tile dense-source col origin (1-based)
    ara::ARAT              # blocked ARA loop scratch (basis grown into Qbuf)
    Qbuf::QBufT            # tm × S × n basis; cannot alias U, since the final
                           # lift U = Q·W would then read and write one array
    omega::OmegaT          # tn × block × n sketch block
end

# Reusable scratch for a matrix layout. The output capacity is also the sketch
# capacity, so no additional work-precision factor panels are allocated.
struct CompressWorkspace{CatsT,StreamV}
    cats::CatsT
    streams::StreamV # one execution stream for each category on gpu
end

# Region, output factor panels, and tile dimensions for every low-rank region.
@inline function _region_specs(A_tlr::AbstractTLRMatrix)
    map(lowrank_regions(A_tlr)) do region
        U = outer_factors(A_tlr, region)
        V = inner_factors(A_tlr, region)
        (; region, n=region_tile_count(A_tlr, region), U, V,
            tm=size(U, 1), tn=size(V, 1))
    end
end

# prepare category's scratch at sketch width S
function _alloc_category_workspace(A_tlr::AbstractTLRMatrix{<:Any,T}, spec, r::Int, ::Type{Thi}) where {T,Thi}
    backend = get_backend(A_tlr)
    rank_type = eltype(A_tlr.ranks)
    n = spec.n
    S = min(r, spec.tm, spec.tn)

    Q_T = view(spec.U, :, 1:S, :)
    V_T = view(spec.V, :, 1:S, :)
    shift_mult = S == 0 ? reshape(view(V_T, 1, 1:0, :), 0) : view(V_T, 1, 1, :)
    Y_hi = zeros(backend, Thi, spec.tm, S, n)
    G_hi = zeros(backend, Thi, S, S, n)
    norm_err_sq = S == 0 ? zeros(backend, Float64, n) : view(G_hi, 1, 1, :)
    p0_host = Vector{Int32}(undef, n)
    q0_host = Vector{Int32}(undef, n)
    rank_indices = Vector{Int}(undef, n)

    @inbounds for k in 1:n
        p0, q0 = tile_origin_coords(A_tlr, region_tile_coords(A_tlr, spec.region, k)...)
        p0_host[k] = Int32(p0)
        q0_host[k] = Int32(q0)
        rank_indices[k] = _rank_index(A_tlr, spec.region, k)
    end

    p0s = copyto!(allocate(backend, Int32, n), p0_host)
    q0s = copyto!(allocate(backend, Int32, n), q0_host)

    # The ARA loop samples in blocks rather than once at full width, so the
    # sketch is never the wide rank-deficient panel that defeated the old
    # CholQR2 prune (docs/TODO.md, worklog items 1 and 3).
    Qbuf = zeros(backend, T, spec.tm, S, n)
    blk = compress_ara_block(S)
    ara = S == 0 ? nothing : ARAWorkspace(Qbuf; block=blk)
    omega = zeros(backend, T, spec.tn, max(blk, 1), n)

    return CompressCategoryWorkspace(
        spec.region, rank_indices, S, min(r, S), spec.U, spec.V, Q_T, V_T,
        _batch_views(Q_T, S), _batch_views(V_T, S),
        [view(V_T, 1:S, 1:S, k) for k in axes(V_T, 3)], Y_hi, G_hi,
        zeros(backend, rank_type, n),
        norm_err_sq,
        shift_mult,
        p0s,
        q0s,
        ara,
        Qbuf,
        omega,
    )
end

"""
    compress_ara_block(S) -> Int

Sampling block width for `compress!`. Purely a performance knob — the recovered
rank and the achieved error do not depend on it (there is a test) — so it is set
to the reference's 32 (a warp) and clamped to the available capacity.
"""
@inline compress_ara_block(S::Int) = max(min(32, S), 1)

"""
    alloc_workspace(A_tlr) → CompressWorkspace

Pre-allocate `compress!` scratch for `A_tlr`, one bundle per off-diagonal tile
category at sketch width `S = min(maxrank, tile)`. The sketch basis and right
factor live directly in the first `S` columns of the output `U`/`V` panels, so
only the high-precision CholQR copy and Gram matrices are allocated. Reuse the
workspace across repeated calls on the same layout:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end
"""
function alloc_workspace(A_tlr::AbstractTLRMatrix{<:Any,T}) where {T}
    Thi = _compress_accum_type(T)

    specs = _region_specs(A_tlr)
    cats = map(spec -> _alloc_category_workspace(A_tlr, spec, A_tlr.maxrank, Thi), specs)

    CompressWorkspace(cats, create_streams(A_tlr.backend, length(cats)))
end

# Reshape `prod(dims)` elements of a flat arena starting after `off`, returning the
# view and the advanced offset (so successive carves thread through one buffer).
@inline function _take(buf::AbstractVector, off::Int, dims::Vararg{Int})
    len = prod(dims)
    return reshape(view(buf, (off+1):(off+len)), dims...), off + len
end

"""
    compress_arena_elems(tm, tn, kout, ntiles) -> (; S, accum, work)

Element counts one category needs from its two scratch arenas, at sketch width
`S = min(kout, tm, tn)`. `accum` is high precision and holds `Y_hi`/`G_hi`;
`work` is working precision and holds the ARA basis `Qbuf` plus one sketch
block `omega`.

`Qbuf` cannot alias the output `U` the way the co-range does: the final lift
`U = Q·W` would then read and write the same array.
"""
@inline function compress_arena_elems(tm::Int, tn::Int, kout::Int, ntiles::Int)
    S = min(kout, tm, tn)
    blk = compress_ara_block(S)
    return (; S,
            accum=(tm * S + S * S) * ntiles,
            work=(tm * S + tn * blk) * ntiles)
end

"""
    compress_bytes(T, tm, tn, kout, ntiles) -> Int

Bytes one category needs in scratch, across both arenas.
"""
@inline function compress_bytes(::Type{T}, tm::Int, tn::Int, kout::Int, ntiles::Int) where {T}
    e = compress_arena_elems(tm, tn, kout, ntiles)
    return e.accum * sizeof(_compress_accum_type(T)) + e.work * sizeof(T)
end

"""
    carve_tile_workspace(U, V, tm, tn, kout, ntiles, accum;
                         accum_off=0, rank_type=Int32) -> (cat, accum_off′)

Carve a [`CompressCategoryWorkspace`](@ref) out of caller-provided typed arenas.
`Q_T`/`V_T` alias the first `S` columns of the caller's output `U`/`V`.
`Y_hi`/`G_hi` are carved from `accum::Vector{Thi}`. The offset advances and is
returned so several categories can share one arena. Size the arena with
[`compress_arena_elems`](@ref).
"""
function carve_tile_workspace(U::AbstractArray{T,3}, V, tm::Int, tn::Int, kout::Int, ntiles::Int,
    accum::AbstractVector{Thi}; accum_off::Int=0,
    rank_type::Type=Int32) where {T,Thi}
    backend = get_backend(U)
    S = min(kout, tm, tn)
    Q_T = view(U, :, 1:S, :)
    V_T = view(V, :, 1:S, :)
    shift_mult = S == 0 ? reshape(view(V_T, 1, 1:0, :), 0) : view(V_T, 1, 1, :)
    Y_hi, accum_off = _take(accum, accum_off, tm, S, ntiles)
    G_hi, accum_off = _take(accum, accum_off, S, S, ntiles)
    norm_err_sq = S == 0 ? zeros(backend, Float64, ntiles) : view(G_hi, 1, 1, :)
    empty_i32 = allocate(backend, Int32, 0)
    blk = compress_ara_block(S)
    Qbuf = zeros(backend, T, tm, S, ntiles)
    omega = zeros(backend, T, tn, blk, ntiles)
    cat = CompressCategoryWorkspace(
        nothing, Int[], S, min(kout, S), U, V, Q_T, V_T,
        _batch_views(Q_T, S), _batch_views(V_T, S),
        [view(V_T, 1:S, 1:S, k) for k in axes(V_T, 3)], Y_hi, G_hi,
        zeros(backend, rank_type, ntiles),
        norm_err_sq,
        shift_mult,
        empty_i32, empty_i32,
        S == 0 ? nothing : ARAWorkspace(Qbuf; block=blk), Qbuf, omega,
    )
    return cat, accum_off
end

"""
    alloc_tile_workspace(U, V, tm, tn, kout, ntiles; rank_type=Int32)

Standalone [`CompressCategoryWorkspace`](@ref) for compressing an `ntiles`-batch of
`tm×tn` tiles into output factors `U` (`tm×kout×ntiles`) and `V` (`tn×kout×ntiles`),
not tied to a TLR matrix. Allocates the high-precision scratch arena and carves it via
[`carve_tile_workspace`](@ref). Pair with [`compress_tiles!`](@ref) and a source
such as [`PackedTiles`](@ref).
"""
function alloc_tile_workspace(U::AbstractArray{T,3}, V, tm::Int, tn::Int,
    kout::Int, ntiles::Int; rank_type::Type=Int32) where {T}
    e = compress_arena_elems(tm, tn, kout, ntiles)
    backend = get_backend(U)
    accum = zeros(backend, _compress_accum_type(T), e.accum)
    cat, _ = carve_tile_workspace(U, V, tm, tn, kout, ntiles, accum; rank_type)
    return cat
end
