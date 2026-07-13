struct CompressCategoryWorkspace{RegionT,PanelT,ScratchT,ScratchHiT,TileVT,RankVT,F64V,ShiftV,I32V,RankIndexT}
    region::RegionT        # TLR region, or `nothing` for a standalone tile batch
    rank_indices::RankIndexT # category-local tile slot -> A_tlr.ranks / A_tlr.resid slot
    S::Int                 # sketch width    = min(maxrank + oversample, tm, tn)
    R_keep::Int            # max stored rank  = min(maxrank, S)
    U::PanelT              # output left factors  (aliases A_tlr, maxrank-wide)
    V::PanelT              # output right factors (aliases A_tlr, maxrank-wide)
    Q_T::ScratchT          # orthonormal basis Q          (tm × S × n)
    V_T::ScratchT          # random Ω, then co-range Aᴴ·Q (tn × S × n)
    Q_tiles::TileVT        # per-tile GEMM operand views into Q_T / V_T
    V_tiles::TileVT
    Y_hi::ScratchHiT       # accumulation-precision Q copy for cholqr (tm × S × n)
    G_hi::ScratchHiT       # accumulation-precision Gram matrix        (S × S × n)
    ranks_local::RankVT    # per-tile detected rank
    err_sq_local::F64V     # per-tile squared error estimate
    normA_sq::F64V         # per-tile ‖A_tile‖²_F
    shift_mult::ShiftV     # per-tile first-pass shift escalation multiplier
    p0s::I32V              # per-tile dense-source row origin (1-based)
    q0s::I32V              # per-tile dense-source col origin (1-based)
end

# reusable scratch for matrix layout + oversampling
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
function _alloc_category_workspace(A_tlr::AbstractTLRMatrix{<:Any,T}, spec, r::Int, p::Int, ::Type{Thi}) where {T,Thi}
    backend = get_backend(A_tlr)
    rank_type = eltype(A_tlr.ranks)
    n = spec.n
    S = max(min(r + p, spec.tm, spec.tn), 1)

    Q_T = zeros(backend, T, spec.tm, S, n)
    V_T = zeros(backend, T, spec.tn, S, n)
    Y_hi = zeros(backend, Thi, spec.tm, S, n)
    G_hi = zeros(backend, Thi, S, S, n)
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

    return CompressCategoryWorkspace(
        spec.region, rank_indices, S, min(r, S), spec.U, spec.V, Q_T, V_T,
        _batch_views(Q_T, S), _batch_views(V_T, S), Y_hi, G_hi,
        zeros(backend, rank_type, n),
        zeros(backend, Float64, n),
        zeros(backend, Float64, n),
        zeros(backend, real(Thi), n),
        p0s,
        q0s,
    )
end

"""
    alloc_workspace(A_tlr; oversample=0) → CompressWorkspace

Pre-allocate `compress!` scratch for `A_tlr`, one bundle per off-diagonal tile
category at sketch width `S = min(maxrank + oversample, tile)`; `U`/`V` alias
`A_tlr` storage directly. Reuse across repeated calls on the same layout and
`oversample`:

    ws = alloc_workspace(A_tlr; oversample=8)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end
"""
function alloc_workspace(A_tlr::AbstractTLRMatrix{<:Any,T}; oversample::Int=0) where {T}
    oversample >= 0 || throw(ArgumentError("oversample must be >= 0"))
    Thi = _compress_accum_type(T)

    specs = _region_specs(A_tlr)
    cats = map(spec -> _alloc_category_workspace(A_tlr, spec, A_tlr.maxrank, oversample, Thi), specs)

    CompressWorkspace(cats, create_streams(A_tlr.backend, length(cats)))
end

# Reshape `prod(dims)` elements of a flat arena starting after `off`, returning the
# view and the advanced offset (so successive carves thread through one buffer).
@inline function _take(buf::AbstractVector, off::Int, dims::Vararg{Int})
    len = prod(dims)
    return reshape(view(buf, (off+1):(off+len)), dims...), off + len
end

"""
    compress_arena_elems(tm, tn, kout, ntiles; oversample=0) -> (; S, work, accum)

Element counts one category needs from the two typed scratch arenas:
`work` (precision `T`: `Q_T` + `V_T`) and `accum` (precision `Thi`: `Y_hi` + `G_hi`),
at sketch width `S = min(kout + oversample, tm, tn)`. Sum over categories to size a
shared arena. (`kin`, the input rank, does not enter for dense/packed sources — it
will once `LowRankTiles` adds its `kin×S` temporaries.)
"""
@inline function compress_arena_elems(tm::Int, tn::Int, kout::Int, ntiles::Int; oversample::Int=0)
    S = max(min(kout + oversample, tm, tn), 1)
    return (; S, work=(tm * S + tn * S) * ntiles, accum=(tm * S + S * S) * ntiles)
end

"""
    compress_bytes(T, tm, tn, kout, ntiles; oversample=0) -> Int

Bytes one category needs across both scratch arenas — `work·sizeof(T) +
accum·sizeof(Thi)` — for budgeting a slice of a larger workspace.
"""
@inline function compress_bytes(::Type{T}, tm::Int, tn::Int, kout::Int, ntiles::Int; oversample::Int=0) where {T}
    e = compress_arena_elems(tm, tn, kout, ntiles; oversample)
    return e.work * sizeof(T) + e.accum * sizeof(_compress_accum_type(T))
end

"""
    carve_tile_workspace(U, V, tm, tn, kout, ntiles, work, accum;
                         work_off=0, accum_off=0, oversample=0, rank_type=Int32)
        -> (cat, work_off′, accum_off′)

Carve a [`CompressCategoryWorkspace`](@ref) out of caller-provided typed arenas:
`Q_T`/`V_T` from `work::Vector{T}` and `Y_hi`/`G_hi` from `accum::Vector{Thi}`
(`Thi = _compress_accum_type(T)`; pass the same buffer for both when `T == Thi`,
slicing so the regions don't overlap). Offsets advance and are returned so several
categories can share one arena. `U`/`V` are the caller's output panels; the small
per-tile vectors are allocated fresh. Size the arenas with [`compress_arena_elems`](@ref).
"""
function carve_tile_workspace(U::AbstractArray{T,3}, V, tm::Int, tn::Int, kout::Int, ntiles::Int,
    work::AbstractVector{T}, accum::AbstractVector{Thi};
    work_off::Int=0, accum_off::Int=0,
    oversample::Int=0, rank_type::Type=Int32) where {T,Thi}
    backend = get_backend(U)
    S = max(min(kout + oversample, tm, tn), 1)
    Q_T, work_off = _take(work, work_off, tm, S, ntiles)
    V_T, work_off = _take(work, work_off, tn, S, ntiles)
    Y_hi, accum_off = _take(accum, accum_off, tm, S, ntiles)
    G_hi, accum_off = _take(accum, accum_off, S, S, ntiles)
    empty_i32 = allocate(backend, Int32, 0)
    cat = CompressCategoryWorkspace(
        nothing, Int[], S, min(kout, S), U, V, Q_T, V_T,
        _batch_views(Q_T, S), _batch_views(V_T, S), Y_hi, G_hi,
        zeros(backend, rank_type, ntiles),
        zeros(backend, Float64, ntiles),
        zeros(backend, Float64, ntiles),
        zeros(backend, real(Thi), ntiles),
        empty_i32, empty_i32,
    )
    return cat, work_off, accum_off
end

"""
    alloc_tile_workspace(U, V, tm, tn, kout, ntiles; oversample=0, rank_type=Int32)

Standalone [`CompressCategoryWorkspace`](@ref) for compressing an `ntiles`-batch of
`tm×tn` tiles into output factors `U` (`tm×kout×ntiles`) and `V` (`tn×kout×ntiles`),
not tied to a TLR matrix. Allocates the two scratch arenas and carves them via
[`carve_tile_workspace`](@ref). Pair with [`compress_tiles!`](@ref) and a source
such as [`PackedTiles`](@ref).
"""
function alloc_tile_workspace(U::AbstractArray{T,3}, V, tm::Int, tn::Int,
    kout::Int, ntiles::Int; oversample::Int=0, rank_type::Type=Int32) where {T}
    e = compress_arena_elems(tm, tn, kout, ntiles; oversample)
    backend = get_backend(U)
    work = zeros(backend, T, e.work)
    accum = zeros(backend, _compress_accum_type(T), e.accum)
    cat, _, _ = carve_tile_workspace(U, V, tm, tn, kout, ntiles, work, accum; oversample, rank_type)
    return cat
end
