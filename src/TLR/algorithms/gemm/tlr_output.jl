# TLR-output dense-slab fallback (ROADMAP milestone 4).
#
# A dense-output contraction writes each result tile straight into `C`. A TLR output
# instead accumulates each result tile into a bounded dense **slab**, then compresses
# the slab's tiles into the output container's low-rank factor panels. It shares the
# output-independent Stages 1/2 and owns a slab-specific terminal stage.
#
# This first milestone is deliberately scoped:
#   * regular grid only (no boundary tiles — one interior contraction per tile);
#   * `beta == 0` (the output factors are overwritten, not accumulated);
#   * the row family (`KAsGemmK`), which writes each output tile exactly once, so a
#     run's tiles complete and can be compressed immediately. The lone column-family
#     layout (A tile-column-major × B tile-row-major) is a later step.

# ─── Output workspace ─────────────────────────────────────────────────────────

"""
Bounded scratch for a TLR-output contraction: the dense accumulation `slab`, the
per-run staged-GEMM workspace `stage_ws`, temporary compressed-factor buffers `Uc`/`Vc`
(the compression aliases them, then they are scattered into the output container's factor
slots), the high-precision compression `accum` arena, and the reusable per-run tile-batch
apparatus (`tiles` slab views plus host/device slab-origin buffers `p0h`/`q0h`/`p0s`/`q0s`
for the norm kernel). Everything is sized to the widest run block `maxI × maxJ`, so
nothing scales with the whole output and no run allocates device memory of its own.
"""
struct TLROutputWorkspace{SlabT,WST,UcT,VcT,AccT,TileVT,I32H,I32D}
    slab::SlabT
    stage_ws::WST
    Uc::UcT
    Vc::VcT
    accum::AccT
    tiles::TileVT
    p0h::I32H
    q0h::I32H
    p0s::I32D
    q0s::I32D
    maxI::Int
    maxJ::Int
end

function _alloc_tlr_output_workspace(C::TLRMatrix{<:Any,T}, geom, placement, ops,
                                     budget::Int, fold) where {T}
    backend = get_backend(C)
    bm = geom.bm
    bn = geom.bn
    maxI, maxJ = _row_block(geom, budget, fold)
    G = maxI * maxJ
    slab = allocate(backend, T, maxI * bm, maxJ * bn)
    stage_ws = allocate_workspace(placement, geom, ops, slab, budget, fold)
    kout = C.maxrank
    Uc = allocate(backend, T, bm, kout, G)
    Vc = allocate(backend, T, bn, kout, G)
    e = compress_arena_elems(bm, bn, kout, G)
    accum = zeros(backend, _compress_accum_type(T), e.accum)
    tiles = Vector{typeof(view(slab, 1:bm, 1:bn))}()
    sizehint!(tiles, G)
    p0h = Vector{Int32}(undef, G)
    q0h = Vector{Int32}(undef, G)
    p0s = allocate(backend, Int32, G)
    q0s = allocate(backend, Int32, G)
    return TLROutputWorkspace(slab, stage_ws, Uc, Vc, accum, tiles, p0h, q0h, p0s, q0s, maxI, maxJ)
end

# ─── Execution: row family ────────────────────────────────────────────────────

"""
    _tlr_gemm_rowfamily!(C, ops, geometry, placement, fold, alpha, budget, compute,
                         workspace; eps_sq, rel) -> C

Run the write-once (`KAsGemmK`) low-rank contraction into a TLR output. Each run fills
the slab with `alpha·Σ_k A_ik B_kj` (β = 0), then the run's completed tiles are
compressed into `C`'s factor panels.
"""
@inline function _slab_tile(slab, run::RowRun, bm::Int, bn::Int, i::Int, j::Int)
    r = (i - run.i0) * bm
    c = (j - run.j0) * bn
    return view(slab, (r + 1):(r + bm), (c + 1):(c + bn))
end

@inline function _slab_rowblock(slab, run::RowRun, bm::Int, bn::Int,
                                i::Int, j0::Int, j1::Int)
    r = (i - run.i0) * bm
    c = (j0 - run.j0) * bn
    return view(slab, (r + 1):(r + bm), (c + 1):(c + (j1 - j0 + 1) * bn))
end

function execute_slab_stage3!(::KAsGemmK, ::FoldRight, run::RowRun, ops,
                              ws::RowWorkspace, slab, alpha, beta, compute)
    bm = size(ws.Ustacked, 1)
    bn = size(ws.T.data, 3)
    noff = size(ws.T.data, 2)
    rA = size(ws.T.data, 1)
    Jw = run_width(run)
    vb = ws.batches
    _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        push!(vb.s3u, view(ws.Ustacked, :, :, i))
        push!(vb.s3t, reshape(view(ws.T.data, :, :, :, 1:Jw, il),
                              noff * rA, Jw * bn))
        push!(vb.s3c, _slab_rowblock(slab, run, bm, bn, i, run.j0, run.j1))
    end
    return precision_gemm_batched!('N', 'N', alpha, vb.s3u, vb.s3t, beta,
                                   vb.s3c, compute)
end

function execute_slab_stage3!(::KAsGemmK, ::FoldLeft, run::RowRun, ops,
                              ws::RowWorkspace, slab, alpha, beta, compute)
    bm = blockdim(ops.au)
    bn = blockdim(ops.bz)
    rB = size(ws.T.data, 2)
    noff = size(ws.T.data, 3)
    vb = ws.batches
    _clear_batches!(vb.s3t, vb.s3z, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1),
                  (jl, j) in enumerate(run.j0:run.j1)
        push!(vb.s3t, reshape(view(ws.T.data, :, :, :, jl, il), bm, rB * noff))
        push!(vb.s3z, view(ws.Ustacked, :, :, j))
        push!(vb.s3c, _slab_tile(slab, run, bm, bn, i, j))
    end
    return precision_gemm_batched!('N', 'T', alpha, vb.s3t, vb.s3z, beta,
                                   vb.s3c, compute)
end

function _tlr_gemm_rowfamily!(C::TLRMatrix{<:Any,T}, ops, geom, placement,
                              fold, alpha, budget::Int, compute,
                              ws::TLROutputWorkspace; eps_sq::Float64,
                              rel::Bool) where {T}
    bm = geom.bm
    bn = geom.bn
    slab = ws.slab
    stage_ws = ws.stage_ws
    beta0 = zero(alpha)

    @inbounds for run in runs(placement, geom, budget, fold)
        prepare_run!(placement, run, stage_ws)
        execute_stage1!(placement, run, ops, stage_ws, compute)
        execute_stage2!(placement, fold, run, ops, stage_ws, compute)
        execute_slab_stage3!(placement, fold, run, ops, stage_ws, slab,
                             alpha, beta0, compute)
        _compress_run_into_factors!(C, ws, run, bm, bn; eps_sq, rel)
    end
    return C
end

# Compress one run's completed dense tiles (in the slab) into `C`'s interior factor
# panels, scattering the detected ranks / residuals into the tile-grid-aligned slots.
function _compress_run_into_factors!(C::TLRMatrix{<:Any,T}, ws::TLROutputWorkspace,
                                     run::RowRun, bm::Int, bn::Int;
                                     eps_sq::Float64, rel::Bool) where {T}
    h = run.i1 - run.i0 + 1
    w = run.j1 - run.j0 + 1
    g = h * w
    slab = ws.slab

    Uc = view(ws.Uc, :, :, 1:g)
    Vc = view(ws.Vc, :, :, 1:g)

    # Refill the reusable tile batch and slab-origin buffers (no per-run allocation).
    # A run's tile `(il, jl)` always sits at the fixed slab block `((il-1)·bm, (jl-1)·bn)`.
    empty!(ws.tiles)
    idx = 0
    @inbounds for jl in 1:w, il in 1:h
        idx += 1
        r = (il - 1) * bm
        c = (jl - 1) * bn
        push!(ws.tiles, view(slab, (r + 1):(r + bm), (c + 1):(c + bn)))
        ws.p0h[idx] = r + 1
        ws.q0h[idx] = c + 1
    end
    # Offset/count copyto! (not view-to-view): CUDA supports host→device this way, whereas
    # copying a host SubArray into a device SubArray falls back to scalar iteration.
    copyto!(ws.p0s, 1, ws.p0h, 1, g)
    copyto!(ws.q0s, 1, ws.q0h, 1, g)
    p0s = view(ws.p0s, 1:g)
    q0s = view(ws.q0s, 1:g)

    # `carve_tile_workspace` reuses the preallocated `Uc`/`Vc`/`accum` device storage but
    # still builds small per-tile view vectors on the host. That host churn is dwarfed by
    # the compression compute it feeds (cholqr2 / batched POTRF / TRSM / prune) and, at the
    # default budget, happens once — so it is left as a bounded, non-hot-path residual.
    cat, _ = carve_tile_workspace(Uc, Vc, bm, bn, C.maxrank, g, ws.accum;
                                  rank_type=eltype(C.ranks))
    src = DenseTiles(slab, ws.tiles, p0s, q0s, bm, bn)
    compress_tiles!(src, cat; eps_sq, rel)

    rk = cat.ranks_local isa Vector ? cat.ranks_local : Array(cat.ranks_local)
    err = cat.norm_err_sq isa Vector ? cat.norm_err_sq : Array(cat.norm_err_sq)
    q_m, q_n = regular_tilegrid_size(C)
    idx = 0
    @inbounds for jl in 1:w, il in 1:h
        idx += 1
        i = run.i0 + il - 1
        j = run.j0 + jl - 1
        slot = tile_linear_index(C.order, q_m, q_n, i, j)
        copyto!(view(C.int_U, :, :, slot), view(ws.Uc, :, :, idx))
        copyto!(view(C.int_V, :, :, slot), view(ws.Vc, :, :, idx))
        ridx = _rank_index(C, i, j)
        C.ranks[ridx] = rk[idx]
        C.resid[ridx] = sqrt(max(Float64(real(err[idx])), 0.0))
    end
    return C
end
