# Stage descriptors

struct StageDescriptor{StageT<:GemmStage,PlacementT<:KAxisSchedule,RunT,OpsT,WST,CT,AlphaT,BetaT,FoldT,BlockingT,ComputeT}
    stage::StageT
    placement::PlacementT
    run::RunT
    ops::OpsT
    workspace::WST
    C::CT
    alpha::AlphaT
    beta::BetaT
    fold::FoldT
    free_axis_schedule::BlockingT
    compute::ComputeT
end

@inline stage1(placement::KAxisSchedule, run, ops, ws, fold::FoldSide, compute) =
    StageDescriptor(Stage1(), placement, run, ops, ws, nothing, nothing, nothing, fold, free_axis_schedule(placement), compute)

@inline stage2(placement::KAxisSchedule, run, ops, ws, fold::FoldSide, compute) =
    StageDescriptor(Stage2(), placement, run, ops, ws, nothing, nothing, nothing, fold, nothing, compute)

# `beta` is applied by Stage 3 on the (single) write of each output tile. The row
# family is write-once so it passes β directly; the column family loops the reduction,
# so `_offdiag_offdiag_gemm!` pre-scales the region and passes β = 1 here. `FoldLeft` is
# only ever paired with the row family, so it too folds β in its single write.
@inline stage3(placement::KAxisSchedule, run, ops, ws, C::AbstractMatrix, alpha, beta, fold::FoldSide, compute) =
    StageDescriptor(Stage3(), placement, run, ops, ws, C, alpha, beta, fold, nothing, compute)

@inline _ws_eltype(ws) = eltype(ws.S.data)

@inline function _clear_batches!(buffers...)
    foreach(empty!, buffers)
    return nothing
end

function prepare_run!(::KAsGemmK, run::RowRun, ws::RowWorkspace)
    Tview = view(ws.T.data, :, :, :, 1:run_width(run), 1:run_height(run))
    fill!(Tview, zero(eltype(ws.T.data)))
    return nothing
end

@inline prepare_run!(::KAsSerialLoop, ::ColumnRun, ::ColumnWorkspace) = nothing


# Dispatch on the descriptor's stored singletons (`stage`, reduction `placement`,
# `fold`, and Stage-1 `free_axis_schedule`) rather than on positional type-parameter
# slots. All four are `StageDescriptor` type parameters, so this forward is resolved at
# compile time — no runtime cost — while keeping the method signatures readable and
# robust to adding further descriptor fields. Stage 1 is fold-independent (ignores the
# fold argument); Stage 2/3 carry `free_axis == nothing` and ignore that last argument.
@inline execute_stage!(d::StageDescriptor) =
    execute_stage!(d.stage, d.placement, d.fold, d.free_axis_schedule, d)

function execute_stage!(::Stage1, ::KAsGemmK, ::Any, ::FreeAsBatch, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s1v, vb.s1w, vb.s1s)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            for j in run.j0:run.j1
                p = col_scratch_pos(ops.bw, run.j0, k, j)
                p == 0 && continue
                push!(vb.s1v, tilefactor(ops.av, i, k))
                push!(vb.s1w, tilefactor(ops.bw, k, j))
                push!(vb.s1s, view(ws.S.data, :, :, p, kk, il))
            end
        end
    end
    isempty(vb.s1v) && return nothing
    return precision_gemm_batched!('T', 'N', one(T), vb.s1v, vb.s1w, zero(T), vb.s1s, d.compute)
end

# Fused Stage 1 for `(k,j)`: B stride-1 `:j` makes `rowpanel(k)` contiguous over
# `j`, so the block's off-diagonal columns form a single wide right operand and
# `j` folds into N — one GEMM per `(i,k)` instead of one per `(i,k,j)`.
function execute_stage!(::Stage1, ::KAsGemmK, ::Any, ::JAsGemmN, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s1jv, vb.s1jw, vb.s1js)
    b = blockdim(ops.bw)
    rB = rankdim(ops.bw)
    rA = size(ws.S.data, 1)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            jf = first_offdiag_col(ops.bw, run.j0, k)        # first included column
            je = last_offdiag_col(ops.bw, run.j1, k)         # last included column
            jf > je && continue                              # block is only the diagonal
            lrange = panel_local(ops.bw, k, jf):panel_local(ops.bw, k, je) # contiguous panel slice
            len = length(lrange)
            Wpanel = rowpanel(ops.bw, k)
            Wsub = reshape(view(Wpanel, :, :, lrange), b, len * rB)
            Ssub = reshape(view(ws.S.data, :, :, 1:len, kk, il), rA, len * rB)
            push!(vb.s1jv, tilefactor(ops.av, i, k))
            push!(vb.s1jw, Wsub)
            push!(vb.s1js, Ssub)
        end
    end
    isempty(vb.s1jv) && return nothing
    return precision_gemm_batched!('T', 'N', one(T), vb.s1jv, vb.s1jw, zero(T), vb.s1js, d.compute)
end

function execute_stage!(::Stage2, ::KAsGemmK, ::FoldRight, ::Any, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s2s, vb.s2z, vb.s2t)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            for j in run.j0:run.j1
                p = col_scratch_pos(ops.bw, run.j0, k, j)
                p == 0 && continue
                jl = j - run.j0 + 1
                push!(vb.s2s, view(ws.S.data, :, :, p, kk, il))
                push!(vb.s2z, tilefactor(ops.bz, k, j))
                push!(vb.s2t, view(ws.T.data, :, kk, :, jl, il))
            end
        end
    end
    isempty(vb.s2s) && return nothing
    return precision_gemm_batched!('N', 'T', one(T), vb.s2s, vb.s2z, zero(T), vb.s2t, d.compute)
end

function execute_stage!(::Stage3, ::KAsGemmK, ::FoldRight, ::Any, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ws = d.workspace
    bm = size(ws.Ustacked, 1)                # C row-tile height (from A's U factor)
    bn = size(ws.T.data, 3)                  # C col-tile width  (T's spatial extent)
    noff = size(ws.T.data, 2)
    rA = size(ws.T.data, 1)
    Jw = run_width(run)
    vb = ws.batches
    _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        Tstack = reshape(view(ws.T.data, :, :, :, 1:Jw, il), noff * rA, Jw * bn)
        push!(vb.s3u, view(ws.Ustacked, :, :, i))
        push!(vb.s3t, Tstack)
        push!(vb.s3c, dense_rowblock(d.C, bm, bn, i, run.j0, run.j1))
    end
    return precision_gemm_batched!('N', 'N', d.alpha, vb.s3u, vb.s3t, d.beta, vb.s3c, d.compute)
end

# ── FoldLeft (row family, FullGrid) ──────────────────────────────────────────
# The mirror of the FoldRight row family: fold A's `U` into Stage 2 (`T' = U·S`,
# `bm×rB`) and stack B's `Z` in the write-once Stage 3. Chosen only when B is
# TileColMajor (so the Z-stack is contiguous) on a FullGrid interior (no diagonal
# skip). Stage 1 is shared with FoldRight (the `FreeAsBatch` executor writes `S`).

# Stage 2: T'_ikj = U_ik · S_ikj  (bm×rB), tilewise over (i, k, j).
function execute_stage!(::Stage2, ::KAsGemmK, ::FoldLeft, ::Any, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s2u, vb.s2s, vb.s2t)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            for j in run.j0:run.j1
                p = col_scratch_pos(ops.bw, run.j0, k, j)
                p == 0 && continue
                jl = j - run.j0 + 1
                push!(vb.s2u, tilefactor(ops.au, i, k))          # U_ik  [bm, rA]
                push!(vb.s2s, view(ws.S.data, :, :, p, kk, il))  # S_ikj [rA, rB]
                push!(vb.s2t, view(ws.T.data, :, :, kk, jl, il)) # T'    [bm, rB]
            end
        end
    end
    isempty(vb.s2u) && return nothing
    return precision_gemm_batched!('N', 'N', one(T), vb.s2u, vb.s2s, zero(T), vb.s2t, d.compute)
end

# Stage 3 (write-once): C_ij = β·C_ij + α·Σ_k T'_ikj Z_kj'. Stack over k: the left
# `T'stack_ij` (bm × noff·rB) is a contiguous reshape of our own T' buffer; the right
# `Zstack_j` (bn × noff·rB) is a per-column view of `reshape(bz.data, bn, rB·noff, q_n)`
# (contiguous ⟺ B TileColMajor). `'N','T'` gives `T'stack · Zstack'` = [bm, bn]; the
# (rB fast, k slow) column order matches on both sides. Batched over (i, j); β folded.
function execute_stage!(::Stage3, ::KAsGemmK, ::FoldLeft, ::Any, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    bm = blockdim(ops.au)                    # C row-tile height
    bn = blockdim(ops.bz)                    # C col-tile width
    rB = size(ws.T.data, 2)
    noff = size(ws.T.data, 3)
    Zall = ws.Ustacked                       # [bn, rB·noff, q_n] — stacked B's Z
    vb = ws.batches
    _clear_batches!(vb.s3t, vb.s3z, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for (jl, j) in enumerate(run.j0:run.j1)
            Tstack = reshape(view(ws.T.data, :, :, :, jl, il), bm, rB * noff)
            push!(vb.s3t, Tstack)
            push!(vb.s3z, view(Zall, :, :, j))
            push!(vb.s3c, dense_tile(d.C, bm, bn, i, j))
        end
    end
    return precision_gemm_batched!('N', 'T', d.alpha, vb.s3t, vb.s3z, d.beta, vb.s3c, d.compute)
end

function execute_stage!(::Stage1, ::KAsSerialLoop, ::Any, ::IAsGemmM, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s1v, vb.s1w, vb.s1s)

    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        Vpanel = view(ws.Vstacked, :, :, k)
        for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
            j = panel_col(ops.bw, k, jpos)
            push!(vb.s1v, Vpanel)
            push!(vb.s1w, tilefactor(ops.bw, k, j))
            push!(vb.s1s, view(ws.S.data, :, :, jx, kx))
        end
    end
    return precision_gemm_batched!('T', 'N', one(T), vb.s1v, vb.s1w, zero(T), vb.s1s, d.compute)
end

function execute_stage!(::Stage1, ::KAsSerialLoop, ::Any, ::IJAsGemmMN, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s1jv, vb.s1jw, vb.s1js)

    Jw = run_width(run)
    b = blockdim(ops.bw)
    rB = rankdim(ops.bw)
    rA_noff = size(ws.S.data, 1)
    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        Vpanel = view(ws.Vstacked, :, :, k)
        Wpanel = rowpanel(ops.bw, k)
        Wsub = reshape(view(Wpanel, :, :, run.jpos0:run.jpos1), b, Jw * rB)
        Ssub = reshape(view(ws.S.data, :, :, 1:Jw, kx), rA_noff, Jw * rB)
        push!(vb.s1jv, Vpanel)
        push!(vb.s1jw, Wsub)
        push!(vb.s1js, Ssub)
    end
    return precision_gemm_batched!('T', 'N', one(T), vb.s1jv, vb.s1jw, zero(T), vb.s1js, d.compute)
end

function execute_stage!(::Stage2, ::KAsSerialLoop, ::FoldRight, ::Any, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    rA_noff = size(ws.S.data, 1)
    b = blockdim(ops.bz)
    _clear_batches!(vb.s2s, vb.s2z, vb.s2t)

    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
            j = panel_col(ops.bz, k, jpos)
            push!(vb.s2s, view(ws.S.data, :, :, jx, kx))
            push!(vb.s2z, tilefactor(ops.bz, k, j))
            push!(vb.s2t, reshape(view(ws.T.data, :, :, :, jx, kx), rA_noff, b))
        end
    end
    return precision_gemm_batched!('N', 'T', one(T), vb.s2s, vb.s2z, zero(T), vb.s2t, d.compute)
end

function execute_stage!(::Stage3, ::KAsSerialLoop, ::FoldRight, ::Any, d::StageDescriptor)
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    bm = size(ws.Ufactored, 1)               # C row-tile height (from A's U factor)
    bn = size(ws.T.data, 3)                  # C col-tile width  (T's spatial extent)
    vb = ws.batches
    noff = size(ws.Ufactored, 3)
    # `k` is the reduction axis: loop it (one accumulate GEMM per k), batching the
    # free axes (i, jpos). Different k write the same C_ij, so they cannot share a
    # batch — but successive launches accumulate with β = 1.
    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
        for li in 1:noff
            i = panel_col(ops.au, k, li)
            for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
                j = panel_col(ops.bw, k, jpos)
                push!(vb.s3u, view(ws.Ufactored, :, :, li, k))
                push!(vb.s3t, view(ws.T.data, :, li, :, jx, kx))
                push!(vb.s3c, dense_tile(d.C, bm, bn, i, j))
            end
        end
        precision_gemm_batched!('N', 'N', d.alpha, vb.s3u, vb.s3t, d.beta, vb.s3c, d.compute)
    end
    return nothing
end
