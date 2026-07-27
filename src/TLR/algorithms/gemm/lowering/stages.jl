# Direct BLAS-stage execution. Layout/family/fold choices are ordinary dispatch
# arguments selected once by the caller; no operation or stage descriptor is built.

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


function execute_stage1!(::KAsGemmK{:k}, run::RowRun, ops, ws::RowWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    _clear_batches!(vb.s1v, vb.s1w, vb.s1s)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            for j in run.j0:run.j1
                p = col_scratch_pos(ops.bu, run.j0, k, j)
                p == 0 && continue
                push!(vb.s1v, tilefactor(ops.av, i, k))
                push!(vb.s1w, tilefactor(ops.bu, k, j))
                push!(vb.s1s, view(ws.S.data, :, :, p, kk, il))
            end
        end
    end
    isempty(vb.s1v) && return nothing
    return precision_gemm_batched!('T', 'N', one(T), vb.s1v, vb.s1w, zero(T), vb.s1s, compute)
end

# Fused Stage 1 for `(k,j)`: B stride-1 `:j` makes `rowpanel(k)` contiguous over
# `j`, so the block's off-diagonal columns form a single wide right operand and
# `j` folds into N — one GEMM per `(i,k)` instead of one per `(i,k,j)`.
function execute_stage1!(::KAsGemmK{:j}, run::RowRun, ops, ws::RowWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    _clear_batches!(vb.s1jv, vb.s1jw, vb.s1js)
    b = blockdim(ops.bu)
    rB = rankdim(ops.bu)
    rA = size(ws.S.data, 1)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            jf = first_offdiag_col(ops.bu, run.j0, k)        # first included column
            je = last_offdiag_col(ops.bu, run.j1, k)         # last included column
            jf > je && continue                              # block is only the diagonal
            lrange = panel_local(ops.bu, k, jf):panel_local(ops.bu, k, je) # contiguous panel slice
            len = length(lrange)
            Wpanel = rowpanel(ops.bu, k)
            Wsub = reshape(view(Wpanel, :, :, lrange), b, len * rB)
            Ssub = reshape(view(ws.S.data, :, :, 1:len, kk, il), rA, len * rB)
            push!(vb.s1jv, tilefactor(ops.av, i, k))
            push!(vb.s1jw, Wsub)
            push!(vb.s1js, Ssub)
        end
    end
    isempty(vb.s1jv) && return nothing
    return precision_gemm_batched!('T', 'N', one(T), vb.s1jv, vb.s1jw, zero(T), vb.s1js, compute)
end

function execute_stage2!(::KAsGemmK, ::FoldRight, run::RowRun, ops, ws::RowWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    _clear_batches!(vb.s2s, vb.s2z, vb.s2t)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            for j in run.j0:run.j1
                p = col_scratch_pos(ops.bu, run.j0, k, j)
                p == 0 && continue
                jl = j - run.j0 + 1
                push!(vb.s2s, view(ws.S.data, :, :, p, kk, il))
                push!(vb.s2z, tilefactor(ops.bv, k, j))
                push!(vb.s2t, view(ws.T.data, :, kk, :, jl, il))
            end
        end
    end
    isempty(vb.s2s) && return nothing
    return precision_gemm_batched!('N', 'T', one(T), vb.s2s, vb.s2z, zero(T), vb.s2t, compute)
end

# ── FoldLeft (row family, FullGrid) ──────────────────────────────────────────
# The mirror of the FoldRight row family: fold A's `U` into Stage 2 (`T' = U·S`,
# `bm×rB`) and stack B's `Z` in the write-once Stage 3. Chosen only when B is
# TileColMajor (so the Z-stack is contiguous) on a FullGrid interior (no diagonal
# skip). Stage 1 is shared with FoldRight and writes `S` tilewise.

# Stage 2: T'_ikj = U_ik · S_ikj  (bm×rB), tilewise over (i, k, j).
function execute_stage2!(::KAsGemmK, ::FoldLeft, run::RowRun, ops, ws::RowWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    _clear_batches!(vb.s2u, vb.s2s, vb.s2t)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:tiles_per_row(ops.av)
            k = panel_col(ops.av, i, kk)
            for j in run.j0:run.j1
                p = col_scratch_pos(ops.bu, run.j0, k, j)
                p == 0 && continue
                jl = j - run.j0 + 1
                push!(vb.s2u, tilefactor(ops.au, i, k))          # U_ik  [bm, rA]
                push!(vb.s2s, view(ws.S.data, :, :, p, kk, il))  # S_ikj [rA, rB]
                push!(vb.s2t, view(ws.T.data, :, :, kk, jl, il)) # T'    [bm, rB]
            end
        end
    end
    isempty(vb.s2u) && return nothing
    return precision_gemm_batched!('N', 'N', one(T), vb.s2u, vb.s2s, zero(T), vb.s2t, compute)
end

# Stage 3 (write-once): C_ij = β·C_ij + α·Σ_k T'_ikj Z_kj'. Stack over k: the left
# `T'stack_ij` (bm × noff·rB) is a contiguous reshape of our own T' buffer; the right
# `Zstack_j` (bn × noff·rB) is a per-column view of `reshape(bv.data, bn, rB·noff, qn)`
# (contiguous ⟺ B TileColMajor). `'N','T'` gives `T'stack · Zstack'` = [bm, bn]; the
# (rB fast, k slow) column order matches on both sides. Batched over (i, j); β folded.
function execute_stage1!(::KAsSerialLoop{:k}, run::ColumnRun, ops, ws::ColumnWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    _clear_batches!(vb.s1v, vb.s1w, vb.s1s)

    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        Vpanel = view(ws.Vstacked, :, :, k)
        for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
            j = panel_col(ops.bu, k, jpos)
            push!(vb.s1v, Vpanel)
            push!(vb.s1w, tilefactor(ops.bu, k, j))
            push!(vb.s1s, view(ws.S.data, :, :, jx, kx))
        end
    end
    return precision_gemm_batched!('T', 'N', one(T), vb.s1v, vb.s1w, zero(T), vb.s1s, compute)
end

function execute_stage1!(::KAsSerialLoop{:j}, run::ColumnRun, ops, ws::ColumnWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    _clear_batches!(vb.s1jv, vb.s1jw, vb.s1js)

    Jw = run_width(run)
    b = blockdim(ops.bu)
    rB = rankdim(ops.bu)
    rA_noff = size(ws.S.data, 1)
    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        Vpanel = view(ws.Vstacked, :, :, k)
        Wpanel = rowpanel(ops.bu, k)
        Wsub = reshape(view(Wpanel, :, :, run.jpos0:run.jpos1), b, Jw * rB)
        Ssub = reshape(view(ws.S.data, :, :, 1:Jw, kx), rA_noff, Jw * rB)
        push!(vb.s1jv, Vpanel)
        push!(vb.s1jw, Wsub)
        push!(vb.s1js, Ssub)
    end
    return precision_gemm_batched!('T', 'N', one(T), vb.s1jv, vb.s1jw, zero(T), vb.s1js, compute)
end

function execute_stage2!(::KAsSerialLoop, ::FoldRight, run::ColumnRun, ops,
                         ws::ColumnWorkspace, compute)
    T = _ws_eltype(ws)
    vb = ws.batches
    rA_noff = size(ws.S.data, 1)
    b = blockdim(ops.bv)
    _clear_batches!(vb.s2s, vb.s2z, vb.s2t)

    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
            j = panel_col(ops.bv, k, jpos)
            push!(vb.s2s, view(ws.S.data, :, :, jx, kx))
            push!(vb.s2z, tilefactor(ops.bv, k, j))
            push!(vb.s2t, reshape(view(ws.T.data, :, :, :, jx, kx), rA_noff, b))
        end
    end
    return precision_gemm_batched!('N', 'T', one(T), vb.s2s, vb.s2z, zero(T), vb.s2t, compute)
end
