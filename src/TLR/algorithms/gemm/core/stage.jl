# Stage descriptors

struct StageDescriptor{StageT<:GemmStage,PlacementT<:KAxisSchedule,RunT,OpsT,WST,CT,AlphaT,BlockingT}
    stage::StageT
    placement::PlacementT
    run::RunT
    ops::OpsT
    workspace::WST
    C::CT
    alpha::AlphaT
    free_axis_schedule::BlockingT
end

@inline stage1(placement::KAxisSchedule, run, ops, ws) =
    StageDescriptor(Stage1(), placement, run, ops, ws, nothing, nothing, free_axis_schedule(placement))

@inline stage2(placement::KAxisSchedule, run, ops, ws) =
    StageDescriptor(Stage2(), placement, run, ops, ws, nothing, nothing, nothing)

@inline stage3(placement::KAxisSchedule, run, ops, ws, C::AbstractMatrix, alpha) =
    StageDescriptor(Stage3(), placement, run, ops, ws, C, alpha, nothing)

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

# Off-diagonal position of output column `j` within the run's block `[j0:j1]` for
# contraction tile `k` (the diagonal column `j=k` is skipped, so columns past `k`
# shift down by one). `S` is indexed by this `p`; `T` stays indexed by `jl`.
@inline _offdiag_pos(j0::Int, k::Int, j::Int) = (j - j0 + 1) - ((j0 <= k) & (k < j) ? 1 : 0)
# Panel-local index (in `rowpanel(k)`) of absolute column `j ≠ k`.
@inline _panel_local(k::Int, j::Int) = j < k ? j : j - 1

function execute_stage!(d::StageDescriptor{Stage1,<:KAsGemmK,<:Any,<:Any,<:Any,<:Any,<:Any,FreeAsBatch})
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s1v, vb.s1w, vb.s1s)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:ops.av.noff
            k = local_to_col(i, kk)
            for j in run.j0:run.j1
                j == k && continue
                p = _offdiag_pos(run.j0, k, j)
                push!(vb.s1v, tilefactor(ops.av, i, k))
                push!(vb.s1w, tilefactor(ops.bw, k, j))
                push!(vb.s1s, view(ws.S.data, :, :, p, kk, il))
            end
        end
    end
    isempty(vb.s1v) && return nothing
    return gemm_batched!('T', 'N', one(T), vb.s1v, vb.s1w, zero(T), vb.s1s)
end

# Fused Stage 1 for `(k,j)`: B stride-1 `:j` makes `rowpanel(k)` contiguous over
# `j`, so the block's off-diagonal columns form a single wide right operand and
# `j` folds into N — one GEMM per `(i,k)` instead of one per `(i,k,j)`.
function execute_stage!(d::StageDescriptor{Stage1,<:KAsGemmK,<:Any,<:Any,<:Any,<:Any,<:Any,JAsGemmN})
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
        for kk in 1:ops.av.noff
            k = local_to_col(i, kk)
            jf = run.j0 == k ? run.j0 + 1 : run.j0          # first off-diagonal column
            je = run.j1 == k ? run.j1 - 1 : run.j1          # last off-diagonal column
            jf > je && continue                              # block is only the diagonal
            lrange = _panel_local(k, jf):_panel_local(k, je) # contiguous panel slice
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
    return gemm_batched!('T', 'N', one(T), vb.s1jv, vb.s1jw, zero(T), vb.s1js)
end

function execute_stage!(d::StageDescriptor{Stage2,<:KAsGemmK})
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s2s, vb.s2z, vb.s2t)

    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        for kk in 1:ops.av.noff
            k = local_to_col(i, kk)
            for j in run.j0:run.j1
                j == k && continue
                p = _offdiag_pos(run.j0, k, j)
                jl = j - run.j0 + 1
                push!(vb.s2s, view(ws.S.data, :, :, p, kk, il))
                push!(vb.s2z, tilefactor(ops.bz, k, j))
                push!(vb.s2t, view(ws.T.data, :, kk, :, jl, il))
            end
        end
    end
    isempty(vb.s2s) && return nothing
    return gemm_batched!('N', 'T', one(T), vb.s2s, vb.s2z, zero(T), vb.s2t)
end

function execute_stage!(d::StageDescriptor{Stage3,<:KAsGemmK})
    T = _ws_eltype(d.workspace)
    run = d.run
    ws = d.workspace
    b = size(ws.Ustacked, 1)                 # nominal tile size (interior tiles are b×b)
    noff = size(ws.T.data, 2)
    rA = size(ws.T.data, 1)
    Jw = run_width(run)
    vb = ws.batches
    _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
    @inbounds for (il, i) in enumerate(run.i0:run.i1)
        Tstack = reshape(view(ws.T.data, :, :, :, 1:Jw, il), noff * rA, Jw * b)
        push!(vb.s3u, view(ws.Ustacked, :, :, i))
        push!(vb.s3t, Tstack)
        push!(vb.s3c, dense_rowblock(d.C, b, i, run.j0, run.j1))
    end
    return gemm_batched!('N', 'N', T(d.alpha), vb.s3u, vb.s3t, one(T), vb.s3c)
end

function execute_stage!(d::StageDescriptor{Stage1,<:KAsSerialLoop,<:Any,<:Any,<:Any,<:Any,<:Any,IAsGemmM})
    T = _ws_eltype(d.workspace)
    run = d.run
    ops = d.ops
    ws = d.workspace
    vb = ws.batches
    _clear_batches!(vb.s1v, vb.s1w, vb.s1s)

    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        Vpanel = view(ws.Vstacked, :, :, k)
        for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
            j = local_to_col(k, jpos)
            push!(vb.s1v, Vpanel)
            push!(vb.s1w, tilefactor(ops.bw, k, j))
            push!(vb.s1s, view(ws.S.data, :, :, jx, kx))
        end
    end
    return gemm_batched!('T', 'N', one(T), vb.s1v, vb.s1w, zero(T), vb.s1s)
end

function execute_stage!(d::StageDescriptor{Stage1,<:KAsSerialLoop,<:Any,<:Any,<:Any,<:Any,<:Any,IJAsGemmMN})
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
    return gemm_batched!('T', 'N', one(T), vb.s1jv, vb.s1jw, zero(T), vb.s1js)
end

function execute_stage!(d::StageDescriptor{Stage2,<:KAsSerialLoop})
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
            j = local_to_col(k, jpos)
            push!(vb.s2s, view(ws.S.data, :, :, jx, kx))
            push!(vb.s2z, tilefactor(ops.bz, k, j))
            push!(vb.s2t, reshape(view(ws.T.data, :, :, :, jx, kx), rA_noff, b))
        end
    end
    return gemm_batched!('N', 'T', one(T), vb.s2s, vb.s2z, zero(T), vb.s2t)
end

function execute_stage!(d::StageDescriptor{Stage3,<:KAsSerialLoop})
    T = _ws_eltype(d.workspace)
    run = d.run
    ws = d.workspace
    b = size(ws.Ufactored, 1)                # nominal tile size (interior tiles are b×b)
    vb = ws.batches
    noff = size(ws.Ufactored, 3)
    # `k` is the reduction axis: loop it (one accumulate GEMM per k), batching the
    # free axes (i, jpos). Different k write the same C_ij, so they cannot share a
    # batch — but successive launches accumulate with β = 1.
    @inbounds for (kx, k) in enumerate(run.k0:run.k1)
        _clear_batches!(vb.s3u, vb.s3t, vb.s3c)
        for li in 1:noff
            i = local_to_col(k, li)
            for (jx, jpos) in enumerate(run.jpos0:run.jpos1)
                j = local_to_col(k, jpos)
                push!(vb.s3u, view(ws.Ufactored, :, :, li, k))
                push!(vb.s3t, view(ws.T.data, :, li, :, jx, kx))
                push!(vb.s3c, dense_tile(d.C, b, i, j))
            end
        end
        gemm_batched!('N', 'N', T(d.alpha), vb.s3u, vb.s3t, one(T), vb.s3c)
    end
    return nothing
end
