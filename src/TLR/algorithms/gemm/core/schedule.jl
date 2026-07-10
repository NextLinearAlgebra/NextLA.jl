"""
A write-once row run: a rectangular block of output tiles, rows `i0:i1` ×
columns `j0:j1`. All rows are independent (no cross-`i` dependence), so every
stage batches over the whole block; `i` is a batch axis, not a serial loop.
"""
struct RowRun
    i0::Int
    i1::Int
    j0::Int
    j1::Int
end

"""
An accumulate column run: a block of contraction tiles `k0:k1` and a block of B
row-`k` panel positions `jpos0:jpos1` (local; actual columns via
`local_to_col(k, jpos)`). Stages 1/2 are independent over `k`, so they batch over
the whole `k`-block; Stage 3 loops `k` (the reduction) accumulating into `C`.
"""
struct ColumnRun
    k0::Int
    k1::Int
    jpos0::Int
    jpos1::Int
end

"""Row-run iterator (`KAsGemmK`); emits `maxI × maxJ` output-tile blocks over the
`q_m × q_n` output grid."""
struct RowSchedule
    q_m::Int
    q_n::Int
    maxI::Int
    maxJ::Int
end

"""Column-run iterator (`KAsSerialLoop`); emits `maxK × maxJ` `(k, jpos)` blocks over
the `q_c` contraction tiles and `perB_row` B panel positions."""
struct ColumnSchedule
    q_c::Int
    perB_row::Int
    maxK::Int
    maxJ::Int
end

# Interior-product geometry from the logical operands. Extents may be rectangular;
# for a square dense-diagonal interior all of `q_m/q_c/q_n` coincide and every
# `per*` reduces to `nt-1`.
@inline function _interior_geom(ops)
    return (
        q_m = ops.av.qm,                  # A output-row tiles           (== ops.au.qm)
        q_c = ops.av.qn,                  # contraction tiles            (== ops.bw.qm)
        q_n = ops.bw.qn,                  # B output-col tiles           (== ops.bz.qn)
        rA  = rankdim(ops.av),            # A rank
        rB  = rankdim(ops.bw),            # B rank
        b   = blockdim(ops.au),           # square tile size
        perA_row = tiles_per_row(ops.av), # A contraction tiles per output row (row family K-stack)
        perA_col = tiles_per_col(ops.av), # A tiles per contraction column     (column family stacking)
        perB_row = tiles_per_row(ops.bw), # B output-col tiles per contraction row (column jpos width)
    )
end

# Scratch bytes for one batched slice: `S` is r_A·per·r_B and `T` is r_A·per·b,
# so per slice = r_A·per·(b + r_B)·sizeof(T). `per` is the family's stacking depth.
@inline _slice_bytes(rA::Int, per::Int, rB::Int, b::Int, ::Type{T}) where {T} =
    max(rA * per * (b + rB) * sizeof(T), 1)

# Row family: block the `q_m × q_n` output grid into rectangular runs whose tile
# count fits the budget — `maxJ` columns wide, `maxI` rows tall (columns filled
# first). All tiles are independent, so the block is a pure batch.
@inline function _row_block(geom, ::Type{T}, budget::Int) where {T}
    per_col = _slice_bytes(geom.rA, geom.perA_row, geom.rB, geom.b, T)
    maxtiles = clamp(div(budget, per_col), 1, geom.q_m * geom.q_n)
    maxJ = clamp(maxtiles, 1, geom.q_n)
    maxI = clamp(div(maxtiles, maxJ), 1, geom.q_m)
    return maxI, maxJ
end

# Column family: block the `(k, jpos)` slice space into runs whose slice count fits
# the budget — `maxJ` panel positions wide, `maxK` contraction tiles deep (positions
# filled first). Stages 1/2 batch over the whole block; Stage 3 loops `k`.
@inline function _column_block(geom, ::Type{T}, budget::Int) where {T}
    per_slice = _slice_bytes(geom.rA, geom.perA_col, geom.rB, geom.b, T)
    maxslices = clamp(div(budget, per_slice), 1, geom.q_c * geom.perB_row)
    maxJ = clamp(maxslices, 1, geom.perB_row)
    maxK = clamp(div(maxslices, maxJ), 1, geom.q_c)
    return maxK, maxJ
end

# Slices in the widest run a traversal can use: all output tiles (`KAsGemmK`) / all
# `(k, jpos)` slices (`KAsSerialLoop`).
@inline _full_run_tiles(::KAsGemmK, geom) = geom.q_m * geom.q_n
@inline _full_run_tiles(::KAsSerialLoop, geom) = geom.q_c * geom.perB_row

"""
    gemm_workspace_bytes(A, B) -> Int

Minimum `max_workspace` (bytes) at which `gemm!(C, A, B; max_workspace=…)` runs the
off-diagonal product `O_A O_B` at full width. For `KAsGemmK` layouts that is the
entire output tile grid batched in a single run; for `KAsSerialLoop` layouts it is
one full `k`-column per run. Passing at least this many bytes maximises batching;
passing less still works but splits the work into smaller runs.
"""
function gemm_workspace_bytes(A::AbstractTLRMatrix, B::AbstractTLRMatrix)
    ops = logical_operands(A, B)
    geom = _interior_geom(ops)
    placement = k_axis_schedule(stride1_axis_left(A), stride1_axis_right(B))
    per = placement isa KAsGemmK ? geom.perA_row : geom.perA_col
    return _slice_bytes(geom.rA, per, geom.rB, geom.b, eltype(ops.av.data)) *
           _full_run_tiles(placement, geom)
end

"""
    runs(placement, ops, budget) -> schedule

Build the budget-sized run iterator selected by the reduction placement:
`RowSchedule` for `KAsGemmK`, `ColumnSchedule` for `KAsSerialLoop`.
"""
@inline function runs(placement::KAsGemmK, ops, budget::Int)
    geom = _interior_geom(ops)
    maxI, maxJ = _row_block(geom, eltype(ops.av.data), budget)
    return RowSchedule(geom.q_m, geom.q_n, maxI, maxJ)
end

@inline function runs(placement::KAsSerialLoop, ops, budget::Int)
    geom = _interior_geom(ops)
    maxK, maxJ = _column_block(geom, eltype(ops.av.data), budget)
    return ColumnSchedule(geom.q_c, geom.perB_row, maxK, maxJ)
end

function Base.iterate(s::RowSchedule, state=(1, 1))
    i0, j0 = state
    i0 > s.q_m && return nothing
    i1 = min(i0 + s.maxI - 1, s.q_m)
    j1 = min(j0 + s.maxJ - 1, s.q_n)
    next = j1 == s.q_n ? (i1 + 1, 1) : (i0, j1 + 1)
    return RowRun(i0, i1, j0, j1), next
end

function Base.iterate(s::ColumnSchedule, state=(1, 1))
    k0, jpos0 = state
    k0 > s.q_c && return nothing
    k1 = min(k0 + s.maxK - 1, s.q_c)
    jpos1 = min(jpos0 + s.maxJ - 1, s.perB_row)
    next = jpos1 == s.perB_row ? (k1 + 1, 1) : (k0, jpos1 + 1)
    return ColumnRun(k0, k1, jpos0, jpos1), next
end

"""Number of output columns (row run) / panel positions (column run) in a run."""
@inline run_width(run::RowRun) = run.j1 - run.j0 + 1
@inline run_width(run::ColumnRun) = run.jpos1 - run.jpos0 + 1

"""Number of output rows batched in a row run / contraction tiles in a column run."""
@inline run_height(run::RowRun) = run.i1 - run.i0 + 1
@inline run_kdepth(run::ColumnRun) = run.k1 - run.k0 + 1

# Preallocated `Vector` of a concrete view type, sized once and refilled per run
# via `empty!`/`push!` (capacity reused, no per-run allocation).
@inline function _batchvec(x, n::Int)
    v = Vector{typeof(x)}()
    sizehint!(v, n)
    return v
end

"""Scratch + batch buffers for the row (write-once) family; `Ustacked` is the
K-stacked left operand for Stage 3."""
struct RowWorkspace{SB,TB,UB,BV}
    S::ScratchS{SB}
    T::ScratchT{TB}
    Ustacked::UB
    batches::BV
end

"""Scratch + batch buffers for the column (accumulate) family; `Vstacked` is the
i-fused Stage-1 left operand, `Ufactored` the per-tile Stage-3 left operand."""
struct ColumnWorkspace{SB,TB,VB,UB,BV}
    S::ScratchS{SB}
    T::ScratchT{TB}
    Vstacked::VB
    Ufactored::UB
    batches::BV
end

function _row_batches(ops, C, Sbuf, Tbuf, Uall, geom, maxI::Int, maxJ::Int)
    b = geom.b
    noff = geom.perA_row
    rA = geom.rA
    rB = geom.rB
    ntrip = noff * maxJ * maxI          # Stage 1/2 tilewise batch over (i, k, j)
    Ssample = view(Sbuf, :, :, 1, 1, 1)               # S[:,:,p,kk,il]
    Tsample = view(Tbuf, :, 1, :, 1, 1)
    Sfused = reshape(view(Sbuf, :, :, 1:maxJ, 1, 1), rA, maxJ * rB)
    Wfused = reshape(view(ops.bw.data, :, :, 1:maxJ), b, maxJ * rB)
    Ustack = view(Uall, :, :, 1)
    Tstack = reshape(view(Tbuf, :, :, :, 1:maxJ, 1), noff * rA, maxJ * b)
    Cblock = view(C, 1:b, 1:maxJ * b)
    return (
        s1v = _batchvec(view(ops.av.data, :, :, 1), ntrip),
        s1w = _batchvec(view(ops.bw.data, :, :, 1), ntrip),
        s1s = _batchvec(Ssample, ntrip),
        s1jv = _batchvec(view(ops.av.data, :, :, 1), noff * maxI),   # fused: batch over (i,k)
        s1jw = _batchvec(Wfused, noff * maxI),
        s1js = _batchvec(Sfused, noff * maxI),
        s2s = _batchvec(Ssample, ntrip),
        s2z = _batchvec(view(ops.bz.data, :, :, 1), ntrip),
        s2t = _batchvec(Tsample, ntrip),
        s3u = _batchvec(Ustack, maxI),   # Stage 3 batches over i
        s3t = _batchvec(Tstack, maxI),
        s3c = _batchvec(Cblock, maxI),
    )
end

function _column_batches(ops, C, Sbuf, Tbuf, Vall, Uall, geom, maxK::Int, maxJ::Int)
    b = geom.b
    noff = geom.perA_col
    rA = geom.rA
    rB = geom.rB
    nkj = maxK * maxJ                    # Stage 1/2 batch over (k, jpos)
    n3 = noff * maxJ                     # Stage 3 batch over (i, jpos) per k
    Vpanel = view(Vall, :, :, 1)
    Wfused = reshape(view(ops.bw.data, :, :, 1:maxJ), b, maxJ * rB)
    Sfused = reshape(view(Sbuf, :, :, 1:maxJ, 1), rA * noff, maxJ * rB)
    return (
        s1v = _batchvec(Vpanel, nkj),
        s1w = _batchvec(view(ops.bw.data, :, :, 1), nkj),
        s1s = _batchvec(view(Sbuf, :, :, 1, 1), nkj),
        s1jv = _batchvec(Vpanel, maxK),
        s1jw = _batchvec(Wfused, maxK),
        s1js = _batchvec(Sfused, maxK),
        s2s = _batchvec(view(Sbuf, :, :, 1, 1), nkj),
        s2z = _batchvec(view(ops.bz.data, :, :, 1), nkj),
        s2t = _batchvec(reshape(view(Tbuf, :, :, :, 1, 1), rA * noff, b), nkj),
        s3u = _batchvec(view(Uall, :, :, 1, 1), n3),
        s3t = _batchvec(view(Tbuf, :, 1, :, 1, 1), n3),
        s3c = _batchvec(view(C, 1:b, 1:b), n3),
    )
end

"""
    allocate_workspace(placement, ops, C, budget) -> RowWorkspace | ColumnWorkspace

Allocate S/T scratch and reusable batch buffers once per hard-term call, sized
to the budgeted run width for the given reduction placement. Geometry comes from
the operands, so this serves both `SkipDiag` and `FullGrid` interiors.
"""
function allocate_workspace(placement::KAsGemmK, ops, C::AbstractMatrix, budget::Int)
    geom = _interior_geom(ops)
    T = eltype(ops.av.data)
    backend = get_backend(ops.av.data)
    b = geom.b
    noff = geom.perA_row
    rA = geom.rA
    rB = geom.rB
    maxI, maxJ = _row_block(geom, T, budget)
    Uall = reshape(ops.au.data, b, rA * noff, geom.q_m)
    # S[:,:,p,kk,il]: p = position within the run's column block, laid out so a fused
    # per-(i,k) Stage-1 GEMM writes a contiguous [rA, len·rB] slice.
    Sbuf = allocate(backend, T, rA, rB, maxJ, noff, maxI)
    Tbuf = allocate(backend, T, rA, noff, b, maxJ, maxI)
    return RowWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Uall,
                        _row_batches(ops, C, Sbuf, Tbuf, Uall, geom, maxI, maxJ))
end

function allocate_workspace(placement::KAsSerialLoop, ops, C::AbstractMatrix, budget::Int)
    geom = _interior_geom(ops)
    T = eltype(ops.av.data)
    backend = get_backend(ops.av.data)
    b = geom.b
    noff = geom.perA_col
    rA = geom.rA
    rB = geom.rB
    maxK, maxJ = _column_block(geom, T, budget)
    Vall = reshape(ops.av.data, b, rA * noff, geom.q_c)
    Uall = reshape(ops.au.data, b, rA, noff, geom.q_c)
    Sbuf = allocate(backend, T, rA * noff, rB, maxJ, maxK)
    Tbuf = allocate(backend, T, rA, noff, b, maxJ, maxK)
    return ColumnWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Vall, Uall,
                           _column_batches(ops, C, Sbuf, Tbuf, Vall, Uall, geom, maxK, maxJ))
end

"""Scale the dense output by `beta` in place before accumulating product terms."""
@inline function _scale_output!(C, beta)
    T = eltype(C)
    if iszero(beta)
        fill!(C, zero(T))
    elseif !isone(beta)
        C .*= T(beta)
    end
    return C
end
