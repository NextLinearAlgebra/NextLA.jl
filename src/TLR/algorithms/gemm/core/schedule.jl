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

"""Row-run iterator (`KAsGemmK`); emits `maxI × maxJ` output-tile blocks per run."""
struct RowSchedule{M}
    matrix::M
    maxI::Int
    maxJ::Int
end

"""Column-run iterator (`KAsSerialLoop`); emits `maxK × maxJ` `(k, jpos)` blocks."""
struct ColumnSchedule{M}
    matrix::M
    maxK::Int
    maxJ::Int
end

# Scratch bytes attributable to one output column of a run: `S` is r_A·noff·r_B
# and `T` is r_A·noff·b, so per column = r_A·noff·(b + r_B)·sizeof(T).
@inline function _per_col_bytes(A::TLRMatrix{<:Any,T}, B::TLRMatrix) where {T}
    _, nt = _interior_grid(A)
    noff = nt - 1
    b = nominal_tile_size(A, 1)
    return max(maxrank(A) * noff * (b + maxrank(B)) * sizeof(T), 1)
end

# Row family: block the whole (i, j) tile grid into rectangular runs whose tile
# count fits the budget — `maxJ` columns wide, `maxI` rows tall (columns filled
# first). All tiles are independent, so the block is a pure batch.
@inline function _row_block(A::TLRMatrix, B::TLRMatrix, budget::Int)
    mt, nt = _interior_grid(A)
    maxtiles = clamp(div(budget, _per_col_bytes(A, B)), 1, mt * nt)
    maxJ = clamp(maxtiles, 1, nt)
    maxI = clamp(div(maxtiles, maxJ), 1, mt)
    return maxI, maxJ
end

# Column family: block the `(k, jpos)` slice space into runs whose slice count
# fits the budget — `maxJ` panel positions wide, `maxK` contraction tiles deep
# (positions filled first). Stages 1/2 batch over the whole block; Stage 3 loops k.
@inline function _column_block(A::TLRMatrix, B::TLRMatrix, budget::Int)
    _, nt = _interior_grid(A)
    noff = nt - 1
    maxslices = clamp(div(budget, _per_col_bytes(A, B)), 1, nt * noff)
    maxJ = clamp(maxslices, 1, noff)
    maxK = clamp(div(maxslices, maxJ), 1, nt)
    return maxK, maxJ
end

# Slices in the widest run a traversal can use: the whole grid either way — all
# (i,j) tiles for `KAsGemmK`, all (k,jpos) slices for `KAsSerialLoop`.
@inline _full_run_tiles(::KAsGemmK, mt::Int, nt::Int) = mt * nt
@inline _full_run_tiles(::KAsSerialLoop, mt::Int, nt::Int) = nt * (nt - 1)

"""
    gemm_workspace_bytes(A, B) -> Int

Minimum `max_workspace` (bytes) at which `gemm!(C, A, B; max_workspace=…)` runs the
off-diagonal product `O_A O_B` at full width. For `KAsGemmK` layouts that is the
entire output tile grid batched in a single run; for `KAsSerialLoop` layouts it is
one full `k`-column per run. Passing at least this many bytes maximises batching;
passing less still works but splits the work into smaller runs.
"""
function gemm_workspace_bytes(A::TLRMatrix, B::TLRMatrix)
    mt, nt = _interior_grid(A)
    placement = k_axis_schedule(stride1_axis_left(A), stride1_axis_right(B))
    return _per_col_bytes(A, B) * _full_run_tiles(placement, mt, nt)
end

"""
    runs(placement, A, B, budget) -> schedule

Build the budget-sized run iterator selected by the reduction placement:
`RowSchedule` for `KAsGemmK`, `ColumnSchedule` for `KAsSerialLoop`.
"""
@inline runs(::KAsGemmK, A::TLRMatrix, B::TLRMatrix, budget::Int) =
    RowSchedule(A, _row_block(A, B, budget)...)

@inline runs(::KAsSerialLoop, A::TLRMatrix, B::TLRMatrix, budget::Int) =
    ColumnSchedule(A, _column_block(A, B, budget)...)

function Base.iterate(s::RowSchedule, state=(1, 1))
    i0, j0 = state
    mt, nt = _interior_grid(s.matrix)
    i0 > mt && return nothing
    i1 = min(i0 + s.maxI - 1, mt)
    j1 = min(j0 + s.maxJ - 1, nt)
    next = j1 == nt ? (i1 + 1, 1) : (i0, j1 + 1)
    return RowRun(i0, i1, j0, j1), next
end

function Base.iterate(s::ColumnSchedule, state=(1, 1))
    k0, jpos0 = state
    _, nt = _interior_grid(s.matrix)
    noff = nt - 1
    k0 > nt && return nothing
    k1 = min(k0 + s.maxK - 1, nt)
    jpos1 = min(jpos0 + s.maxJ - 1, noff)
    next = jpos1 == noff ? (k1 + 1, 1) : (k0, jpos1 + 1)
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

function _row_batches(A, B, C, Sbuf, Tbuf, Uall, maxI::Int, maxJ::Int)
    b = nominal_tile_size(A, 1)
    noff = _interior_grid(A)[2] - 1
    rA = maxrank(A)
    rB = maxrank(B)
    ntrip = noff * maxJ * maxI          # Stage 1/2 tilewise batch over (i, k, j)
    Ssample = view(Sbuf, :, :, 1, 1, 1)               # S[:,:,p,kk,il]
    Tsample = view(Tbuf, :, 1, :, 1, 1)
    Sfused = reshape(view(Sbuf, :, :, 1:maxJ, 1, 1), rA, maxJ * rB)
    Wfused = reshape(view(B.int_U, :, :, 1:maxJ), b, maxJ * rB)
    Ustack = view(Uall, :, :, 1)
    Tstack = reshape(view(Tbuf, :, :, :, 1:maxJ, 1), noff * rA, maxJ * b)
    Cblock = view(C, 1:b, 1:maxJ * b)
    return (
        s1v = _batchvec(view(A.int_V, :, :, 1), ntrip),
        s1w = _batchvec(view(B.int_U, :, :, 1), ntrip),
        s1s = _batchvec(Ssample, ntrip),
        s1jv = _batchvec(view(A.int_V, :, :, 1), noff * maxI),   # fused: batch over (i,k)
        s1jw = _batchvec(Wfused, noff * maxI),
        s1js = _batchvec(Sfused, noff * maxI),
        s2s = _batchvec(Ssample, ntrip),
        s2z = _batchvec(view(B.int_V, :, :, 1), ntrip),
        s2t = _batchvec(Tsample, ntrip),
        s3u = _batchvec(Ustack, maxI),   # Stage 3 batches over i
        s3t = _batchvec(Tstack, maxI),
        s3c = _batchvec(Cblock, maxI),
    )
end

function _column_batches(A, B, C, Sbuf, Tbuf, Vall, Uall, maxK::Int, maxJ::Int)
    b = nominal_tile_size(A, 1)
    noff = _interior_grid(A)[1] - 1
    rA = maxrank(A)
    rB = maxrank(B)
    nkj = maxK * maxJ                    # Stage 1/2 batch over (k, jpos)
    n3 = noff * maxJ                     # Stage 3 batch over (i, jpos) per k
    Vpanel = view(Vall, :, :, 1)
    Wfused = reshape(view(B.int_U, :, :, 1:maxJ), b, maxJ * rB)
    Sfused = reshape(view(Sbuf, :, :, 1:maxJ, 1), rA * noff, maxJ * rB)
    return (
        s1v = _batchvec(Vpanel, nkj),
        s1w = _batchvec(view(B.int_U, :, :, 1), nkj),
        s1s = _batchvec(view(Sbuf, :, :, 1, 1), nkj),
        s1jv = _batchvec(Vpanel, maxK),
        s1jw = _batchvec(Wfused, maxK),
        s1js = _batchvec(Sfused, maxK),
        s2s = _batchvec(view(Sbuf, :, :, 1, 1), nkj),
        s2z = _batchvec(view(B.int_V, :, :, 1), nkj),
        s2t = _batchvec(reshape(view(Tbuf, :, :, :, 1, 1), rA * noff, b), nkj),
        s3u = _batchvec(view(Uall, :, :, 1, 1), n3),
        s3t = _batchvec(view(Tbuf, :, 1, :, 1, 1), n3),
        s3c = _batchvec(view(C, 1:b, 1:b), n3),
    )
end

"""
    allocate_workspace(placement, A, B, C, budget) -> RowWorkspace | ColumnWorkspace

Allocate S/T scratch and reusable batch buffers once per hard-term call, sized
to the budgeted run width for the given reduction placement.
"""
function allocate_workspace(::KAsGemmK, A::TLRMatrix{<:Any,T}, B::TLRMatrix,
                            C::AbstractMatrix, budget::Int) where {T}
    mt, nt = _interior_grid(A)
    b = nominal_tile_size(A, 1)
    noff = nt - 1
    rA = maxrank(A)
    rB = maxrank(B)
    maxI, maxJ = _row_block(A, B, budget)
    Uall = reshape(A.int_U, b, rA * noff, mt)
    # S[:,:,p,kk,il]: p = off-diagonal position within the run's column block, laid
    # out so a fused per-(i,k) Stage-1 GEMM writes a contiguous [rA, len·rB] slice.
    Sbuf = allocate(A.backend, T, rA, rB, maxJ, noff, maxI)
    Tbuf = allocate(A.backend, T, rA, noff, b, maxJ, maxI)
    return RowWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Uall,
                        _row_batches(A, B, C, Sbuf, Tbuf, Uall, maxI, maxJ))
end

function allocate_workspace(::KAsSerialLoop, A::TLRMatrix{<:Any,T}, B::TLRMatrix,
                            C::AbstractMatrix, budget::Int) where {T}
    _, nt = _interior_grid(A)
    b = nominal_tile_size(A, 1)
    noff = nt - 1
    rA = maxrank(A)
    rB = B.maxrank
    maxK, maxJ = _column_block(A, B, budget)
    Vall = reshape(A.int_V, b, rA * noff, nt)
    Uall = reshape(A.int_U, b, rA, noff, nt)
    Sbuf = allocate(A.backend, T, rA * noff, rB, maxJ, maxK)
    Tbuf = allocate(A.backend, T, rA, noff, b, maxJ, maxK)
    return ColumnWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Vall, Uall,
                           _column_batches(A, B, C, Sbuf, Tbuf, Vall, Uall, maxK, maxJ))
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
