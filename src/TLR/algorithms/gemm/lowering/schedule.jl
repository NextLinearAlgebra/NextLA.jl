# Run geometry, workspace scheduling, and output initialization.

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
`qm × qn` output grid."""
struct RowSchedule
    qm::Int
    qn::Int
    maxI::Int
    maxJ::Int
end

"""Column-run iterator (`KAsSerialLoop`); emits `maxK × maxJ` `(k, jpos)` blocks over
the `qk` contraction tiles and `perB_row` B panel positions."""
struct ColumnSchedule
    qk::Int
    perB_row::Int
    maxK::Int
    maxJ::Int
end

"""
    RegularGeometry{T}

The sizes the scheduler needs to budget a run of one structured contraction, with the
operand storage type `T` as a **type parameter**.

`T` is a type parameter rather than a field on purpose. Scratch is allocated as
`allocate(backend, eltype(geom), …)`, and `allocate`'s result type is only inferable when
the element type is known to the compiler. Carrying it as a field (`Tin::DataType`) makes
the geometry a runtime value, so `allocate` infers `Array{<:Any}`, the workspace type and
its batch-view eltypes go abstract, and the staged loops allocate ~1.4 KB per run through
dynamic dispatch — with correctness tests still passing. `test/TLR/gemm_core.jl` pins the
inferability with `@inferred`; note `isconcretetype(typeof(ws))` does **not** test this
(it is true of any runtime value).
"""
struct RegularGeometry{T}
    qm::Int          # A output-row tiles
    qk::Int          # contraction tiles
    qn::Int          # B output-col tiles
    rA::Int           # A rank
    rB::Int           # B rank
    bm::Int           # A row / C row tile height
    bk::Int           # contraction tile size
    bn::Int           # B col / C col tile width (T spatial extent)
    perA_row::Int     # A contraction tiles per output row (row family K-stack)
    perA_col::Int     # A tiles per contraction column     (column family stacking)
    perB_row::Int     # B output-col tiles per contraction row (column jpos width)
end

Base.eltype(::RegularGeometry{T}) where {T} = T
Base.eltype(::Type{RegularGeometry{T}}) where {T} = T

"""
    regular_geometry(qm, qk, qn, ops) -> RegularGeometry

The sizes needed to budget a direct low-rank run. Extents are explicit arguments; ranks,
block dimensions, element type, and stacking depths come from canonical factor accessors.

Tiles may be rectangular, so three distinct block sizes are carried: `bm` (A row / C row
height, from the outer factor), `bk` (contraction tile size, from the inner factor), and
`bn` (B col / C col width, from B's inner factor, also the T scratch's spatial extent).
For a square interior all three coincide, and for a square dense-diagonal interior
`qm/qk/qn` coincide too and every `per*` reduces to `nt-1`.

The operand storage type rides as the `RegularGeometry` type parameter (read via
`eltype`), because the scheduler budgets *bytes* and `allocate` must stay inferable.

The `per*` stacking depths are asked of the factor accessors: an interior grid uses its
`GridKind`, a panel its live axis, and a corner degenerately uses 1.
"""
@inline function regular_geometry(qm::Int, qk::Int, qn::Int, ops)
    return RegularGeometry{eltype(ops.av.data)}(
        qm, qk, qn,
        rankdim(ops.av), rankdim(ops.bu),
        blockdim(ops.au), blockdim(ops.av), blockdim(ops.bv),
        tiles_per_row(ops.av), tiles_per_col(ops.av), tiles_per_row(ops.bu))
end

"""Geometry of the interior contraction of `op(A)` × `op(B)`."""
@inline function interior_geometry(A::LogicalTLROperand, B::LogicalTLROperand)
    qm, qk = regular_grid_size(A)
    qkB, qn = regular_grid_size(B)
    qk == qkB || throw(DimensionMismatch("interior contraction grids do not match"))
    return regular_geometry(qm, qk, qn, logical_operands(A, B))
end

# Scratch bytes for one batched slice: `S` is r_A·per·r_B and `T` is r_A·per·b_n,
# so per slice = r_A·per·(b_n + r_B)·sizeof(T). `per` is the family's stacking depth.
@inline _slice_bytes(rA::Int, per::Int, rB::Int, bn::Int, ::Type{T}) where {T} =
    max(rA * per * (bn + rB) * sizeof(T), 1)

@inline _row_slice_bytes(geom, ::FoldRight) =
    _slice_bytes(geom.rA, geom.perA_row, geom.rB, geom.bn, eltype(geom))
@inline _row_slice_bytes(geom, ::FoldLeft) =
    max(geom.perA_row * geom.rB * (geom.rA + geom.bm) * sizeof(eltype(geom)), 1)

# Row family: block the `qm × qn` output grid into rectangular runs whose tile
# count fits the budget — `maxJ` columns wide, `maxI` rows tall (columns filled
# first). All tiles are independent, so the block is a pure batch.
@inline function _row_block(geom, budget::Int, fold::FoldSide)
    per_col = _row_slice_bytes(geom, fold)
    maxtiles = clamp(div(budget, per_col), 1, geom.qm * geom.qn)
    maxJ = clamp(maxtiles, 1, geom.qn)
    maxI = clamp(div(maxtiles, maxJ), 1, geom.qm)
    return maxI, maxJ
end

# Column family: block the `(k, jpos)` slice space into runs whose slice count fits
# the budget — `maxJ` panel positions wide, `maxK` contraction tiles deep (positions
# filled first). Stages 1/2 batch over the whole block; Stage 3 loops `k`.
@inline function _column_block(geom, budget::Int)
    per_slice = _slice_bytes(geom.rA, geom.perA_col, geom.rB, geom.bn, eltype(geom))
    maxslices = clamp(div(budget, per_slice), 1, geom.qk * geom.perB_row)
    maxJ = clamp(maxslices, 1, geom.perB_row)
    maxK = clamp(div(maxslices, maxJ), 1, geom.qk)
    return maxK, maxJ
end

# Slices in the widest run a traversal can use: all output tiles (`KAsGemmK`) / all
# `(k, jpos)` slices (`KAsSerialLoop`).
@inline _full_run_tiles(::KAsGemmK, geom) = geom.qm * geom.qn
@inline _full_run_tiles(::KAsSerialLoop, geom) = geom.qk * geom.perB_row

"""
    choose_fold(ops) -> FoldSide

Pick the Stage-2/3 association from the leaves' contiguous iterators so the reduction
becomes a write-once fused Stage 3 without transposing either operand (see `FoldSide`).
`FoldLeft` stacks B's `Z` and is write-once when its leaf is contiguous along `k`.
Reassociation additionally requires complete reduction stacks: `FullGrid` interiors and
panels qualify; `SkipDiag` does not because its missing diagonal breaks a plain reshape.
When both sides admit a write-once stack, the smaller intermediate breaks the tie
(`bm·rB` vs `rA·bn`).
"""
@inline function choose_fold(ops)
    complete_k_stack(ops.av) || return FoldRight()
    complete_k_stack(ops.bv) || return FoldRight()
    stride1_axis_right(ops.bv) isa Stride1Axis{:k} || return FoldRight()
    stride1_axis_left(ops.au) isa Stride1Axis{:k} || return FoldLeft()
    # Both write-once → smaller intermediate wins (`bm·rB` vs `rA·bn`). These are storage
    # sizes, not extents, so they come straight off the operands — the fold is a layout
    # decision and needs no domain.
    return blockdim(ops.au) * rankdim(ops.bu) < rankdim(ops.av) * blockdim(ops.bv) ?
           FoldLeft() : FoldRight()
end

# Reduction → hardware-axis placement, from the operands' effective tile orders.
# `FoldRight` keys on op(A) (stacks op(A)'s `U`). `FoldLeft` is only ever chosen when
# op(B) is `TileColMajor`, which makes op(B)'s `Z` k-stack contiguous — always the
# write-once row family with tilewise Stage 1.
@inline placement_for_fold(::FoldRight, ops) =
    k_axis_schedule(stride1_axis_left(ops.au), stride1_axis_right(ops.bv))
@inline placement_for_fold(::FoldLeft, ops) = KAsGemmK{:k}()

"""
    runs(placement, geom, budget, fold) -> schedule

Build the budget-sized run iterator selected by the reduction placement:
`RowSchedule` for `KAsGemmK`, `ColumnSchedule` for `KAsSerialLoop`.

Takes a `geometry` rather than the operand bundle, so the run space is derived from the
contraction's domain — the same path a panel or corner term will use once its extents
are spans of one tile.
"""
@inline function runs(placement::KAsGemmK, geom, budget::Int, fold::FoldSide)
    maxI, maxJ = _row_block(geom, budget, fold)
    return RowSchedule(geom.qm, geom.qn, maxI, maxJ)
end

@inline function runs(placement::KAsSerialLoop, geom, budget::Int, ::FoldSide)
    maxK, maxJ = _column_block(geom, budget)
    return ColumnSchedule(geom.qk, geom.perB_row, maxK, maxJ)
end

function Base.iterate(s::RowSchedule, state=(1, 1))
    i0, j0 = state
    i0 > s.qm && return nothing
    i1 = min(i0 + s.maxI - 1, s.qm)
    j1 = min(j0 + s.maxJ - 1, s.qn)
    next = j1 == s.qn ? (i1 + 1, 1) : (i0, j1 + 1)
    return RowRun(i0, i1, j0, j1), next
end

function Base.iterate(s::ColumnSchedule, state=(1, 1))
    k0, jpos0 = state
    k0 > s.qk && return nothing
    k1 = min(k0 + s.maxK - 1, s.qk)
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

@inline function _row_stage1_batches(::KAsGemmK{:k}, ops, Ssample, Sfused,
                                     Wfused, ntrip, noff, maxI)
    return (
        s1v = _batchvec(view(ops.av.data, :, :, 1), ntrip),
        s1w = _batchvec(view(ops.bu.data, :, :, 1), ntrip),
        s1s = _batchvec(Ssample, ntrip),
    )
end

@inline function _row_stage1_batches(::KAsGemmK{:j}, ops, Ssample, Sfused,
                                     Wfused, ntrip, noff, maxI)
    return (
        s1jv = _batchvec(view(ops.av.data, :, :, 1), noff * maxI),
        s1jw = _batchvec(Wfused, noff * maxI),
        s1js = _batchvec(Sfused, noff * maxI),
    )
end

function _row_batches(placement::KAsGemmK, ops, C, Sbuf, Tbuf, Uall,
                      geom, maxI::Int, maxJ::Int)
    bm = geom.bm; bk = geom.bk; bn = geom.bn
    noff = geom.perA_row
    rA = geom.rA
    rB = geom.rB
    ntrip = noff * maxJ * maxI          # Stage 1/2 tilewise batch over (i, k, j)
    Ssample = view(Sbuf, :, :, 1, 1, 1)               # S[:,:,p,kk,il]
    Tsample = view(Tbuf, :, 1, :, 1, 1)
    Sfused = reshape(view(Sbuf, :, :, 1:maxJ, 1, 1), rA, maxJ * rB)
    Wfused = reshape(view(ops.bu.data, :, :, 1:maxJ), bk, maxJ * rB)
    Ustack = view(Uall, :, :, 1)
    Tstack = reshape(view(Tbuf, :, :, :, 1:maxJ, 1), noff * rA, maxJ * bn)
    Cblock = view(C, 1:bm, 1:maxJ * bn)
    stage1 = _row_stage1_batches(placement, ops, Ssample, Sfused, Wfused,
                                 ntrip, noff, maxI)
    return merge(stage1, (
        s2s = _batchvec(Ssample, ntrip),
        s2z = _batchvec(view(ops.bv.data, :, :, 1), ntrip),
        s2t = _batchvec(Tsample, ntrip),
        s3u = _batchvec(Ustack, maxI),   # Stage 3 batches over i
        s3t = _batchvec(Tstack, maxI),
        s3c = _batchvec(Cblock, maxI),
    ))
end

@inline function _column_stage1_batches(::KAsSerialLoop{:k}, ops, Vpanel, Wfused,
                                        Sfused, Ssample, nkj, maxK)
    return (
        s1v = _batchvec(Vpanel, nkj),
        s1w = _batchvec(view(ops.bu.data, :, :, 1), nkj),
        s1s = _batchvec(Ssample, nkj),
    )
end

@inline function _column_stage1_batches(::KAsSerialLoop{:j}, ops, Vpanel, Wfused,
                                        Sfused, Ssample, nkj, maxK)
    return (
        s1jv = _batchvec(Vpanel, maxK),
        s1jw = _batchvec(Wfused, maxK),
        s1js = _batchvec(Sfused, maxK),
    )
end

function _column_batches(placement::KAsSerialLoop, ops, C, Sbuf, Tbuf, Vall,
                         Uall, geom, maxK::Int, maxJ::Int)
    bm = geom.bm; bk = geom.bk; bn = geom.bn
    noff = geom.perA_col
    rA = geom.rA
    rB = geom.rB
    nkj = maxK * maxJ                    # Stage 1/2 batch over (k, jpos)
    n3 = noff * maxJ                     # Stage 3 batch over (i, jpos) per k
    Vpanel = view(Vall, :, :, 1)
    Wfused = reshape(view(ops.bu.data, :, :, 1:maxJ), bk, maxJ * rB)
    Sfused = reshape(view(Sbuf, :, :, 1:maxJ, 1), rA * noff, maxJ * rB)
    Ssample = view(Sbuf, :, :, 1, 1)
    stage1 = _column_stage1_batches(placement, ops, Vpanel, Wfused, Sfused,
                                    Ssample, nkj, maxK)
    return merge(stage1, (
        s2s = _batchvec(view(Sbuf, :, :, 1, 1), nkj),
        s2z = _batchvec(view(ops.bv.data, :, :, 1), nkj),
        s2t = _batchvec(reshape(view(Tbuf, :, :, :, 1, 1), rA * noff, bn), nkj),
        s3u = _batchvec(view(Uall, :, :, 1, 1), n3),
        s3t = _batchvec(view(Tbuf, :, 1, :, 1, 1), n3),
        s3c = _batchvec(view(C, 1:bm, 1:bn), n3),
    ))
end

# FoldLeft (row family, FullGrid): T' = U·S is [bm, rB, noff]; Stage 3 stacks B's Z
# (`Zall = reshape(bv.data, bn, rB·noff, qn)` — contiguous per output column iff B is
# TileColMajor, which `choose_fold` guarantees). Stored in the `Ustacked` field, which
# generically holds "the fold's stacked operand".
function _row_batches_left(ops, C, Sbuf, Tbuf, Zall, geom, maxI::Int, maxJ::Int)
    bm = geom.bm
    noff = geom.perA_row
    rA = geom.rA
    rB = geom.rB
    ntrip = noff * maxJ * maxI          # Stage 2 tilewise batch over (i, k, j)
    Usample = view(ops.au.data, :, :, 1)               # U_ik  [bm, rA]
    Ssample = view(Sbuf, :, :, 1, 1, 1)                # S[:,:,p,kk,il]  [rA, rB]
    Tsample = view(Tbuf, :, :, 1, 1, 1)                # T'[:,:,kk,jl,il] [bm, rB]
    Tstack = reshape(view(Tbuf, :, :, :, 1, 1), bm, rB * noff)
    Zstack = view(Zall, :, :, 1)                       # [bn, rB·noff]
    Cblock = view(C, 1:bm, 1:geom.bn)
    return (
        s1v = _batchvec(view(ops.av.data, :, :, 1), ntrip),
        s1w = _batchvec(view(ops.bu.data, :, :, 1), ntrip),
        s1s = _batchvec(Ssample, ntrip),
        s2u = _batchvec(Usample, ntrip),
        s2s = _batchvec(Ssample, ntrip),
        s2t = _batchvec(Tsample, ntrip),
        s3t = _batchvec(Tstack, maxJ * maxI),   # Stage 3 batches over (i, j)
        s3z = _batchvec(Zstack, maxJ * maxI),
        s3c = _batchvec(Cblock, maxJ * maxI),
    )
end

"""
    allocate_workspace(placement, geom, ops, C, budget, fold) -> RowWorkspace | ColumnWorkspace

Promote the contraction's intermediates: allocate S/T scratch and reusable batch buffers
once per term call, sized to the budgeted run width for the given reduction placement and
`fold`.

`geom` fixes the sizes and `ops` supplies storage for reshaped K-stacks and batch views.
This serves interior, panel, and corner factor accessors for both grid kinds.
"""
function allocate_workspace(placement::KAsGemmK, geom, ops, C::AbstractMatrix,
                            budget::Int, ::FoldRight; arena=nothing)
    T = eltype(geom)
    backend = get_backend(ops.av.data)
    bm = geom.bm; bn = geom.bn
    noff = geom.perA_row
    rA = geom.rA
    rB = geom.rB
    maxI, maxJ = _row_block(geom, budget, FoldRight())
    Uall = reshape(ops.au.data, bm, rA * noff, geom.qm)
    # S[:,:,p,kk,il]: p = position within the run's column block, laid out so a fused
    # per-(i,k) Stage-1 GEMM writes a contiguous [rA, len·rB] slice.
    _arena_reset!(arena)
    Sbuf = _workspace_array!(arena, backend, T, rA, rB, maxJ, noff, maxI)
    Tbuf = _workspace_array!(arena, backend, T, rA, noff, bn, maxJ, maxI)
    return RowWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Uall,
                        _row_batches(placement, ops, C, Sbuf, Tbuf, Uall,
                                     geom, maxI, maxJ))
end

function allocate_workspace(placement::KAsGemmK, geom, ops, C::AbstractMatrix,
                            budget::Int, ::FoldLeft; arena=nothing)
    T = eltype(geom)
    backend = get_backend(ops.av.data)
    bm = geom.bm; bn = geom.bn
    noff = geom.perA_row                 # = qk (FullGrid: every contraction tile)
    rA = geom.rA
    rB = geom.rB
    maxI, maxJ = _row_block(geom, budget, FoldLeft())
    # Z-stack: for fixed output column j, all k contiguous ⟺ B TileColMajor.
    Zall = reshape(ops.bv.data, bn, rB * noff, geom.qn)
    _arena_reset!(arena)
    Sbuf = _workspace_array!(arena, backend, T, rA, rB, maxJ, noff, maxI)
    Tbuf = _workspace_array!(arena, backend, T, bm, rB, noff, maxJ, maxI)   # T' = U·S is bm×rB
    return RowWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Zall,
                        _row_batches_left(ops, C, Sbuf, Tbuf, Zall, geom, maxI, maxJ))
end

function allocate_workspace(placement::KAsSerialLoop, geom, ops, C::AbstractMatrix,
                            budget::Int, ::FoldSide; arena=nothing)
    T = eltype(geom)
    backend = get_backend(ops.av.data)
    bm = geom.bm; bk = geom.bk; bn = geom.bn
    noff = geom.perA_col
    rA = geom.rA
    rB = geom.rB
    maxK, maxJ = _column_block(geom, budget)
    Vall = reshape(ops.av.data, bk, rA * noff, geom.qk)
    Uall = reshape(ops.au.data, bm, rA, noff, geom.qk)
    _arena_reset!(arena)
    Sbuf = _workspace_array!(arena, backend, T, rA * noff, rB, maxJ, maxK)
    Tbuf = _workspace_array!(arena, backend, T, rA, noff, bn, maxJ, maxK)
    return ColumnWorkspace(ScratchS(Sbuf), ScratchT(Tbuf), Vall, Uall,
                           _column_batches(placement, ops, C, Sbuf, Tbuf, Vall, Uall,
                                           geom, maxK, maxJ))
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
