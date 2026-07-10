"""Tile-column of `SkipDiag` local panel position `pos` within row `r` (skips `r`)."""
@inline local_to_col(r::Integer, pos::Integer) = pos < r ? pos : pos + 1
# `SkipDiag` scratch column of output column `j` within block `[j0:j1]` for contraction
# `k`: the diagonal column `j=k` is removed, so columns past `k` shift down by one.
@inline _offdiag_pos(j0::Int, k::Int, j::Int) = (j - j0 + 1) - ((j0 <= k) & (k < j) ? 1 : 0)
# `SkipDiag` panel-local index (in `rowpanel(k)`) of absolute column `j ≠ k`.
@inline _panel_local(k::Int, j::Int) = j < k ? j : j - 1

@inline _tiles_per_row(::SkipDiag, qn::Int) = qn - 1
@inline _tiles_per_row(::FullGrid, qn::Int) = qn
@inline _panel_col(::SkipDiag, r::Int, pos::Int) = local_to_col(r, pos)
@inline _panel_col(::FullGrid, ::Int, pos::Int) = pos
# scratch column, or 0 to signal "skip this (diagonal) column".
@inline _col_scratch_pos(::SkipDiag, j0::Int, k::Int, j::Int) = j == k ? 0 : _offdiag_pos(j0, k, j)
@inline _col_scratch_pos(::FullGrid, j0::Int, ::Int, j::Int) = j - j0 + 1
@inline _first_offdiag_col(::SkipDiag, j0::Int, k::Int) = j0 == k ? j0 + 1 : j0
@inline _first_offdiag_col(::FullGrid, j0::Int, ::Int) = j0
@inline _last_offdiag_col(::SkipDiag, j1::Int, k::Int) = j1 == k ? j1 - 1 : j1
@inline _last_offdiag_col(::FullGrid, j1::Int, ::Int) = j1
@inline _panel_local_pos(::SkipDiag, k::Int, j::Int) = _panel_local(k, j)
@inline _panel_local_pos(::FullGrid, ::Int, j::Int) = j
@inline _tile_slot(::SkipDiag, order, qm::Int, qn::Int, i::Int, j::Int) = _offdiag_index(order, qm, qn, i, j)
@inline _tile_slot(::FullGrid, order, qm::Int, qn::Int, i::Int, j::Int) = tile_linear_index(order, qm, qn, i, j)

# ─── Interior operand ─────────────────────────────────────────────────────────

"""
    InteriorOperand{Kind,OrderT,A3}

Zero-copy view over one operand's flat interior factor storage `[b, maxrank, ntiles]`,
tagged with its tile-grid extents `(qm, qn)`, traversal `order`, and enumeration
`Kind` (`SkipDiag` / `FullGrid`). All tile addressing goes through the `Kind` policy,
so the staged core is agnostic to whether the diagonal is stored separately.
"""
struct InteriorOperand{Kind<:GridKind,OrderT<:TileOrderStyle,A3<:AbstractArray}
    data::A3
    order::OrderT
    qm::Int
    qn::Int
end

@inline _kind(::InteriorOperand{Kind}) where {Kind} = Kind()

@inline blockdim(p::InteriorOperand) = size(p.data, 1)   # b
@inline rankdim(p::InteriorOperand)  = size(p.data, 2)   # maxrank

"""Tiles per row of this operand's grid (excludes the diagonal for `SkipDiag`)."""
@inline tiles_per_row(p::InteriorOperand) = _tiles_per_row(_kind(p), p.qn)
"""Tiles per column of this operand's grid (excludes the diagonal for `SkipDiag`)."""
@inline tiles_per_col(p::InteriorOperand) = _tiles_per_row(_kind(p), p.qm)
"""Actual tile-column of local panel position `pos` within row `r`."""
@inline panel_col(p::InteriorOperand, r::Integer, pos::Integer) = _panel_col(_kind(p), Int(r), Int(pos))
"""Scratch column of output column `j` in block `[j0:_]` for contraction `k`, or 0 to skip it."""
@inline col_scratch_pos(p::InteriorOperand, j0::Integer, k::Integer, j::Integer) = _col_scratch_pos(_kind(p), Int(j0), Int(k), Int(j))
"""First / last included output column of block `[j0:j1]` for contraction `k`."""
@inline first_offdiag_col(p::InteriorOperand, j0::Integer, k::Integer) = _first_offdiag_col(_kind(p), Int(j0), Int(k))
@inline last_offdiag_col(p::InteriorOperand, j1::Integer, k::Integer) = _last_offdiag_col(_kind(p), Int(j1), Int(k))
"""Panel-local index (in `rowpanel(k)`) of absolute column `j`."""
@inline panel_local(p::InteriorOperand, k::Integer, j::Integer) = _panel_local_pos(_kind(p), Int(k), Int(j))

"""Contiguous `[b, maxrank, tiles_per_row]` view of tile-row `r`'s panel."""
@inline function rowpanel(p::InteriorOperand, r::Integer)
    npr = tiles_per_row(p)
    return view(p.data, :, :, (Int(r) - 1) * npr + 1 : Int(r) * npr)
end

"""Zero-copy factor view for logical interior tile `(i,j)`."""
@inline tilefactor(p::InteriorOperand, i::Integer, j::Integer) =
    view(p.data, :, :, _tile_slot(_kind(p), p.order, p.qm, p.qn, Int(i), Int(j)))

"""Wrap one interior factor array with its grid geometry and enumeration policy."""
@inline function interior_operand(kind::GridKind, data::AbstractArray, order::TileOrderStyle, qm::Int, qn::Int)
    return InteriorOperand{typeof(kind),typeof(order),typeof(data)}(data, order, qm, qn)
end

# ─── Logical operands ───────────────────────────────────────────────────────────
#
# Thin typed wrappers so stage builders can name operands ("use V' · W") instead
# of spelling raw `view(A.int_V, …)` / `view(B.int_U, …)`.

"""
    LogicalTLROperands(av, bw, bz, au)

The interior factor panels used by the staged off-diagonal product, named for the
formulas `A_ik = U_ik V_ik'` and `B_kj = W_kj Z_kj'`: `av = V`, `au = U` (from `A`);
`bw = W`, `bz = Z` (from `B`). Stage 3 of the row family stacks `U` in workspace; the
column family enumerates it tilewise, hence `au` is carried here.
"""
struct LogicalTLROperands{AV,BW,BZ,AU}
    av::AV
    bw::BW
    bz::BZ
    au::AU
end

"""Stage-1 scratch buffer holding `S_ikj = V_ik' W_kj`."""
struct ScratchS{A}
    data::A
end

"""Stage-2 scratch buffer holding `T_ikj = S_ikj Z_kj'`."""
struct ScratchT{A}
    data::A
end

"""
    logical_operands(A, B) -> LogicalTLROperands

Wrap the interior factor arrays of `A` and `B` as zero-copy `InteriorOperand`s.
Dense-diagonal operands use the `SkipDiag` enumeration policy.
"""
function logical_operands(A::TLRDenseDiagMatrix, B::TLRDenseDiagMatrix)
    qAm, qAn = _full_regular_grid(A)
    qBm, qBn = _full_regular_grid(B)
    ordA = tile_order(A)
    ordB = tile_order(B)
    return LogicalTLROperands(
        interior_operand(SkipDiag(), A.int_V, ordA, qAm, qAn),   # av = V (A)
        interior_operand(SkipDiag(), B.int_U, ordB, qBm, qBn),   # bw = W (B)
        interior_operand(SkipDiag(), B.int_V, ordB, qBm, qBn),   # bz = Z (B)
        interior_operand(SkipDiag(), A.int_U, ordA, qAm, qAn),   # au = U (A)
    )
end

"""
    logical_operands(A::TLRMatrix, B::TLRMatrix) -> LogicalTLROperands

Fully low-rank operands: every interior tile is present, so the `FullGrid`
enumeration policy is used (no diagonal skip). Grid extents may be rectangular.
"""
function logical_operands(A::TLRMatrix, B::TLRMatrix)
    qAm, qAn = _full_regular_grid(A)
    qBm, qBn = _full_regular_grid(B)
    ordA = tile_order(A)
    ordB = tile_order(B)
    return LogicalTLROperands(
        interior_operand(FullGrid(), A.int_V, ordA, qAm, qAn),   # av = V (A)
        interior_operand(FullGrid(), B.int_U, ordB, qBm, qBn),   # bw = W (B)
        interior_operand(FullGrid(), B.int_V, ordB, qBm, qBn),   # bz = Z (B)
        interior_operand(FullGrid(), A.int_U, ordA, qAm, qAn),   # au = U (A)
    )
end

"""Zero-copy `b×b` view of dense output tile `(i, j)`."""
@inline dense_tile(C::AbstractMatrix, b::Int, i::Int, j::Int) =
    view(C, (i - 1) * b + 1:i * b, (j - 1) * b + 1:j * b)

"""Zero-copy view of dense output row-block `i`, columns `j0:j1`."""
@inline dense_rowblock(C::AbstractMatrix, b::Int, i::Int, j0::Int, j1::Int) =
    view(C, (i - 1) * b + 1:i * b, (j0 - 1) * b + 1:j1 * b)
