# Operand leaves (ROADMAP milestone 3, step 1).
#
# A **leaf** is a logical view a structured contraction consumes. The whole point of the
# abstraction is one method:
#
#     tilefactor(leaf, i, j) -> zero-copy factor view of logical tile (i, j)
#
# `InteriorOperand` (`operands.jl`) already answers it for the two-index interior grid, with
# `GridKind` absorbing diagonal skipping. What the eight terms additionally need is the
# same interface over the *boundary* storage: panels are indexed by one live coordinate
# and corners by none. Once every leaf answers `tilefactor`, a term reads
# `tilefactor(Aleaf, i, k)` / `tilefactor(Bleaf, k, j)` regardless of which corner of the
# `(i,k,j) ∈ {regular, boundary}³` cube it occupies, and the eight bespoke indexing
# bodies collapse into one.
#
# Leaves reference existing storage. They own no copied factors, transposed tiles,
# identity factors, or workspace — every accessor here is a view.
#
# The interior `ContractOp` consumes these leaves directly. Boundary terms use the same
# leaf vocabulary as their migration to operation-level lowering continues.

# ─── Panel leaf ───────────────────────────────────────────────────────────────

"""
    PanelLeafAxis

Which of a panel's own `(row, col)` tile coordinates is the live one — the coordinate
that selects a storage slot. The other is pinned to the boundary tile.

A right panel holds tiles `X[i, bnd]`, so its live axis is `:row`; a bottom panel holds
`X[bnd, j]`, so its live axis is `:col`. This is a property of the *panel*, not of the
operand's role in the product: A's right panel and B's right panel are both `:row`-live,
even though the contraction reaches them through `(i,k)` and `(k,j)` respectively.
"""
abstract type PanelLeafAxis end
struct PanelRowAxis <: PanelLeafAxis end   # X[i, bnd] — slot from the row coordinate
struct PanelColAxis <: PanelLeafAxis end   # X[bnd, j] — slot from the column coordinate

"""
    PanelOperand{Ax,A3}

Zero-copy view over one boundary panel's flat factor storage `[b, maxrank, ntiles]`,
tagged with the live axis `Ax`. The sibling of `InteriorOperand` for the one-index
regions: the interior needs `(order, qm, qn)` and a `GridKind` to map two coordinates
onto a slot, whereas a panel's live coordinate *is* its slot, so no order or grid
extents are carried.
"""
struct PanelOperand{Ax<:PanelLeafAxis,A3<:AbstractArray}
    data::A3
    axis::Ax
end

@inline panel_operand(axis::PanelLeafAxis, data::AbstractArray) =
    PanelOperand{typeof(axis),typeof(data)}(data, axis)

@inline blockdim(p::PanelOperand) = size(p.data, 1)   # b
@inline rankdim(p::PanelOperand)  = size(p.data, 2)   # maxrank

"""Number of tiles along the panel's live axis."""
@inline panel_tiles(p::PanelOperand) = size(p.data, 3)

@inline tilefactor(p::PanelOperand{PanelRowAxis}, i::Integer, ::Integer) =
    view(p.data, :, :, Int(i))
@inline tilefactor(p::PanelOperand{PanelColAxis}, ::Integer, j::Integer) =
    view(p.data, :, :, Int(j))

# Scheduler addressing for a right panel `X[k,bnd]`. Its logical output-column extent
# is one, while the storage slot is selected by the row/reduction coordinate `k`.
@inline panel_col(::PanelOperand{PanelRowAxis}, ::Integer, pos::Integer) = Int(pos)
@inline col_scratch_pos(::PanelOperand{PanelRowAxis}, j0::Integer, ::Integer, j::Integer) =
    Int(j - j0 + 1)
@inline first_offdiag_col(::PanelOperand{PanelRowAxis}, j0::Integer, ::Integer) = Int(j0)
@inline last_offdiag_col(::PanelOperand{PanelRowAxis}, j1::Integer, ::Integer) = Int(j1)
@inline panel_local(::PanelOperand{PanelRowAxis}, ::Integer, ::Integer) = 1
@inline rowpanel(p::PanelOperand{PanelRowAxis}, k::Integer) =
    view(p.data, :, :, Int(k):Int(k))
@inline stride1_axis_right(::PanelOperand{PanelRowAxis}) = Stride1Axis{:k}()
@inline stride1_axis_left(::PanelOperand{PanelRowAxis}) = Stride1Axis{:i}()

# Mirror interface for a bottom panel used as the left leaf: `X[bnd,k]` has one local
# output row and its storage slots form a complete, contiguous reduction stack.
@inline panel_col(::PanelOperand{PanelColAxis}, ::Integer, pos::Integer) = Int(pos)
@inline stride1_axis_left(::PanelOperand{PanelColAxis}) = Stride1Axis{:k}()
@inline stride1_axis_right(::PanelOperand{PanelColAxis}) = Stride1Axis{:j}()
@inline col_scratch_pos(::PanelOperand{PanelColAxis}, j0::Integer, ::Integer, j::Integer) =
    Int(j - j0 + 1)
@inline first_offdiag_col(::PanelOperand{PanelColAxis}, j0::Integer, ::Integer) = Int(j0)
@inline last_offdiag_col(::PanelOperand{PanelColAxis}, j1::Integer, ::Integer) = Int(j1)
@inline panel_local(::PanelOperand{PanelColAxis}, ::Integer, j::Integer) = Int(j)
@inline rowpanel(p::PanelOperand{PanelColAxis}, ::Integer) = p.data
@inline complete_k_stack(::PanelOperand) = true

# Stacking depths, in the same sense as `InteriorOperand`'s: how many tiles a given row
# (resp. column) of this leaf holds. A panel is one tile wide along its pinned axis and
# `panel_tiles` long along its live one, so the two answers swap with the live axis. These
# feed `geometry`'s `per*` fields, which size the scheduler's scratch slices.
@inline tiles_per_row(p::PanelOperand{PanelRowAxis}) = 1                 # X[i, bnd]: one tile per row
@inline tiles_per_col(p::PanelOperand{PanelRowAxis}) = panel_tiles(p)    # …all of them in one column
@inline tiles_per_row(p::PanelOperand{PanelColAxis}) = panel_tiles(p)    # X[bnd, j]: one row, all tiles
@inline tiles_per_col(p::PanelOperand{PanelColAxis}) = 1

# ─── Corner leaf ──────────────────────────────────────────────────────────────

"""
    CornerOperand{A3}

Zero-copy view over the single low-rank corner tile's factor storage
`[b, maxrank, 1]`. Both coordinates are pinned to the boundary, so the slot is
constant — the degenerate leaf, carried so that corner terms share the leaf interface
rather than special-casing storage access.
"""
struct CornerOperand{A3<:AbstractArray}
    data::A3
end

@inline corner_operand(data::AbstractArray) = CornerOperand{typeof(data)}(data)

@inline blockdim(p::CornerOperand) = size(p.data, 1)
@inline rankdim(p::CornerOperand)  = size(p.data, 2)

@inline tilefactor(p::CornerOperand, ::Integer, ::Integer) = view(p.data, :, :, 1)
@inline panel_col(::CornerOperand, ::Integer, pos::Integer) = Int(pos)
@inline col_scratch_pos(::CornerOperand, j0::Integer, ::Integer, j::Integer) =
    Int(j - j0 + 1)
@inline first_offdiag_col(::CornerOperand, j0::Integer, ::Integer) = Int(j0)
@inline last_offdiag_col(::CornerOperand, j1::Integer, ::Integer) = Int(j1)
@inline panel_local(::CornerOperand, ::Integer, j::Integer) = Int(j)
@inline rowpanel(p::CornerOperand, ::Integer) = p.data
@inline stride1_axis_left(::CornerOperand) = Stride1Axis{:i}()
# Both axes are degenerate. Expose `k` on the right so right-panel × corner selects the
# row family, whose `i` tiling keeps promoted workspace budget-responsive.
@inline stride1_axis_right(::CornerOperand) = Stride1Axis{:k}()
@inline complete_k_stack(::CornerOperand) = true

# One tile in every direction — the degenerate stacking depth.
@inline tiles_per_row(::CornerOperand) = 1
@inline tiles_per_col(::CornerOperand) = 1

# ─── Algebra leaves ───────────────────────────────────────────────────────────

"""
    LowRankLeaf(outer, inner)

A low-rank leaf representing `outer * inner'` over some tile region: `A_ik = U_ik V_ik'`
gives `LowRankLeaf(U, V)`. `outer` and `inner` are factor operands over the same region
and share its addressing, so `factors(leaf, i, j)` returns the `(outer, inner)` view pair
for one logical tile.

Canonicalisation has already happened upstream: `LogicalTLROperand` swaps the two factors
for a transposed operand, so a leaf is `outer * inner'` for every `Op`. Executors never
see a transpose flag on a low-rank leaf.
"""
struct LowRankLeaf{O,I}
    outer::O
    inner::I
end

"""Zero-copy `(outer, inner)` factor views of logical tile `(i, j)`."""
@inline factors(l::LowRankLeaf, i::Integer, j::Integer) =
    (tilefactor(l.outer, i, j), tilefactor(l.inner, i, j))

# Rank is shared by the two factors, so it is a property of the pair.
@inline rankdim(l::LowRankLeaf) = rankdim(l.outer)

# `blockdim` is deliberately *not* defined on a `LowRankLeaf`: for `A_ik = U_ik V_ik'`
# the outer factor's block dim is `b_m` and the inner's is `b_k`, and they differ on
# rectangular tiles. Callers name the one they mean — `blockdim(l.outer)` /
# `blockdim(l.inner)` — exactly as `_interior_geom` already distinguishes
# `blockdim(ops.au)` from `blockdim(ops.av)`. A single accessor here would silently pick
# one of two different sizes.

"""
    DenseLeaf(tile)

A dense algebra leaf: a physical tile view paired with the logical `N/T` operation
needed to read it, wrapping the existing `LogicalDenseTile`.

Dense tiles are a distinct leaf kind, not a low-rank leaf of full rank with an identity
factor. They share the contraction IR but lower through one- or two-stage forms — the
point of keeping them separate is that no identity factor is ever materialised.
"""
struct DenseLeaf{D<:LogicalDenseTile}
    tile::D
end

@inline dense_leaf(tile::LogicalDenseTile) = DenseLeaf{typeof(tile)}(tile)

"""Dense corner leaf of a canonical dense-diagonal operand."""
@inline dense_corner_leaf(A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}) =
    dense_leaf(_diag_tile_ref(A, ndiag_tiles(A)))

"""The physical tile data and the GEMM transpose flag that makes it logical."""
@inline leafdata(l::DenseLeaf) = _dense_data(l.tile)
@inline leafop(l::DenseLeaf) = _dense_op(l.tile)

@inline blockdim(l::DenseLeaf) = size(leafdata(l), 1)

# ─── Region → leaf construction ───────────────────────────────────────────────
#
# The live axis of a panel leaf follows the region, and the factor pair follows the
# canonical `outer * inner'` accessors — both already transpose-aware, so these builders
# need no `Op` branch.

"""Low-rank leaf over a canonical operand's right panel (`X[i, bnd]`)."""
@inline right_panel_leaf(A::LogicalTLROperand) = LowRankLeaf(
    panel_operand(PanelRowAxis(), outer_factors(A, _RIGHT)),
    panel_operand(PanelRowAxis(), inner_factors(A, _RIGHT)),
)

"""Low-rank leaf over a canonical operand's bottom panel (`X[bnd, j]`)."""
@inline bottom_panel_leaf(A::LogicalTLROperand) = LowRankLeaf(
    panel_operand(PanelColAxis(), outer_factors(A, _BOTTOM)),
    panel_operand(PanelColAxis(), inner_factors(A, _BOTTOM)),
)

"""Low-rank leaf over a canonical operand's corner tile."""
@inline corner_leaf(A::LogicalTLROperand) = LowRankLeaf(
    corner_operand(outer_factors(A, _CORNER)),
    corner_operand(inner_factors(A, _CORNER)),
)

"""
Low-rank leaf over a canonical operand's interior grid. `kind` selects the enumeration
policy: `SkipDiag()` for a dense-diagonal container (the diagonal lives elsewhere),
`FullGrid()` for a fully low-rank one.
"""
@inline function interior_leaf(kind::GridKind, A::LogicalTLROperand)
    qm, qn = regular_tilegrid_size(A)
    ord = tile_order(A)
    return LowRankLeaf(
        interior_operand(kind, outer_factors(A, _INTERIOR), ord, qm, qn),
        interior_operand(kind, inner_factors(A, _INTERIOR), ord, qm, qn),
    )
end

"""
    interior_grid_kind(A) -> GridKind

The interior enumeration policy implied by the container: a dense-diagonal operand
stores its diagonal outside the low-rank grid (`SkipDiag`), a fully low-rank one does
not (`FullGrid`). Mirrors the two `logical_operands` methods, so leaf selection and the
existing operand builder cannot drift apart.
"""
@inline interior_grid_kind(::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}) = SkipDiag()
@inline interior_grid_kind(::LogicalTLROperand{<:Any,<:TLRMatrix}) = FullGrid()

"""
    interior_leaves(A, B) -> (Aleaf, Bleaf)

The interior low-rank leaf pair for canonical operands `op(A)` and `op(B)`, each with the
enumeration policy its container implies. Equivalent to `logical_operands(A, B)` regrouped
as `(LowRankLeaf(au, av), LowRankLeaf(bw, bz))`.
"""
@inline interior_leaves(A::LogicalTLROperand, B::LogicalTLROperand) =
    (interior_leaf(interior_grid_kind(A), A), interior_leaf(interior_grid_kind(B), B))
