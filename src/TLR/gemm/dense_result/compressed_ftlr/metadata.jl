"""Pure per-tile rank facts for one CompressedFTLR operand pair.

No cost or workspace concept lives here — only "what are the ranks and output
tile shapes." `_compressed_ftlr_fold_cost` (`costs.jl`) turns these facts into
FoldRight/FoldLeft cost estimates; nothing here depends on which fold is used.
"""

@inline _compressed_ftlr_axis_range(A::AbstractTLRMatrix, tile::Int, axis::Int) =
    _tile_axis_range(A, tile, axis)

@inline _compressed_ftlr_right_valid(A, B) =
    compressed_ftlr_outer_order(A) isa TileRowMajor && compressed_ftlr_outer_order(B) isa TileRowMajor
@inline _compressed_ftlr_left_valid(A, B) =
    compressed_ftlr_outer_order(B) isa TileRowMajor && compressed_ftlr_inner_order(B) isa TileColMajor

@inline function _compressed_ftlr_prefix(values::Vector{Int})
    prefix = Base.zeros(Int, length(values) + 1)
    @inbounds for i in eachindex(values)
        prefix[i + 1] = prefix[i] + values[i]
    end
    return prefix
end

@inline _compressed_ftlr_range_total(prefix::Vector{Int}, rows::UnitRange{Int}) =
    prefix[last(rows) + 1] - prefix[first(rows)]

# --------------------------------------------------------- column-range views
# Work is scheduled over output tile rows `irange` × tile columns `jrange`.
# Passing the full `1:qn` reproduces whole-width behaviour exactly; a narrower
# `jrange` shrinks both arenas, which is what lets the workspace budget be met
# at a granularity finer than one output row. Every quantity that aggregates
# over `j` must therefore be queried on the range rather than read whole, and
# every offset into a `j`-indexed arena rebased to the range start.
#
# These are deliberately untyped in their first argument: they read only field
# names shared by the rank-metadata NamedTuple and `CompressedFTLRRankPlan`, so
# both `costs.jl` (which has the former) and `three_stage.jl` (the latter) use
# the same accessors.

"""`σ_k` restricted to `jrange`: `Σ_{j ∈ jrange} rB_kj`."""
@inline _compressed_ftlr_row_rank(meta, k::Int, jrange) =
    meta.b_row_k_prefix[k, last(jrange) + 1] - meta.b_row_k_prefix[k, first(jrange)]

"""Column offset of tile `j` inside a `jrange`-local S block."""
@inline _compressed_ftlr_row_rank_offset(meta, k::Int, j::Int, jrange) =
    meta.b_row_k_prefix[k, j] - meta.b_row_k_prefix[k, first(jrange)]

"""Total output width spanned by `jrange`, in matrix columns."""
@inline _compressed_ftlr_width(meta, jrange) =
    meta.output_col_prefix[last(jrange) + 1] - meta.output_col_prefix[first(jrange)]

"""Column offset of tile `j` inside a `jrange`-local dense block."""
@inline _compressed_ftlr_width_offset(meta, j::Int, jrange) =
    meta.output_col_prefix[j] - meta.output_col_prefix[first(jrange)]

"""`Γ` restricted to `jrange`: `Σ_{j ∈ jrange} γ_j`."""
@inline _compressed_ftlr_total_rank(meta, jrange) =
    meta.b_col_prefix[last(jrange) + 1] - meta.b_col_prefix[first(jrange)]

"""Matrix column span of `jrange` in the dense output."""
@inline function _compressed_ftlr_output_cols(B, jrange)
    lo = first(_compressed_ftlr_axis_range(B, first(jrange), 2))
    hi = last(_compressed_ftlr_axis_range(B, last(jrange), 2))
    return lo:hi
end

"""Does B have a rank-zero tile at contraction row `k` within `jrange`?"""
@inline function _compressed_ftlr_row_has_zero(meta, k::Int, jrange)
    nonzero = meta.b_row_nonzero_prefix[k, last(jrange) + 1] -
              meta.b_row_nonzero_prefix[k, first(jrange)]
    return nonzero < length(jrange)
end

"""
    _compressed_ftlr_rank_metadata(A, B)

Host-only rank reductions and O(1) range-query tables: `a_k_prefix[i,·]` is A's
running rank prefix over `k`; `b_row_ranks[k] = σ_k = Σ_j rB_kj`; `b_col_ranks[j]
= γ_j = Σ_k rB_kj`; `pair_ranks[i] = Σ_k rA_ik·σ_k` is the number of S-arena
elements row `i` contributes (fold-independent, since Stage 1 is shared).
`output_row_heights`/`output_col_widths` are the physical tile extents (`m_i`,
`n_j` in the paper's notation), independent of rank.
"""
function _compressed_ftlr_rank_metadata(A, B)
    qm, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    a_k_prefix = Base.zeros(Int, qm, qk + 1)
    b_row_ranks = Base.zeros(Int, qk)
    b_col_ranks = Base.zeros(Int, qn)
    b_col_k_prefix = Base.zeros(Int, qn, qk + 1)
    b_row_k_prefix = Base.zeros(Int, qk, qn + 1)
    # Count of NONZERO B tiles per contraction row, as a prefix over j. Stage 2
    # skips `(k,j)` GEMMs where `rB_kj == 0`, leaving FoldRight's arena with
    # reserved-but-unwritten space, so `three_stage.jl` must know whether such a
    # hole exists. A prefix (rather than a plain flag) keeps that query O(1)
    # for an arbitrary column range, which the column-block scheduler needs.
    b_row_nonzero_prefix = Base.zeros(Int, qk, qn + 1)
    @inbounds for k in 1:qk, j in 1:qn
        r = _compressed_ftlr_storage_rank(B, k, j)
        b_row_ranks[k] += r
        b_col_ranks[j] += r
        b_row_nonzero_prefix[k, j + 1] = b_row_nonzero_prefix[k, j] + (r != 0)
    end
    @inbounds for j in 1:qn, k in 1:qk
        b_col_k_prefix[j, k + 1] = b_col_k_prefix[j, k] + _compressed_ftlr_storage_rank(B, k, j)
    end
    @inbounds for k in 1:qk, j in 1:qn
        b_row_k_prefix[k, j + 1] = b_row_k_prefix[k, j] + _compressed_ftlr_storage_rank(B, k, j)
    end
    pair_ranks = Base.zeros(Int, qm)
    @inbounds for i in 1:qm, k in 1:qk
        r = _compressed_ftlr_storage_rank(A, i, k)
        a_k_prefix[i, k + 1] = a_k_prefix[i, k] + r
        pair_ranks[i] += r * b_row_ranks[k]
    end
    b_total_rank = sum(b_row_ranks)
    b_col_prefix = _compressed_ftlr_prefix(b_col_ranks)
    output_row_heights = [length(_compressed_ftlr_axis_range(A, i, 1)) for i in 1:qm]
    output_col_widths = [length(_compressed_ftlr_axis_range(B, j, 2)) for j in 1:qn]
    output_col_prefix = _compressed_ftlr_prefix(output_col_widths)
    return (; a_k_prefix, b_row_ranks, b_col_ranks, b_col_k_prefix, b_row_k_prefix,
              b_col_prefix, b_row_nonzero_prefix, pair_ranks, b_total_rank,
              output_row_heights, output_col_widths, output_col_prefix)
end
