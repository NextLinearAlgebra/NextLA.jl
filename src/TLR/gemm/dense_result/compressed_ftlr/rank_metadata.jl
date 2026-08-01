"""Pure per-tile rank facts for one CompressedFTLR operand pair.

No cost or workspace concept lives here — only "what are the ranks and output
tile shapes." `_compressed_ftlr_fold_cost` (fold_cost.jl) turns these facts into
FoldRight/FoldLeft cost estimates; nothing here depends on which fold is used.
"""

@inline _compressed_ftlr_axis_range(A::LogicalTLROperand, tile::Int, axis::Int) =
    _tile_axis_range(A, tile, axis)
@inline function _compressed_ftlr_axis_range(A::CompressedFTLRMatrix, tile::Int, axis::Int)
    width = axis == 1 ? tile_size(A, tile, 1)[1] : tile_size(A, 1, tile)[2]
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    return first:(first + width - 1)
end

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
    @inbounds for k in 1:qk, j in 1:qn
        r = _compressed_ftlr_execution_rank(B, k, j)
        b_row_ranks[k] += r
        b_col_ranks[j] += r
    end
    @inbounds for j in 1:qn, k in 1:qk
        b_col_k_prefix[j, k + 1] = b_col_k_prefix[j, k] + _compressed_ftlr_execution_rank(B, k, j)
    end
    @inbounds for k in 1:qk, j in 1:qn
        b_row_k_prefix[k, j + 1] = b_row_k_prefix[k, j] + _compressed_ftlr_execution_rank(B, k, j)
    end
    pair_ranks = Base.zeros(Int, qm)
    @inbounds for i in 1:qm, k in 1:qk
        r = _compressed_ftlr_execution_rank(A, i, k)
        a_k_prefix[i, k + 1] = a_k_prefix[i, k] + r
        pair_ranks[i] += r * b_row_ranks[k]
    end
    b_total_rank = sum(b_row_ranks)
    b_col_prefix = _compressed_ftlr_prefix(b_col_ranks)
    output_row_heights = [length(_compressed_ftlr_axis_range(A, i, 1)) for i in 1:qm]
    output_col_widths = [length(_compressed_ftlr_axis_range(B, j, 2)) for j in 1:qn]
    output_col_prefix = _compressed_ftlr_prefix(output_col_widths)
    return (; a_k_prefix, b_row_ranks, b_col_ranks, b_col_k_prefix, b_row_k_prefix,
              b_col_prefix, pair_ranks, b_total_rank,
              output_row_heights, output_col_widths, output_col_prefix)
end
