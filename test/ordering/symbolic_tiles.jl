using SparseArrays

struct SymbolicLimit <: Exception
    limit::Int
end
Base.showerror(io::IO, e::SymbolicLimit) = print(io, "symbolic factor exceeds $(e.limit) entries")

# Input is the upper triangle in the desired elimination order. An elimination
# tree reach for row j gives exactly the columns i < j of structural L[j, i].
function symbolic_lower!(visit, upper::SparseMatrixCSC; limit=typemax(Int))
    n = size(upper, 1)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    rows = rowvals(upper)
    for j in 1:n
        for k in nzrange(upper, j)
            i = rows[k]
            i >= j && break
            while i != 0 && i < j
                next = ancestor[i]
                ancestor[i] = j
                next == 0 && (parent[i] = j)
                i = next
            end
        end
    end
    seen = zeros(Int, n)
    count = n
    count <= limit || throw(SymbolicLimit(limit))
    for j in 1:n
        for k in nzrange(upper, j)
            i = rows[k]
            i >= j && break
            while i != 0 && i < j && seen[i] != j
                seen[i] = j
                count += 1
                count <= limit || throw(SymbolicLimit(limit))
                visit(j, i)
                i = parent[i]
            end
        end
    end
    count
end

mutable struct Support
    rows::BitSet
    cols::BitSet
    entries::Int
end
Support() = Support(BitSet(), BitSet(), 0)

function record!(support, i, j)
    push!(support.rows, i)
    push!(support.cols, j)
    support.entries += 1
end

struct PanelPattern
    region::NamedTuple{(:lo, :cut, :hi, :level),NTuple{4,Int}}
    full::Support
    factor_tiles::Vector{Dict{Tuple{Int,Int},Support}}
    input_tiles::Vector{Set{Tuple{Int,Int}}}
    input_entries::Base.RefValue{Int}
end

function observe_panels(upper, tree_regions, widths; limit)
    panels = Dict(r.lo => Dict{Int,PanelPattern}() for r in tree_regions)
    for r in tree_regions
        panels[r.lo][r.hi] = PanelPattern(r, Support(),
            [Dict{Tuple{Int,Int},Support}() for _ in widths],
            [Set{Tuple{Int,Int}}() for _ in widths], Ref(0))
    end
    function locate(i, j)
        lo, hi = 1, size(upper, 1)
        while haskey(panels, lo) && haskey(panels[lo], hi)
            panel = panels[lo][hi]
            cut = panel.region.cut
            i <= cut ? (hi = cut) : j > cut ? (lo = cut + 1) : (return panel)
        end
        nothing
    end
    rows = rowvals(upper)
    for i in axes(upper, 2)
        for k in nzrange(upper, i)
            j = rows[k]
            j >= i && break
            panel = locate(i, j)
            panel === nothing && continue
            r = panel.region
            panel.input_entries[] += 1
            for (w, tiles) in zip(widths, panel.input_tiles)
                push!(tiles, (cld(i-r.cut, w), cld(j-r.lo+1, w)))
            end
        end
    end
    nnz_L = symbolic_lower!(upper; limit) do i, j
        panel = locate(i, j)
        panel === nothing && return
        r = panel.region
        i, j = i-r.cut, j-r.lo+1
        record!(panel.full, i, j)
        for (w, tiles) in zip(widths, panel.factor_tiles)
            support = get!(Support, tiles, (cld(i, w), cld(j, w)))
            record!(support, mod1(i, w), mod1(j, w))
        end
    end
    nnz_L, [panels[r.lo][r.hi] for r in tree_regions]
end

contiguous(indices) = isempty(indices) || last(indices) - first(indices) + 1 == length(indices)

function panel_metrics(panel, width, index)
    (; lo, cut, hi, level) = panel.region
    m, n = hi-cut, cut-lo+1
    tiles = panel.factor_tiles[index]
    occupied = length(tiles)
    row_tiles, col_tiles = cld(m,width), cld(n,width)
    slots = row_tiles * col_tiles
    col_supports = [BitSet[] for _ in 1:col_tiles]
    kept = row_packed = col_packed = both_packed = 0
    partial_rows = partial_cols = contiguous_rows = contiguous_cols = 0
    for ((i,j), support) in tiles
        h, w = min(width, m-(i-1)*width), min(width, n-(j-1)*width)
        nr, nc = length(support.rows), length(support.cols)
        kept += h*w
        row_packed += nr*w
        col_packed += h*nc
        both_packed += nr*nc
        partial_rows += nr < h
        partial_cols += nc < w
        contiguous_rows += nr < h && contiguous(support.rows)
        contiguous_cols += nc < w && contiguous(support.cols)
        push!(col_supports[j], support.cols)
    end
    # X*X' has one product per unordered row-tile pair at each column tile.
    triangular(k) = k*(k+1) ÷ 2
    source_products = sum(c -> triangular(length(c)), col_supports)
    shared_products = occupied
    for masks in col_supports, i in eachindex(masks), j in 1:i-1
        shared_products += !isdisjoint(masks[i], masks[j])
    end
    (; level, lo, cut, hi, panel_rows=m, panel_cols=n,
       input_entries=panel.input_entries[], factor_entries=panel.full.entries,
       active_panel_rows=length(panel.full.rows), active_panel_cols=length(panel.full.cols),
       panel_rows_contiguous=contiguous(panel.full.rows), panel_cols_contiguous=contiguous(panel.full.cols),
       tile_width=width, tiles=slots, empty_tiles=slots-occupied,
       input_tiles=length(panel.input_tiles[index]),
       fill_only_tiles=count(key -> key ∉ panel.input_tiles[index], keys(tiles)),
       partial_row_tiles=partial_rows, partial_col_tiles=partial_cols,
       contiguous_partial_row_tiles=contiguous_rows, contiguous_partial_col_tiles=contiguous_cols,
       dense_cells=m*n, kept_cells=kept, row_packed_cells=row_packed,
       col_packed_cells=col_packed, both_packed_cells=both_packed,
       dense_schur_products=col_tiles*triangular(row_tiles), source_schur_products=source_products,
       shared_col_schur_products=shared_products)
end
