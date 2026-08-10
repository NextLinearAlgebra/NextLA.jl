# Canonical logical operands and factor-storage views.

"""
    LogicalTLROperand{Op}(parent)

Zero-copy logical view of `op(parent)`, where `Op` is `:N` or `:T`.  The view
canonicalises the complete TLR partition, not only the regular interior: a
transpose swaps matrix geometry, tile order, right/bottom panels, and the two
low-rank factors.  Consequently every exposed low-rank tile is still written
as `outer * inner'`, independent of the requested operation.
"""
struct LogicalTLROperand{Op,A<:AbstractTLRMatrix}
    parent::A
end

@inline function _normalize_tlr_op(op::Char)
    c = uppercase(op)
    c in ('N', 'T') || throw(ArgumentError("TLR GEMM operation must be 'N' or 'T', got '$op'"))
    return c
end

@inline logical_operand(A::AbstractTLRMatrix, op::Char='N') =
    _logical_operand(A, Val(_normalize_tlr_op(op)))
@inline _logical_operand(A, ::Val{'N'}) = LogicalTLROperand{:N,typeof(A)}(A)
@inline _logical_operand(A, ::Val{'T'}) = LogicalTLROperand{:T,typeof(A)}(A)

@inline physical(A::LogicalTLROperand) = getfield(A, :parent)
@inline _orient_axes(::LogicalTLROperand{:N}, axes) = axes
@inline _orient_axes(::LogicalTLROperand{:T}, axes) = reverse(axes)

@inline outer_factors(A::LogicalTLROperand{:N}, region::TLRRegion) =
    outer_factors(physical(A), region)
@inline inner_factors(A::LogicalTLROperand{:N}, region::TLRRegion) =
    inner_factors(physical(A), region)
@inline outer_factors(A::LogicalTLROperand{:T}, region::TLRRegion) =
    inner_factors(physical(A), transpose_region(region))
@inline inner_factors(A::LogicalTLROperand{:T}, region::TLRRegion) =
    outer_factors(physical(A), transpose_region(region))

Base.eltype(::Type{<:LogicalTLROperand{<:Any,A}}) where {A} = eltype(A)
Base.eltype(A::LogicalTLROperand) = eltype(physical(A))
Base.size(A::LogicalTLROperand) = _orient_axes(A, size(physical(A)))
Base.size(A::LogicalTLROperand, d::Int) = size(A)[d]

@inline nominal_tile_size(A::LogicalTLROperand) =
    _orient_axes(A, nominal_tile_size(physical(A)))
@inline nominal_tile_size(A::LogicalTLROperand, d::Integer) = nominal_tile_size(A)[Int(d)]
@inline tail_tile_size(A::LogicalTLROperand) =
    _orient_axes(A, tail_tile_size(physical(A)))
@inline tail_tile_size(A::LogicalTLROperand, d::Integer) = tail_tile_size(A)[Int(d)]
@inline grid_size(A::LogicalTLROperand) =
    _orient_axes(A, grid_size(physical(A)))
@inline regular_grid_size(A::LogicalTLROperand) =
    _orient_axes(A, regular_grid_size(physical(A)))
@inline tile_order(A::LogicalTLROperand{:N}) = tile_order(physical(A))
@inline tile_order(A::LogicalTLROperand{:T}) = _transpose_order(tile_order(physical(A)))
@inline maxrank(A::LogicalTLROperand) = maxrank(physical(A))
@inline KernelAbstractions.get_backend(A::LogicalTLROperand) = get_backend(physical(A))

"""Zero-copy logical `N/T` view of a standalone dense GEMM operand."""
struct LogicalDenseOperand{Op,A<:AbstractMatrix}
    data::A
end

@inline logical_dense_operand(A::AbstractMatrix, op::Char='N') =
    _logical_dense_operand(A, Val(_normalize_tlr_op(op)))
@inline _logical_dense_operand(A, ::Val{'N'}) = LogicalDenseOperand{:N,typeof(A)}(A)
@inline _logical_dense_operand(A, ::Val{'T'}) = LogicalDenseOperand{:T,typeof(A)}(A)
Base.eltype(A::LogicalDenseOperand) = eltype(A.data)
Base.size(A::LogicalDenseOperand{:N}) = size(A.data)
Base.size(A::LogicalDenseOperand{:T}) = reverse(size(A.data))
Base.size(A::LogicalDenseOperand, d::Int) = size(A)[d]
@inline KernelAbstractions.get_backend(A::LogicalDenseOperand) = get_backend(A.data)

"""Physical dense block and BLAS operation representing logical `rows × cols`."""
@inline _dense_block(A::LogicalDenseOperand{:N}, rows, cols) = (view(A.data, rows, cols), 'N')
@inline _dense_block(A::LogicalDenseOperand{:T}, rows, cols) = (view(A.data, cols, rows), 'T')

"""A physical dense tile together with the operation needed to make it logical."""
struct LogicalDenseTile{Op,A<:AbstractMatrix}
    data::A
end
@inline _dense_data(A::LogicalDenseTile) = A.data
@inline _dense_op(::LogicalDenseTile{:N}) = 'N'
@inline _dense_op(::LogicalDenseTile{:T}) = 'T'

@inline function _diag_tile_ref(A::LogicalTLROperand{Op,<:TLRMatrix}, k::Int) where {Op}
    LogicalDenseTile{Op,typeof(_diag_tile_view(physical(A), k))}(_diag_tile_view(physical(A), k))
end
@inline ndiag_tiles(A::LogicalTLROperand{<:Any,<:TLRMatrix}) = ndiag_tiles(physical(A))
@inline _nfull_diag_tiles(A::LogicalTLROperand{<:Any,<:TLRMatrix}) = _nfull_diag_tiles(physical(A))

"""Logical row/column range of one TLR tile along `axis`."""
@inline function _tile_axis_range(A::LogicalTLROperand, tile::Int, axis::Int)
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    extent = tile_size(A, axis == 1 ? tile : 1, axis == 2 ? tile : 1)[axis]
    return first:(first + extent - 1)
end

"""Canonical full-rank-column factor views of logical full-LR tile `(i,j)`."""
@inline function logical_tile_factors(A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix}, i::Int, j::Int)
    qm, qn = regular_grid_size(A)
    region, slot = if i <= qm && j <= qn
        (_INTERIOR, tile_linear_index(tile_order(A), qm, qn, i, j))
    elseif i <= qm
        (_RIGHT, i)
    elseif j <= qn
        (_BOTTOM, j)
    else
        (_CORNER, 1)
    end
    return (view(outer_factors(A, region), :, :, slot),
            view(inner_factors(A, region), :, :, slot))
end

# Exact-rank CompressedFTLR tile access through the same logical N/T view.  A transpose
# swaps factor roles and tile coordinates; the physical column packing of V/Z
# thereby becomes logical row packing, and conversely for U/W.
@inline _compressed_ftlr_logical_coords(::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) = (i, j)
@inline _compressed_ftlr_logical_coords(::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) = (j, i)

@inline _compressed_ftlr_rank(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_rank(physical(A), i, j)
@inline _compressed_ftlr_rank(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_rank(physical(A), j, i)
@inline _compressed_ftlr_execution_rank(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_execution_rank(physical(A), i, j)
@inline _compressed_ftlr_execution_rank(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    _compressed_ftlr_execution_rank(physical(A), j, i)

@inline compressed_ftlr_outer(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_outer(physical(A), i, j)
@inline compressed_ftlr_outer(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_inner(physical(A), j, i)
@inline compressed_ftlr_inner(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_inner(physical(A), i, j)
@inline compressed_ftlr_inner(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_outer(physical(A), j, i)
@inline compressed_ftlr_execution_outer(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_outer(physical(A), i, j)
@inline compressed_ftlr_execution_outer(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_inner(physical(A), j, i)
@inline compressed_ftlr_execution_inner(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_inner(physical(A), i, j)
@inline compressed_ftlr_execution_inner(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    compressed_ftlr_execution_outer(physical(A), j, i)

@inline logical_tile_factors(
    A::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    (compressed_ftlr_outer(A, i, j), compressed_ftlr_inner(A, i, j))

@inline compressed_ftlr_outer_order(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) =
    compressed_ftlr_outer_order(physical(A))
@inline compressed_ftlr_outer_order(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) =
    _transpose_order(compressed_ftlr_inner_order(physical(A)))
@inline compressed_ftlr_inner_order(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) =
    compressed_ftlr_inner_order(physical(A))
@inline compressed_ftlr_inner_order(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) =
    _transpose_order(compressed_ftlr_outer_order(physical(A)))

@inline _compressed_ftlr_outer_storage(A::CompressedFTLRMatrix) = A.outer
@inline _compressed_ftlr_inner_storage(A::CompressedFTLRMatrix) = A.inner
@inline _compressed_ftlr_parent(A::CompressedFTLRMatrix) = A
@inline _compressed_ftlr_outer_storage(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) = physical(A).outer
@inline _compressed_ftlr_outer_storage(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) = physical(A).inner
@inline _compressed_ftlr_inner_storage(A::LogicalTLROperand{:N,<:CompressedFTLRMatrix}) = physical(A).inner
@inline _compressed_ftlr_inner_storage(A::LogicalTLROperand{:T,<:CompressedFTLRMatrix}) = physical(A).outer
@inline _compressed_ftlr_parent(A::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix}) = physical(A)

# ─── Interior operand ─────────────────────────────────────────────────────────

"""
    InteriorOperand{OrderT,A3}

Zero-copy view over one operand's flat interior factor storage `[b, maxrank, ntiles]`,
tagged with its tile-grid extents `(qm, qn)` and traversal `order`.
"""
struct InteriorOperand{OrderT<:TileOrderStyle,A3<:AbstractArray}
    data::A3
    order::OrderT
    qm::Int
    qn::Int
end

@inline rankdim(p::InteriorOperand)  = size(p.data, 2)   # maxrank

@inline tiles_per_row(p::InteriorOperand) = p.qn
@inline tiles_per_col(p::InteriorOperand) = p.qm

"""Contiguous `[b, maxrank, tiles_per_row]` view of tile-row `r`'s panel."""
@inline function rowpanel(p::InteriorOperand, r::Integer)
    npr = tiles_per_row(p)
    return view(p.data, :, :, (Int(r) - 1) * npr + 1 : Int(r) * npr)
end

"""Contiguous `[b, maxrank, tiles_per_col]` view of tile-column `c`'s panel.

The caller must select this accessor only when the logical operand is
tile-column-major.  Keeping that decision at the scheduler makes the returned
view zero-copy and avoids a gather for fixed-row factor stacks.
"""
@inline function colpanel(p::InteriorOperand, c::Integer)
    npc = tiles_per_col(p)
    return view(p.data, :, :, (Int(c) - 1) * npc + 1 : Int(c) * npc)
end

"""Zero-copy factor view for logical interior tile `(i,j)`."""
@inline tilefactor(p::InteriorOperand, i::Integer, j::Integer) =
    view(p.data, :, :, tile_linear_index(p.order, p.qm, p.qn, Int(i), Int(j)))

"""Wrap one interior factor array with its grid geometry and tile order."""
@inline function interior_operand(data::AbstractArray, order::TileOrderStyle,
                                  qm::Int, qn::Int)
    return InteriorOperand{typeof(order),typeof(data)}(data, order, qm, qn)
end

"""
    LogicalTLROperands(av, bu, bv, au)

The interior factor panels used by the staged off-diagonal product, named for the
formulas `A_ik = U_ik V_ik'` and `B_kj = W_kj Z_kj'`: `av = V`, `au = U` (from `A`);
`bu = W`, `bv = Z` (from `B`). Stage 3 of the row family stacks `U` in workspace; the
column family enumerates it tilewise, hence `au` is carried here.
"""
struct LogicalTLROperands{AV,BU,BV,AU}
    av::AV
    bu::BU
    bv::BV
    au::AU
end

@inline _transpose_order(::TileColMajor) = TileRowMajor()
@inline _transpose_order(::TileRowMajor) = TileColMajor()

"""
    logical_operands(A, B) -> LogicalTLROperands

Wrap the canonical interior factor arrays of logical `op(A)` and `op(B)` as zero-copy
`InteriorOperand`s using a full grid (rectangular grids allowed).

The whole-matrix `LogicalTLROperand` has already swapped factors, grid extents, and
tile order for `T`. Because effective order drives placement, Stage-3 K-stacks and
fused Stage-1 reshapes still group physically contiguous panels; executors need no
transpose awareness.
"""
logical_operands(A::AbstractTLRMatrix, B::AbstractTLRMatrix) =
    logical_operands(logical_operand(A), logical_operand(B))

function logical_operands(A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix},
                          B::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix})
    qmA, qnA = regular_grid_size(A)
    qmB, qnB = regular_grid_size(B)
    ordA = tile_order(A)
    ordB = tile_order(B)
    return LogicalTLROperands(
        interior_operand(inner_factors(A, _INTERIOR), ordA, qmA, qnA), # av
        interior_operand(outer_factors(B, _INTERIOR), ordB, qmB, qnB), # bu
        interior_operand(inner_factors(B, _INTERIOR), ordB, qmB, qnB), # bv
        interior_operand(outer_factors(A, _INTERIOR), ordA, qmA, qnA), # au
    )
end
