# Result-independent logical view of a TLR operand.

"""
    LogicalTLROperand{Op}(parent)

Zero-copy logical view of `op(parent)`, where `Op` is `:N` or `:T`. A transpose
swaps matrix geometry, tile order, and low-rank factor roles; result-specific
operand adapters are defined inside `dense_result/` and `padded_result/`.
"""
struct LogicalTLROperand{Op,A<:AbstractTLRMatrix}
    parent::A
end

@inline function _normalize_tlr_op(op::Char)
    c = uppercase(op)
    c in ('N', 'T') || throw(ArgumentError(
        "TLR GEMM operation must be 'N' or 'T', got '$op'"))
    return c
end

@inline logical_operand(A::AbstractTLRMatrix, op::Char='N') =
    _logical_operand(A, Val(_normalize_tlr_op(op)))
@inline _logical_operand(A, ::Val{'N'}) = LogicalTLROperand{:N,typeof(A)}(A)
@inline _logical_operand(A, ::Val{'T'}) = LogicalTLROperand{:T,typeof(A)}(A)

@inline physical(A::LogicalTLROperand) = getfield(A, :parent)
@inline _orient_axes(::LogicalTLROperand{:N}, axes) = axes
@inline _orient_axes(::LogicalTLROperand{:T}, axes) = reverse(axes)
@inline _transpose_order(::TileColMajor) = TileRowMajor()
@inline _transpose_order(::TileRowMajor) = TileColMajor()

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
@inline nominal_tile_size(A::LogicalTLROperand, d::Integer) =
    nominal_tile_size(A)[Int(d)]
@inline tail_tile_size(A::LogicalTLROperand) =
    _orient_axes(A, tail_tile_size(physical(A)))
@inline tail_tile_size(A::LogicalTLROperand, d::Integer) =
    tail_tile_size(A)[Int(d)]
@inline grid_size(A::LogicalTLROperand) =
    _orient_axes(A, grid_size(physical(A)))
@inline regular_grid_size(A::LogicalTLROperand) =
    _orient_axes(A, regular_grid_size(physical(A)))
@inline tile_order(A::LogicalTLROperand{:N}) = tile_order(physical(A))
@inline tile_order(A::LogicalTLROperand{:T}) =
    _transpose_order(tile_order(physical(A)))
@inline maxrank(A::LogicalTLROperand) = maxrank(physical(A))
@inline KernelAbstractions.get_backend(A::LogicalTLROperand) =
    get_backend(physical(A))

"""Logical row/column range of one TLR tile along `axis`."""
@inline function _tile_axis_range(A::LogicalTLROperand, tile::Int, axis::Int)
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    extent = tile_size(A, axis == 1 ? tile : 1, axis == 2 ? tile : 1)[axis]
    return first:(first + extent - 1)
end
