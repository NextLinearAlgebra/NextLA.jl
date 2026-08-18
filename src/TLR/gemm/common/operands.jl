# Result-independent TLR operand normalization.

@inline function _normalize_tlr_op(op::Char)
    c = uppercase(op)
    c in ('N', 'T') || throw(ArgumentError(
        "TLR GEMM operation must be 'N' or 'T', got '$op'"))
    return c
end

@inline function logical_operand(A::AbstractTLRMatrix, op::Char='N')
    return _normalize_tlr_op(op) == 'N' ? A : transpose(A)
end

"""Logical row/column range of one TLR tile along `axis`."""
@inline function _tile_axis_range(A::AbstractTLRMatrix, tile::Int, axis::Int)
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    extent = tile_size(A, axis == 1 ? tile : 1, axis == 2 ? tile : 1)[axis]
    return first:(first + extent - 1)
end
