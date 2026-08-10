# Factor-panel adapters used only by PaddedFTLR accumulation.

"""Canonical full-rank-column factors of logical PaddedFTLR tile `(i,j)`."""
@inline function logical_tile_factors(
    A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix}, i::Int, j::Int)
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

"""Zero-copy view over flat interior factor storage and its tile grid."""
struct InteriorOperand{OrderT<:TileOrderStyle,A3<:AbstractArray}
    data::A3
    order::OrderT
    qm::Int
    qn::Int
end

@inline rankdim(p::InteriorOperand) = size(p.data, 2)
@inline tiles_per_row(p::InteriorOperand) = p.qn
@inline tiles_per_col(p::InteriorOperand) = p.qm

@inline function rowpanel(p::InteriorOperand, row::Integer)
    count = tiles_per_row(p)
    return view(p.data, :, :,
                (Int(row) - 1) * count + 1:Int(row) * count)
end

@inline function colpanel(p::InteriorOperand, column::Integer)
    count = tiles_per_col(p)
    return view(p.data, :, :,
                (Int(column) - 1) * count + 1:Int(column) * count)
end

@inline tilefactor(p::InteriorOperand, i::Integer, j::Integer) =
    view(p.data, :, :,
         tile_linear_index(p.order, p.qm, p.qn, Int(i), Int(j)))

@inline function interior_operand(data::AbstractArray, order::TileOrderStyle,
                                  qm::Int, qn::Int)
    return InteriorOperand{typeof(order),typeof(data)}(data, order, qm, qn)
end

"""Interior factor panels for the implicit PaddedFTLR product operator."""
struct LogicalTLROperands{AV,BU,BV,AU}
    av::AV
    bu::BU
    bv::BV
    au::AU
end

logical_operands(A::AbstractTLRMatrix, B::AbstractTLRMatrix) =
    logical_operands(logical_operand(A), logical_operand(B))

function logical_operands(
    A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix},
    B::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix})
    qmA, qnA = regular_grid_size(A)
    qmB, qnB = regular_grid_size(B)
    ordA = tile_order(A)
    ordB = tile_order(B)
    return LogicalTLROperands(
        interior_operand(inner_factors(A, _INTERIOR), ordA, qmA, qnA),
        interior_operand(outer_factors(B, _INTERIOR), ordB, qmB, qnB),
        interior_operand(inner_factors(B, _INTERIOR), ordB, qmB, qnB),
        interior_operand(outer_factors(A, _INTERIOR), ordA, qmA, qnA),
    )
end
