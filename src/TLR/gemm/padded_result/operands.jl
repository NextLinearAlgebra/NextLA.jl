# Factor-panel adapters used only by canonical (ARA-sampling) TLR accumulation.

"""
    _beta_tile_factors(C, i, j)

Canonical, uniform-width factors of logical tile `(i,j)`, read from an
already-populated *reserved-capacity* `CompressedFTLRMatrix` (the `C` this
GEMM writes into). Deliberately reads the *execution-rank*-width view, not
[`logical_tile_factors`](@ref)'s real-rank width — `C` is constructed with
uniform `execution_ranks` (its reserved capacity) that never change after
construction, so this stays uniformly capacity-wide across every tile for
`C`'s whole lifetime; that uniformity is required by the pointer/strided
-batched beta-accumulation GEMMs in `run_coupling.jl`, which batch multiple
tiles' factors together and therefore need identical widths across the batch.
The real per-tile rank width `logical_tile_factors` returns is, in general,
ragged and would break that assumption.
"""
@inline _beta_tile_factors(A::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix}, i::Int, j::Int) =
    (compressed_ftlr_execution_outer(A, i, j), compressed_ftlr_execution_inner(A, i, j))

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

"""
Build the implicit factor-list product operator over `A`, `B`'s packed
storage. `au`/`av` (`bu`/`bv`) are `A`'s (`B`'s) outer/inner factors reshaped
into dense per-tile-grid arrays via
[`_compressed_ftlr_uniform_view`](@ref) — valid because `padded_result`
requires a regular grid (`_validate_canonical_tlr_gemm`), so every tile's
stored rank is uniformly `A`'s (`B`'s) `execution_maxrank`.

Each panel's `InteriorOperand.order` is set from
`compressed_ftlr_outer_order`/`compressed_ftlr_inner_order` on the *logical*
operand, not the packed factor's own physical `.order` field: composing the
logical order (transpose-aware) with the physical slot addressing that
`outer_factors`/`inner_factors` already selected (swapping to the `.inner`/
`.outer` field under a logical `:T` view) is what makes `tile_linear_index`
land on the correct physical slot for a *logical* `(i,j)` — the same identity
the rest of the `LogicalTLROperand` machinery relies on throughout. Under the
default complementary packing this GEMM requires, both orders are in fact
transpose-invariant (`compressed_ftlr_outer_order` is always `TileRowMajor`,
`compressed_ftlr_inner_order` always `TileColMajor`, for either `'N'` or
`'T'`) — this is exactly what lets a single code path serve all four
transpose combinations without a zero-copy-availability branch.
"""
function logical_operands(
    A::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix},
    B::LogicalTLROperand{<:Any,<:CompressedFTLRMatrix})
    qmA, qnA = regular_grid_size(A)
    qmB, qnB = regular_grid_size(B)
    return LogicalTLROperands(
        interior_operand(_compressed_ftlr_uniform_view(inner_factors(A, _INTERIOR)),
                         compressed_ftlr_inner_order(A), qmA, qnA),
        interior_operand(_compressed_ftlr_uniform_view(outer_factors(B, _INTERIOR)),
                         compressed_ftlr_outer_order(B), qmB, qnB),
        interior_operand(_compressed_ftlr_uniform_view(inner_factors(B, _INTERIOR)),
                         compressed_ftlr_inner_order(B), qmB, qnB),
        interior_operand(_compressed_ftlr_uniform_view(outer_factors(A, _INTERIOR)),
                         compressed_ftlr_outer_order(A), qmA, qnA),
    )
end
