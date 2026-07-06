"""
    PanelView{Side,Factor,Ax}

Zero-copy view wrapper over an operand's flat uniform-core factor storage
`[b, maxrank, n_off]`. `Ax` names the contiguous logical tile axis.
"""
abstract type OperandSide end
struct LeftOperand <: OperandSide end
struct RightOperand <: OperandSide end

abstract type FactorKind end
struct VFactor <: FactorKind end
struct WFactor <: FactorKind end
struct ZFactor <: FactorKind end

struct PanelView{Side<:OperandSide,Factor<:FactorKind,Ax,M<:TLRMatrix,A3<:AbstractArray}
    matrix::M
    data::A3
    noff::Int
    contig::Stride1Axis{Ax}
end

@inline blockdim(p::PanelView) = size(p.data, 1)   # b
@inline rankdim(p::PanelView)  = size(p.data, 2)   # maxrank

"""
Interior sub-grid `(Q, Q)`, `Q = ⌊m/b⌋` — the block of full `b×b` tiles, excluding
the `m%b` boundary row/column (which are separate terms). Equals `tilegrid_size`
when `m%b == 0`, so the uniform path is unchanged. `int_U`/`int_V` store exactly
this sub-grid's off-diagonal tiles in `Q×Q` enumeration order.
"""
@inline _interior_grid(A::TLRMatrix) = (q = fld(A.m, A.nominal_tile_size); (q, q))

"""Contiguous `[b, maxrank, noff]` view of tile-row `r`'s off-diagonal panel."""
@inline rowpanel(p::PanelView, r::Integer) =
    view(p.data, :, :, (r - 1) * p.noff + 1 : r * p.noff)

"""Zero-copy factor view for logical interior off-diagonal tile `(i,j)`."""
@inline tilefactor(p::PanelView, i::Integer, j::Integer) =
    view(p.data, :, :, _offdiag_index(tile_order(p.matrix), p.noff + 1, p.noff + 1, Int(i), Int(j)))

"""Tile-column index of local panel position `pos` within row `r` (skips `r`)."""
@inline local_to_col(r::Integer, pos::Integer) = pos < r ? pos : pos + 1

# `_offdiag_coords` (tile coords of off-diagonal slot `ob`) lives in tlrmatrix.jl.

# Construction

"""
    panelview(Side, Factor, M, factors, contig) -> PanelView

Wrap one uniform-core factor array as a zero-copy `PanelView`. No factor data is
copied or packed.
"""
function panelview(::Type{Side}, ::Type{Factor}, M::TLRMatrix{<:Any,T},
                   factors::AbstractArray{T,3}, contig::Stride1Axis{Ax}) where {Side<:OperandSide,Factor<:FactorKind,T,Ax}
    _, nt = _interior_grid(M)
    return PanelView{Side,Factor,Ax,typeof(M),typeof(factors)}(M, factors, nt - 1, contig)
end

# ─── Logical operands ───────────────────────────────────────────────────────────
#
# Thin typed wrappers so stage builders can name operands ("use V' · W") instead
# of spelling raw `view(A.int_V, …)` / `view(B.int_U, …)`.

"""
    LogicalTLROperands(av, bw, bz)

The uniform-core factor panels used by the staged off-diagonal product, named
for the formulas `A_ik = U_ik V_ik'` and `B_kj = W_kj Z_kj'`:
`av = V` (from `A`); `bw = W`, `bz = Z` (from `B`). Stage 3 uses the
workspace-stacked `U` views directly.
"""
struct LogicalTLROperands{AV,BW,BZ}
    av::AV
    bw::BW
    bz::BZ
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

Wrap the uniform-core factor arrays of `A` and `B` as zero-copy `PanelView`s.
"""
function logical_operands(A::TLRMatrix, B::TLRMatrix)
    return LogicalTLROperands(
        panelview(LeftOperand, VFactor, A, A.int_V, stride1_axis_left(A)),
        panelview(RightOperand, WFactor, B, B.int_U, stride1_axis_right(B)),
        panelview(RightOperand, ZFactor, B, B.int_V, stride1_axis_right(B)),
    )
end

"""Zero-copy `b×b` view of dense output tile `(i, j)`."""
@inline dense_tile(C::AbstractMatrix, b::Int, i::Int, j::Int) =
    view(C, (i - 1) * b + 1:i * b, (j - 1) * b + 1:j * b)

"""Zero-copy view of dense output row-block `i`, columns `j0:j1`."""
@inline dense_rowblock(C::AbstractMatrix, b::Int, i::Int, j0::Int, j1::Int) =
    view(C, (i - 1) * b + 1:i * b, (j0 - 1) * b + 1:j1 * b)
