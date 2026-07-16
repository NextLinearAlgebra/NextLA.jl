# Iteration domains 
# The eight GEMM terms are the eight corners of the `(i,k,j) ∈ {regular, boundary}³`
# cube, and the four regions of `C` are that cube projected onto `(i,j)`. Each axis
# independently either ranges over the regular tiles or is pinned to the single tail
# tile, so one parameterised span triple describes every term:
#
#     interior          (1:q_m, 1:q_c, 1:q_n)      rpanel_by_corner  (1:q_m, bnd,   bnd  )
#     int_by_rpanel     (1:q_m, 1:q_c, bnd  )      corner_by_bpanel  (bnd,   bnd,   1:q_n)
#     bpanel_by_int     (bnd,   1:q_c, 1:q_n)      bpanel_by_rpanel  (bnd,   1:q_c, bnd  )
#     rpanel_by_bpanel  (1:q_m, bnd,   1:q_n)      corner_by_corner  (bnd,   bnd,   bnd  )
#
# A domain is **pure geometry**: it says which tile coordinates a term covers, not which
# of them hold a stored tile. Diagonal skipping stays a leaf property (`GridKind`), where
# it already lives, and the staged core applies it through `panel_col`/`col_scratch_pos`.
# Keeping the two apart is what makes the eight domains an exact partition of the
# tile-triple space for *both* containers — asserted in `test/TLR/gemm_ir.jl`, which owns
# the coordinate enumeration it checks against, since nothing here schedules through one.
#
# Emptiness is derived, not checked: a tile-aligned axis has no tail, so its boundary
# span is empty and every term touching it drops out. That replaces the `qkB == 0 &&
# return C` guard each term currently opens with.
#
# `ContractOp` now lowers the all-regular interior through this domain. The other seven
# corners already use its geometry for budgeting and migrate to operation-level lowering
# next.

# ─── Axis spans ───────────────────────────────────────────────────────────────

"""
    AxisSpan

Which tiles one logical axis of a contraction covers: either the regular tiles or the
single boundary (tail) tile.
"""
abstract type AxisSpan end

"""Tiles `1:q` of an axis's regular grid. Empty when `q == 0`."""
struct RegularSpan <: AxisSpan
    q::Int
end

"""
The single tail tile at grid index `idx`, or nothing when the axis is tile-aligned
(`idx == 0`). A tile-aligned axis makes every term that pins it empty — which is exactly
the boundary decomposition switching itself off.
"""
struct BoundarySpan <: AxisSpan
    idx::Int
end

@inline span_range(s::RegularSpan) = 1:s.q
@inline span_range(s::BoundarySpan) = s.idx == 0 ? (1:0) : (s.idx:s.idx)

"""Number of tiles the span covers."""
@inline span_extent(s::AxisSpan) = length(span_range(s))

@inline Base.isempty(s::AxisSpan) = span_extent(s) == 0

# ─── Contraction domain ───────────────────────────────────────────────────────

"""
    ContractDomain(i, k, j)

The set of logical tile coordinates one structured contraction visits: parallel
iterators `i` and `j`, reduction iterator `k`. Concretely parameterised on the three
span types so a term tuple stays inferable and the loops over it unroll.
"""
struct ContractDomain{I<:AxisSpan,K<:AxisSpan,J<:AxisSpan}
    i::I
    k::K
    j::J
end

@inline Base.isempty(d::ContractDomain) = isempty(d.i) || isempty(d.k) || isempty(d.j)

# ─── The eight corners ────────────────────────────────────────────────────────

"""
    contract_domains(A, B) -> NamedTuple of ContractDomain

The eight corners of the `(i,k,j) ∈ {regular, boundary}³` cube for canonical operands
`op(A)` and `op(B)`, keyed by the term each corresponds to today. Geometry comes from
the operands' effective (transpose-folded) grids, so no `Op` branch is needed.

The contraction axis is shared: `op(A)`'s column grid is `op(B)`'s row grid, so both
agree on `q_c` and on the tail index — `gemm!` validates the tiling that guarantees it.

Returns a `NamedTuple` of eight distinct concrete types, so a loop over `values(...)`
unrolls and each term specialises.
"""
function contract_domains(A::LogicalTLROperand, B::LogicalTLROperand)
    qmA, qcA = regular_tilegrid_size(A)
    qcB, qnB = regular_tilegrid_size(B)
    mtA, ktA = tilegrid_size(A)
    ktB, ntB = tilegrid_size(B)

    qcA == qcB && ktA == ktB ||
        throw(DimensionMismatch("op(A)'s column tile grid must equal op(B)'s row tile grid: " *
                                "got $((qcA, ktA)) and $((qcB, ktB))"))

    # A tail exists iff the full grid is longer than the regular grid; its index is the
    # last grid position. No tail ⇒ index 0 ⇒ empty span ⇒ the term drops out.
    ireg = RegularSpan(qmA)
    kreg = RegularSpan(qcA)
    jreg = RegularSpan(qnB)
    ibnd = BoundarySpan(mtA > qmA ? mtA : 0)
    kbnd = BoundarySpan(ktA > qcA ? ktA : 0)
    jbnd = BoundarySpan(ntB > qnB ? ntB : 0)

    return (
        interior         = ContractDomain(ireg, kreg, jreg),
        int_by_rpanel    = ContractDomain(ireg, kreg, jbnd),
        bpanel_by_int    = ContractDomain(ibnd, kreg, jreg),
        rpanel_by_bpanel = ContractDomain(ireg, kbnd, jreg),
        rpanel_by_corner = ContractDomain(ireg, kbnd, jbnd),
        corner_by_bpanel = ContractDomain(ibnd, kbnd, jreg),
        bpanel_by_rpanel = ContractDomain(ibnd, kreg, jbnd),
        corner_by_corner = ContractDomain(ibnd, kbnd, jbnd),
    )
end

@inline contract_domains(A::AbstractTLRMatrix, B::AbstractTLRMatrix) =
    contract_domains(logical_operand(A), logical_operand(B))
