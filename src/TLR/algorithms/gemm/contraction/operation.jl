# Structured contraction operations (ROADMAP milestone 3, step 5).
#
# This is the semantic side of the semantic/schedule boundary. `ContractOp` describes
# *what* is computed and deliberately contains no fold, reduction placement, run,
# workspace, or backend call. `contraction/lowering.jl` adds those decisions and
# delegates scheduled stages to `lowering/stages.jl`.

"""
    DenseOutput(data, A, B, i, j)

Dense destination and output indexing map of a structured contraction. Scheduled
coordinates are local to the operation (`1:span_extent(i)`, `1:span_extent(j)`); the
spans map them back to logical tile coordinates, and canonical `A`/`B` supply exact
regular or tail tile geometry. No output view is materialised until lowering executes.
"""
struct DenseOutput{C<:AbstractMatrix,A,B,I<:AxisSpan,J<:AxisSpan}
    data::C
    left_geometry::A
    right_geometry::B
    i::I
    j::J
end

@inline function dense_output(C::AbstractMatrix, A::LogicalTLROperand,
                              B::LogicalTLROperand, d::ContractDomain)
    return DenseOutput{typeof(C),typeof(A),typeof(B),typeof(d.i),typeof(d.j)}(
        C, A, B, d.i, d.j)
end

@inline span_coord(s::AxisSpan, pos::Integer) = first(span_range(s)) + Int(pos) - 1

"""Dense tile addressed in operation-local output coordinates."""
@inline function output_tile(output::DenseOutput, i::Integer, j::Integer)
    li = span_coord(output.i, i)
    lj = span_coord(output.j, j)
    return _output_tile_view(output.data, output.left_geometry, output.right_geometry, li, lj)
end

"""Contiguous dense output block addressed in operation-local tile coordinates."""
@inline function output_block(output::DenseOutput, i0::Integer, i1::Integer,
                              j0::Integer, j1::Integer)
    li0 = span_coord(output.i, i0)
    li1 = span_coord(output.i, i1)
    lj0 = span_coord(output.j, j0)
    lj1 = span_coord(output.j, j1)
    A = output.left_geometry
    B = output.right_geometry
    p0 = (li0 - 1) * nominal_tile_size(A, 1) + 1
    q0 = (lj0 - 1) * nominal_tile_size(B, 2) + 1
    p1 = (li1 - 1) * nominal_tile_size(A, 1) + tile_size(A, li1, 1)[1]
    q1 = (lj1 - 1) * nominal_tile_size(B, 2) + tile_size(B, 1, lj1)[2]
    return view(output.data, p0:p1, q0:q1)
end

@inline output_region(output::DenseOutput) =
    output_block(output, 1, span_extent(output.i), 1, span_extent(output.j))

"""One output tile-row across a local scheduled column range."""
@inline output_rowblock(output::DenseOutput, i::Integer, j0::Integer, j1::Integer) =
    output_block(output, i, i, j0, j1)

"""
    ContractOp(domain, left, right, output, init, alpha)

One structured tile contraction

`output[i,j] = init(output[i,j]) + alpha * sum(k, left[i,k] * right[k,j])`.

The domain owns the `(i,k,j)` iteration space, the leaves own logical-to-storage
indexing, and `init` owns first-writer/accumulator semantics. Scheduling is absent by
construction.
"""
struct ContractOp{D,L,R,O,I<:InitPolicy,A}
    domain::D
    left::L
    right::R
    output::O
    init::I
    alpha::A
end

@inline contract(domain::ContractDomain, left, right, output::DenseOutput,
                 init::InitPolicy, alpha) =
    ContractOp{typeof(domain),typeof(left),typeof(right),typeof(output),typeof(init),typeof(alpha)}(
        domain, left, right, output, init, alpha)

@inline corner_algebra_leaf(A::LogicalTLROperand{<:Any,<:TLRMatrix}) = corner_leaf(A)
@inline corner_algebra_leaf(A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}) =
    dense_corner_leaf(A)

"""
    _term_leaves(A, B) -> NamedTuple of (left, right)

The eight corners' operand leaves, keyed and ordered exactly like `contract_domains`.
This is the single source of the corner → leaf-pair mapping: both the per-term builders
and the workspace query read it, so they cannot disagree about which leaves a term
contracts.

A dense-diagonal operand's corner is a `DenseLeaf`, which is what selects the specialized
one-/two-stage lowerings instead of the generic three-stage one.
"""
@inline function _term_leaves(A::LogicalTLROperand, B::LogicalTLROperand)
    Aint = interior_leaf(interior_grid_kind(A), A)
    Bint = interior_leaf(interior_grid_kind(B), B)
    Aright, Abottom, Acorner = right_panel_leaf(A), bottom_panel_leaf(A), corner_algebra_leaf(A)
    Bright, Bbottom, Bcorner = right_panel_leaf(B), bottom_panel_leaf(B), corner_algebra_leaf(B)
    return (
        interior         = (Aint,    Bint),
        int_by_rpanel    = (Aint,    Bright),
        bpanel_by_int    = (Abottom, Bint),
        rpanel_by_bpanel = (Aright,  Bbottom),
        rpanel_by_corner = (Aright,  Bcorner),
        corner_by_bpanel = (Acorner, Bbottom),
        bpanel_by_rpanel = (Abottom, Bright),
        corner_by_corner = (Acorner, Bcorner),
    )
end

"""
    region_contract(Val(name), C, A, B, alpha, init) -> ContractOp

Build the structured contraction for one of the eight cube corners. `name` is a type
parameter so the domain and leaf lookups are compile-time constants.
"""
@inline function region_contract(::Val{name}, C::AbstractMatrix, A::LogicalTLROperand,
                                 B::LogicalTLROperand, alpha, init::InitPolicy) where {name}
    domain = contract_domains(A, B)[name]
    left, right = _term_leaves(A, B)[name]
    return contract(domain, left, right, dense_output(C, A, B, domain), init, alpha)
end

# Named entry points, one per corner. Each is the term's `(i,k,j)` span triple:
#   interior (1:q_m, 1:q_c, 1:q_n)   int_by_rpanel    (1:q_m, 1:q_c, bnd)
#   bpanel_by_int (bnd, 1:q_c, 1:q_n)  rpanel_by_bpanel (1:q_m, bnd, 1:q_n)
#   rpanel_by_corner (1:q_m, bnd, bnd) corner_by_bpanel (bnd, bnd, 1:q_n)
#   bpanel_by_rpanel (bnd, 1:q_c, bnd) corner_by_corner (bnd, bnd, bnd)
for name in (:interior, :int_by_rpanel, :bpanel_by_int, :rpanel_by_bpanel,
             :rpanel_by_corner, :corner_by_bpanel, :bpanel_by_rpanel, :corner_by_corner)
    @eval @inline $(Symbol(name, :_contract))(C::AbstractMatrix, A::LogicalTLROperand,
            B::LogicalTLROperand, alpha, init::InitPolicy) =
        region_contract(Val($(QuoteNode(name))), C, A, B, alpha, init)
end
