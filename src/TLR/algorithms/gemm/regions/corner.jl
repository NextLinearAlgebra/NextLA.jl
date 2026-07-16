# Corner output region:
#
#   C_corner = A_corner * B_corner
#            + sum(k, A_bottom[boundary,k] * B_right[k,boundary]).
#
# Both terms now emit the same structured contraction used by the interior and the
# other boundary regions. Low-rank corner leaves use the generic three-stage lowering;
# dense-diagonal corners select the one-stage Dense × Dense lowering.

"""
    tlr_gemm_corner_by_corner(C, A, B, alpha; beta=1) -> C

Accumulate the single logical corner × corner contraction into `C`'s corner tile.
This is the corner region's first writer and therefore folds `beta`.
"""
function tlr_gemm_corner_by_corner(C,
        A::LogicalTLROperand{<:Any,<:AbstractTLRMatrix{<:Any,T}},
        B::LogicalTLROperand, alpha;
        beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    op = corner_by_corner_contract(C, A, B, alpha, ScaleExisting(beta))
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    return execute!(lower(op; compute, budget=1))
end

"""
    tlr_gemm_bpanel_by_rpanel(C, A, B, alpha; beta=1, budget) -> C

Accumulate the reduction of A's logical bottom panel with B's logical right panel into
`C`'s corner tile. The lowering keeps the shared panel index as a budget-blocked serial
reduction and K-stacks the factors within each run; only the first run applies `beta`.
"""
function tlr_gemm_bpanel_by_rpanel(C,
        A::LogicalTLROperand{<:Any,<:AbstractTLRMatrix{<:Any,T}},
        B::LogicalTLROperand, alpha;
        beta=one(alpha), budget::Int,
        compute=default_gemm_compute_mode(T)) where {T}
    op = bpanel_by_rpanel_contract(C, A, B, alpha, ScaleExisting(beta))
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    return execute!(lower(op; compute, budget))
end
