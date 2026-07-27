# Corner output region:
#
#   C_corner = A_corner * B_corner
#            + sum(k, A_bottom[boundary,k] * B_right[k,boundary]).
#
# Low-rank corners use the shared three-stage core; dense-diagonal corners call one
# direct Dense × Dense GEMM.

"""
    tlr_gemm_corner_by_corner(C, A, B, alpha; beta=1) -> C

Accumulate the single logical corner × corner contraction into `C`'s corner tile.
This is the corner region's first writer and therefore folds `beta`.
"""
function tlr_gemm_corner_by_corner(C,
        A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}},
        B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
        beta=one(alpha), compute=default_gemm_compute_mode(T),
        budget::Int=1, arena=nothing) where {T}
    mt, kt = grid_size(A)
    _, nt = grid_size(B)
    (mt > regular_grid_size(A)[1] &&
     kt > regular_grid_size(A)[2] &&
     nt > regular_grid_size(B)[2]) || return C
    return execute_lowrank_term!(C, A, B, _corner_pair(A), _corner_pair(B),
                                 1, 1, 1, mt, nt;
                                 alpha, beta, budget, compute, arena)
end

function tlr_gemm_corner_by_corner(C,
        A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}},
        B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha;
        beta=one(alpha), compute=default_gemm_compute_mode(T),
        budget::Int=1, arena=nothing) where {T}
    (size(physical(A).D_corner, 3) == 0 ||
     size(physical(B).D_corner, 3) == 0) && return C
    mt, _ = grid_size(A)
    _, nt = grid_size(B)
    a_dense_corner = _diag_tile_ref(A, ndiag_tiles(A))
    b_dense_corner = _diag_tile_ref(B, ndiag_tiles(B))
    output_tile = _output_tile_view(C, A, B, mt, nt)
    return precision_gemm_batched!(_dense_op(a_dense_corner), _dense_op(b_dense_corner), alpha,
                                   [_dense_data(a_dense_corner)], [_dense_data(b_dense_corner)], beta,
                                   [output_tile], compute)
end

"""
    tlr_gemm_bpanel_by_rpanel(C, A, B, alpha; beta=1, budget) -> C

Accumulate the reduction of A's logical bottom panel with B's logical right panel into
`C`'s corner tile. The shared panel index is a budget-blocked serial reduction; only
the initial output scaling applies `beta`.
"""
function tlr_gemm_bpanel_by_rpanel(C,
        A::LogicalTLROperand{<:Any,<:AbstractTLRMatrix{<:Any,T}},
        B::LogicalTLROperand, alpha;
        beta=one(alpha), budget::Int,
        compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qk = region_tile_count(A, _BOTTOM)
    qk == region_tile_count(B, _RIGHT) || return C
    qk == 0 && return C
    mt, _ = grid_size(A)
    _, nt = grid_size(B)
    return execute_lowrank_term!(C, A, B, _bottom_pair(A), _right_pair(B),
                                 1, qk, 1, mt, nt;
                                 alpha, beta, budget, compute, fold=FoldRight(),
                                 placement=KAsSerialLoop{:k}(), arena)
end
