# ─── Right region of C:  C_right = A_int u_B + u_A γ_B  (only when n%b ≠ 0) ────────
#
# The right region is the last tile-column of C (rows 1..Q·b, the boundary column),
# a `Q·b × s` block.  Splitting the operands around the partial last tile,
#   A = [A_int  u_A ;  v_Aᵀ  γ_A],   B = [B_int  u_B ;  v_Bᵀ  γ_B],
# gives  C_right = A_int u_B + u_A γ_B.  Two terms:
#   `tlr_gemm_int_by_rpanel`    — A_int u_B, interior of A × B's right panel (a
#                                 reduction over j).
#   `tlr_gemm_rpanel_by_corner` — u_A γ_B, A's right panel × B's corner (batched).

"""
    tlr_gemm_int_by_rpanel(C, A, B, alpha; beta=1, budget) -> C

Accumulate `A_int u_B` into the right region: `C_right := beta·C_right + α·A_int u_B`.
For each output tile-row `i`, `A_int u_B[i] = Σ_j A_ij B_{j,Q+1}` reduces over j:
  * diagonal j=i:  `A_ii B_{i,Q+1} = (A_ii W_i) Z_iᵀ`   — first writer, folds β.
  * off-diagonal:  `Σ_{j≠i} U_ij (V_ijᵀ W_j) Z_jᵀ`      — accumulates (β=1) via the
    usual three stages (S = VᵀW, T = SZᵀ, then the K-reduction over j), with the
    reduction looped and the free row axis batched, budget-split over rows.
First writer of C_right, so it folds β.  No-op when `n_B % b == 0`.
"""
function tlr_gemm_int_by_rpanel(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha; beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qkB = region_tile_count(B, _RIGHT)
    qkB == 0 && return C                         # no right panel (n_B % b == 0)
    qmA, _ = regular_grid_size(A)
    qmA == qkB || return C                       # non-square

    _, nt = grid_size(B)                    # output boundary tile-column index
    bm = nominal_tile_size(A, 1)
    rB = maxrank(B)
    b_outer_factors = outer_factors(B, _RIGHT)
    b_inner_factors = inner_factors(B, _RIGHT)

    # diagonal j=i:  C_right[i] = β·C_right + α·(A_ii W_i) Z_iᵀ   (folds β).
    _arena_reset!(arena)
    a_diag_b_outer_work = _workspace_array!(arena, get_backend(A), T, bm, rB, qmA)
    precision_gemm_batched!(_opchar(A), 'N', one(T),
        [_dense_data(_diag_tile_ref(A, i)) for i in 1:qmA],
        [view(b_outer_factors, :, :, i) for i in 1:qmA],
        zero(T), [view(a_diag_b_outer_work, :, :, i) for i in 1:qmA], compute)
    precision_gemm_batched!('N', 'T', alpha,
        [view(a_diag_b_outer_work, :, :, i) for i in 1:qmA],
        [view(b_inner_factors, :, :, i) for i in 1:qmA],
        beta, [_output_tile_view(C, A, B, i, nt) for i in 1:qmA], compute)

    # The dense diagonal above is the first writer. The budgeted `SkipDiag` low-rank
    # contraction accumulates, so β is not applied twice.
    return execute_lowrank_term!(C, A, B, _interior_pair(A), _right_pair(B),
                                 qmA, qkB, 1, 1, nt;
                                 alpha, beta=one(alpha), budget, compute, arena)
end

"""
    tlr_gemm_rpanel_by_corner(C, A, B, alpha; beta=1) -> C

Accumulate `α · u_A γ_B` into the right region of `C`.  A's right-panel tiles
`A_{i,Q+1} = U_i V_iᵀ` times B's dense corner `γ_B` give, for each i,
`A_{i,Q+1} γ_B = U_i (V_iᵀ γ_B)` — a two-stage batched product over i (Q tiles):
  Stage 1 (strided, γ_B broadcast):  M_i = V_iᵀ γ_B   (rA×s_n)
  Stage 2 (batched over i):          C_{i,Q+1} += α · U_i M_i   (b×s_n)
No-op when `n_A % b == 0`, `A.maxrank == 0`, or B has no corner.
"""
# Budget blocks the free row axis `i` (workspace was `O(qm)` with no knob). `γ_B` is a
# dense corner uses a direct two-stage helper — no identity factor is formed — and the
# corner tile is broadcast across the batch.
function tlr_gemm_rpanel_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qmA = region_tile_count(A, _RIGHT)
    qmA == 0 && return C                      # no right panel (n_A % b == 0)
    size(physical(B).D_corner, 3) == 0 && return C
    _, nt = grid_size(B)
    a_outer_factors, a_inner_factors = _right_pair(A)
    b_dense_corner = _diag_tile_ref(B, ndiag_tiles(B))
    return execute_lowrank_dense_term!(C, A, B, a_outer_factors, a_inner_factors, b_dense_corner,
                                       qmA, 1, nt;
                                       alpha, beta, budget, compute, arena)
end

# ── Fully low-rank variants (PaddedFTLRMatrix) ──────────────────────────────────────
#
# Every tile is low-rank, so there is no dense diagonal to split out: `A_int u_B`
# reduces over ALL contraction tiles `k`, and `u_A γ_B` uses a low-rank corner `γ_B`.

# A_int u_B uses the regular low-rank core with a right-panel factor accessor and maps
# its single local output column directly to B's physical tail tile-column.
function tlr_gemm_int_by_rpanel(C, A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qm, qk = regular_grid_size(A)
    qk == region_tile_count(B, _RIGHT) || return C
    _, nt = grid_size(B)
    nt > regular_grid_size(B)[2] || return C
    return execute_lowrank_term!(C, A, B, _interior_pair(A), _right_pair(B),
                                 qm, qk, 1, 1, nt;
                                 alpha, beta, budget, compute, arena)
end

# u_A γ_B:  C_right[i] += A_{i,bnd} γ_B,  i = 1:qm^A.
#
# The `(1:qm, bnd, bnd)` corner: A's right panel against B's low-rank corner. The
# reduction axis is a single tile, so there is nothing to reduce — the whole term is
# Stage 1/2/3 batched over the free row axis, with the corner's factors broadcast. Budget
# blocks `i`; each `i` writes a distinct output tile, so β folds in Stage 3 for every block
# rather than only the first.
function tlr_gemm_rpanel_by_corner(C, A::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:PaddedFTLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qm = region_tile_count(A, _RIGHT)
    qm == 0 && return C
    _, nt = grid_size(B)
    nt > regular_grid_size(B)[2] || return C
    return execute_lowrank_term!(C, A, B, _right_pair(A), _corner_pair(B),
                                 qm, 1, 1, 1, nt;
                                 alpha, beta, budget, compute, arena)
end
