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
function tlr_gemm_int_by_rpanel(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha; beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    qkB = region_tile_count(B, _RIGHT)
    qkB == 0 && return C                         # no right panel (n_B % b == 0)
    qmA, _ = regular_tilegrid_size(A)
    qmA == qkB || return C                       # non-square

    _, nt = tilegrid_size(B)                    # output boundary tile-column index
    bm = nominal_tile_size(A, 1)
    rB = maxrank(B)
    BU = outer_factors(B, _RIGHT)
    BV = inner_factors(B, _RIGHT)

    # diagonal j=i:  C_right[i] = β·C_right + α·(A_ii W_i) Z_iᵀ   (folds β).
    ADWdiag = allocate(get_backend(A), T, bm, rB, qmA)
    precision_gemm_batched!(_opchar(A), 'N', one(T),
        [_dense_data(_diag_tile_ref(A, i)) for i in 1:qmA],
        [view(BU, :, :, i) for i in 1:qmA],
        zero(T), [view(ADWdiag, :, :, i) for i in 1:qmA], compute)
    precision_gemm_batched!('N', 'T', alpha,
        [view(ADWdiag, :, :, i) for i in 1:qmA],
        [view(BV, :, :, i) for i in 1:qmA],
        beta, [_output_tile_view(C, A, B, i, nt) for i in 1:qmA], compute)

    # The dense diagonal above is the first writer. The low-rank off-diagonal leaf is a
    # `SkipDiag` interior and lowers through the same ContractOp as a FullGrid operand,
    # declaring accumulation so β is not applied twice.
    op = int_by_rpanel_contract(C, A, B, alpha, AccumulateExisting(typeof(alpha)))
    return execute!(lower(op; compute, budget))
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
# Budget blocks the free row axis `i` (scratch was `O(q_m)` with no knob). `γ_B` is a
# *dense* leaf, so this stays the two-stage lowering — no identity factor is formed — and
# the corner tile is broadcast across the batch.
function tlr_gemm_rpanel_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    qmA = region_tile_count(A, _RIGHT)
    qmA == 0 && return C                      # no right panel (n_A % b == 0)
    size(physical(B).D_corner, 3) == 0 && return C
    op = rpanel_by_corner_contract(C, A, B, alpha, ScaleExisting(beta))
    return execute!(lower(op; compute, budget))
end

# ── Fully low-rank variants (TLRMatrix) ──────────────────────────────────────
#
# Every tile is low-rank, so there is no dense diagonal to split out: `A_int u_B`
# reduces over ALL contraction tiles `k`, and `u_A γ_B` uses a low-rank corner `γ_B`.

# A_int u_B is the first boundary contraction lowered from `ContractOp`. Its domain is
# `(regular i, regular k, boundary j)`; the right-panel leaf exposes `k` as its contiguous
# iterator, and `DenseOutput` maps the scheduler's local `j=1` back to B's physical tail
# tile-column. The same row/serial reduction families and promoted S/T workspaces used by
# the interior therefore execute this term without a boundary-specific stage loop.
function tlr_gemm_int_by_rpanel(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    op = int_by_rpanel_contract(C, A, B, alpha, ScaleExisting(beta))
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    return execute!(lower(op; compute, budget))
end

# u_A γ_B:  C_right[i] += A_{i,bnd} γ_B,  i = 1:q_m^A.
#
# The `(1:q_m, bnd, bnd)` corner: A's right panel against B's low-rank corner. The
# reduction axis is a single tile, so there is nothing to reduce — the whole term is
# Stage 1/2/3 batched over the free row axis, with the corner's factors broadcast. Budget
# blocks `i`; each `i` writes a distinct output tile, so β folds in Stage 3 for every block
# rather than only the first.
function tlr_gemm_rpanel_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    op = rpanel_by_corner_contract(C, A, B, alpha, ScaleExisting(beta))
    (isempty(op.domain.i) || isempty(op.domain.j)) && return C
    return execute!(lower(op; compute, budget))
end
