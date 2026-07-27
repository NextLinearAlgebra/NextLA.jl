# ─── Bottom region of C:  C_bottom = v_Aᵀ B_int + γ_A v_Bᵀ  (only when m%b ≠ 0) ────
"""
    tlr_gemm_bpanel_by_int(C, A, B, alpha; beta=1, budget) -> C

Accumulate `v_Aᵀ B_int` into the bottom region: `C_bottom := beta·C_bottom + α·v_Aᵀ B_int`.
For each output tile-column `j`, `v_Aᵀ B_int[j] = Σ_i A_{Q+1,i} B_ij` reduces over i:
  * diagonal i=j:  `A_{Q+1,j} B_jj = U_j (V_jᵀ B_jj)`   — first writer, folds β.
  * off-diagonal:  `Σ_{i≠j} U_i (V_iᵀ W_ij) Z_ijᵀ`      — accumulates (β=1) via the
    three stages (S = VᵀW, T = SZᵀ, K-reduction over i), reduction looped and the
    free column axis batched, budget-split over columns.
where `U_i = A.bottom_U[i] (s×rA)`, `V_i = A.bottom_V[i] (b×rA)` are the bottom-panel
factors and `W_ij, Z_ij` are B's interior factors.  First writer of C_bottom, folds β.
No-op when `m_A % b == 0`.
"""
function tlr_gemm_bpanel_by_int(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha; beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qkA = region_tile_count(A, _BOTTOM)        # A bottom-panel tiles (= interior columns)
    qkA == 0 && return C                       # no bottom panel (m_A % b == 0)

    mt, _ = tilegrid_size(A)                    # boundary tile-row index (Q+1)
    bn = nominal_tile_size(B, 2)
    rA = maxrank(A)
    AU = outer_factors(A, _BOTTOM)
    AV = inner_factors(A, _BOTTOM)

    # Diagonal i=j:  C_bottom[j] = β·C_bottom + α·U_j (V_jᵀ B_jj)   (first writer, folds β).
    _arena_reset!(arena)
    VDBdiag = _workspace_array!(arena, get_backend(A), T, rA, bn, qkA)
    precision_gemm_batched!('T', _opchar(B), one(T),
        [view(AV, :, :, j) for j in 1:qkA],
        [_dense_data(_diag_tile_ref(B, j)) for j in 1:qkA],
        zero(T), [view(VDBdiag, :, :, j) for j in 1:qkA], compute)
    precision_gemm_batched!('N', 'N', alpha,
        [view(AU, :, :, j) for j in 1:qkA],
        [view(VDBdiag, :, :, j) for j in 1:qkA],
        beta, [_output_tile_view(C, A, B, mt, j) for j in 1:qkA], compute)

    # Dense diagonal × bottom panel above owns β. The remaining bottom-panel ×
    # `SkipDiag` interior contraction accumulates through the shared low-rank core.
    qm, qn = regular_tilegrid_size(B)
    return execute_lowrank_term!(C, A, B, _bottom_pair(A), _interior_pair(B),
                                 1, qkA, qn, mt, 1;
                                 alpha, beta=one(alpha), budget, compute, arena)
end

"""
    tlr_gemm_corner_by_bpanel(C, A, B, alpha; beta=1) -> C

Accumulate `α · γ_A v_Bᵀ` into the bottom region of `C`.  A's dense corner `γ_A`
times B's bottom-panel tiles `B_{Q+1,j} = W_j Z_jᵀ` give, for each j,
`γ_A B_{Q+1,j} = (γ_A W_j) Z_jᵀ` — a two-stage batched product over j (Q tiles):
  Stage 1 (strided, γ_A broadcast):  N_j = γ_A W_j   (s_m×rB)
  Stage 2 (batched over j):          C_{Q+1,j} += α · N_j Z_jᵀ   (s_m×b)
No-op when `m_B % b == 0`, `B.maxrank == 0`, or A has no corner.
"""
# Budget blocks the free column axis `j` (scratch was `O(qn)` with no knob). `γ_A` is a
# dense corner uses a direct two-stage helper, with the corner tile broadcast.
function tlr_gemm_corner_by_bpanel(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qnB = region_tile_count(B, _BOTTOM)
    qnB == 0 && return C                      # no bottom panel (m_B % b == 0)
    size(physical(A).D_corner, 3) == 0 && return C
    mt, _ = tilegrid_size(A)
    outer, inner = _bottom_pair(B)
    dense = _diag_tile_ref(A, ndiag_tiles(A))
    return execute_dense_lowrank_term!(C, A, B, dense, outer, inner,
                                       qnB, mt, 1;
                                       alpha, beta, budget, compute, arena)
end

# ── Fully low-rank variants (TLRMatrix) ──────────────────────────────────────
#
# `v_Aᵀ B_int` reduces over ALL contraction tiles `k` (no dense diagonal), and
# `γ_A v_Bᵀ` uses a low-rank corner `γ_A`.

# v_Aᵀ B_int is the mirror direct boundary path. The bottom panel supplies a complete
# contiguous left k-stack and the local row maps to A's physical tail tile-row.
function tlr_gemm_bpanel_by_int(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qk = region_tile_count(A, _BOTTOM)
    qk == 0 && return C
    _, qn = regular_tilegrid_size(B)
    mt, _ = tilegrid_size(A)
    mt > regular_tilegrid_size(A)[1] || return C
    return execute_lowrank_term!(C, A, B, _bottom_pair(A), _interior_pair(B),
                                 1, qk, qn, mt, 1;
                                 alpha, beta, budget, compute, arena)
end

# γ_A v_Bᵀ:  C_bottom[j] += γ_A B_{bnd,j},  j = 1:qn^B.
#
# The mirror of `rpanel_by_corner`: the `(bnd, bnd, 1:qn)` corner, so A's low-rank corner
# is broadcast and the budget blocks the free *column* axis `j`. Each `j` writes a distinct
# output tile, so β folds in Stage 3 for every block.
function tlr_gemm_corner_by_bpanel(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qn = region_tile_count(B, _BOTTOM)
    qn == 0 && return C
    mt, _ = tilegrid_size(A)
    mt > regular_tilegrid_size(A)[1] || return C
    return execute_lowrank_term!(C, A, B, _corner_pair(A), _bottom_pair(B),
                                 1, 1, qn, mt, 1;
                                 alpha, beta, budget, compute, arena)
end
