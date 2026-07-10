# C_corner = γ_A γ_B + v_Aᵀ u_B  (only when m%b ≠ 0) ──────
"""
    tlr_gemm_corner_by_corner(C, A, B, alpha; beta=1) -> C

Dense corner product `C_corner := beta·C_corner + α·γ_A γ_B` (a single small GEMM).
First corner writer, so it folds β.  No-op when there is no boundary corner
(`m % b == 0`).
"""
function tlr_gemm_corner_by_corner(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T, BackendT}
    (size(A.D_corner, 3) != 0 && size(B.D_corner, 3) != 0) || return C

    tile_k = ndiag_tiles(A)
    mul!(_dense_tile_view(C, A, tile_k, tile_k),
         _diag_tile_view(A, tile_k), _diag_tile_view(B, tile_k), alpha, beta)
    return C
end

"""
    tlr_gemm_bpanel_by_rpanel(C, A, B, alpha; beta=1) -> C

Corner low-rank product `v_Aᵀ u_B = Σ_p A_{Q+1,p} B_{p,Q+1}`, accumulated into the
corner block `C_corner := beta·C_corner + α·v_Aᵀ u_B`.

A's bottom-panel tiles `A_{Q+1,p} = U_p V_pᵀ` and B's right-panel tiles
`B_{p,Q+1} = W_p Z_pᵀ` pair by the shared interior index `p` — which is the identity
in storage order for both tile layouts (bottom stored by column, right by row) — so
each product `A_{Q+1,p} B_{p,Q+1} = U_p (V_pᵀ W_p) Z_pᵀ` lowers to three fully batched
stages:
  Stage 1 (strided):  S_p = V_pᵀ W_p   (rA×rB, contract b)
  Stage 2 (strided):  T_p = S_p Z_pᵀ   (rA×s_n)
  Stage 3 (one GEMM): C_corner += α · Σ_p U_p T_p,  K-stacked over p.
where  U_p = A.bottom_U[p] (s_m×rA),  V_p = A.bottom_V[p] (b×rA),
       W_p = B.right_U[p]  (b×rB),    Z_p = B.right_V[p]  (s_n×rB).
No-op when `m % b == 0` or the panels are unpaired (non-square boundary).
"""
# Pure low-rank panels (A's bottom row × B's right column): identical for both
# container types, so it dispatches on `AbstractTLRMatrix`.
function tlr_gemm_bpanel_by_rpanel(C, A::AbstractTLRMatrix{BackendT,T}, B::AbstractTLRMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T, BackendT}
    qkA = _bottom_panel_tiles(A)
    qkB = _right_panel_tiles(B)
    qkA == 0 && return C
    qkA == qkB || return C                    # contraction mismatch (q_c^A ≠ q_c^B)

    rA = maxrank(A)
    rB = maxrank(B)
    (rA == 0 || rB == 0) && return C

    mt, _ = tilegrid_size(A)                   # A boundary tile-row
    _, ntB = tilegrid_size(B)                  # B boundary tile-col
    sm = size(A.bottom_U, 1)                   # tail tile row
    sn = size(B.right_V, 1)                    # tail tile column

    Swork = allocate(A.backend, T, rA, rB, qkA)
    Twork = allocate(A.backend, T, rA, sn, qkA)

    # Stage 1 (strided):  S_p = V_pᵀ W_p
    gemm_batched!('T', 'N', one(T), A.bottom_V, B.right_U, zero(T), Swork)
    # Stage 2 (strided):  T_p = S_p Z_pᵀᵀ
    gemm_batched!('N', 'T', one(T), Swork, B.right_V, zero(T), Twork)
    # Stage 3 (GEMM): Σ_p U_p T_p, K-stacked over p.  Twork is [rA, sn, qkA]
    Ustack = reshape(A.bottom_U, sm, qkA * rA)
    Tstack = reshape(permutedims(Twork, (1, 3, 2)), qkA * rA, sn) #TODO permutation is a copy, fix it
    Ccorner = _output_tile_view(C, A, B, mt, ntB)
    gemm_batched!('N', 'N', alpha, [Ustack], [Tstack], beta, [Ccorner])
    return C
end
