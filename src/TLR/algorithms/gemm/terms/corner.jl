# C_corner = γ_A γ_B + v_Aᵀ u_B  (only when m%b ≠ 0) ──────
"""
    tlr_gemm_corner_by_corner(C, A, B, alpha; beta=1) -> C

Dense corner product `C_corner := beta·C_corner + α·γ_A γ_B` (a single small GEMM).
First corner writer, so it folds β.  No-op when there is no boundary corner
(`m % b == 0`).
"""
function tlr_gemm_corner_by_corner(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T, BackendT}
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
function tlr_gemm_bpanel_by_rpanel(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T, BackendT}
    Q = size(A.bottom_U, 3)
    Q == 0 && return C                        # no bottom panel (m % b == 0)
    Q == size(B.right_U, 3) || return C       # non-square boundary

    rA = maxrank(A)
    rB = maxrank(B)
    (rA == 0 || rB == 0) && return C

    mt, nt = tilegrid_size(A)
    s_m = size(A.bottom_U, 1) # tail tile row
    s_n = size(B.right_V, 1)  # tail tile column

    S = allocate(A.backend, T, rA, rB, Q)
    Tw = allocate(A.backend, T, rA, s_n, Q)

    # Stage 1 (strided):  S_p = V_pᵀ W_p
    gemm_batched!('T', 'N', one(T), A.bottom_V, B.right_U, zero(T), S)
    # Stage 2 (strided):  T_p = S_p Z_pᵀᵀ
    gemm_batched!('N', 'T', one(T), S, B.right_V, zero(T), Tw)
    # Stage 3 (GEMM): Σ_p U_p T_p, K-stacked over p.  Tw is [rA, s_n, Q]
    Ustack = reshape(A.bottom_U, s_m, Q * rA)
    Tstack = reshape(permutedims(Tw, (1, 3, 2)), Q * rA, s_n) #TODO permutation is a copy, fix it
    Ccorner = _dense_tile_view(C, A, mt, nt)
    gemm_batched!('N', 'N', alpha, [Ustack], [Tstack], beta, [Ccorner])
    return C
end
