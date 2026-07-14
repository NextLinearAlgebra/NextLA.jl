# C_corner = γ_A γ_B + v_Aᵀ u_B  (only when m%b ≠ 0) ──────
"""
    tlr_gemm_corner_by_corner(C, A, B, alpha; beta=1) -> C

Dense corner product `C_corner := beta·C_corner + α·γ_A γ_B` (a single small GEMM).
First corner writer, so it folds β.  No-op when there is no boundary corner
(`m % b == 0`).
"""
function tlr_gemm_corner_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    (size(physical(A).D_corner, 3) != 0 && size(physical(B).D_corner, 3) != 0) || return C

    tile_k = ndiag_tiles(A)
    Atile = _diag_tile_ref(A, tile_k)
    Btile = _diag_tile_ref(B, tile_k)
    Ctile = _output_tile_view(C, A, B, tile_k, tile_k)
    precision_gemm_batched!(_dense_op(Atile), _dense_op(Btile), alpha,
                            [_dense_data(Atile)], [_dense_data(Btile)], beta, [Ctile], compute)
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
function tlr_gemm_bpanel_by_rpanel(C, A::LogicalTLROperand{<:Any,<:AbstractTLRMatrix{<:Any,T}}, B::LogicalTLROperand, alpha; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    qkA = region_tile_count(A, _BOTTOM)
    qkB = region_tile_count(B, _RIGHT)
    qkA == 0 && return C
    qkA == qkB || return C                    # contraction mismatch (q_c^A ≠ q_c^B)

    rA = maxrank(A)
    rB = maxrank(B)
    (rA == 0 || rB == 0) && return C

    AU = outer_factors(A, _BOTTOM)
    AV = inner_factors(A, _BOTTOM)
    BU = outer_factors(B, _RIGHT)
    BV = inner_factors(B, _RIGHT)

    mt, _ = tilegrid_size(A)                   # A boundary tile-row
    _, ntB = tilegrid_size(B)                  # B boundary tile-col
    sm = size(AU, 1)                           # tail tile row
    sn = size(BV, 1)                           # tail tile column

    Swork = allocate(get_backend(A), T, rA, rB, qkA)
    Twork = allocate(get_backend(A), T, rA, sn, qkA)

    # Stage 1 (strided):  S_p = V_pᵀ W_p
    precision_gemm_batched!('T', 'N', one(T), AV, BU, zero(T), Swork, compute)
    # Stage 2 (strided):  T_p = S_p Z_pᵀᵀ
    precision_gemm_batched!('N', 'T', one(T), Swork, BV, zero(T), Twork, compute)
    # Stage 3 (GEMM): Σ_p U_p T_p, K-stacked over p.  Twork is [rA, sn, qkA]
    Ustack = reshape(AU, sm, qkA * rA)
    Tstack = reshape(permutedims(Twork, (1, 3, 2)), qkA * rA, sn) #TODO permutation is a copy, fix it
    Ccorner = _output_tile_view(C, A, B, mt, ntB)
    precision_gemm_batched!('N', 'N', alpha, [Ustack], [Tstack], beta, [Ccorner], compute)
    return C
end

# ── Fully low-rank variant (TLRMatrix) ───────────────────────────────────────

# γ_A γ_B: low-rank corner × low-rank corner, a single 3-stage product (vs. the dense
# `mul!` of the dense-diagonal container).
function tlr_gemm_corner_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    AU = outer_factors(A, _CORNER)
    AV = inner_factors(A, _CORNER)
    BU = outer_factors(B, _CORNER)
    BV = inner_factors(B, _CORNER)
    (size(AU, 3) != 0 && size(BU, 3) != 0) || return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row
    _, ntB = tilegrid_size(B)                     # B boundary tile-col
    Cc = _output_tile_view(C, A, B, mt, ntB)
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && (_scale_output!(Cc, beta); return C)  # γ contributes 0; still fold β
    sn = size(BV, 1)

    Swork = allocate(get_backend(A), T, rA, rB, 1)    # S = Vc^A' Wc^B
    precision_gemm_batched!('T', 'N', one(T), AV, BU, zero(T), Swork, compute)
    Twork = allocate(get_backend(A), T, rA, sn, 1)    # T = S Zc^B'
    precision_gemm_batched!('N', 'T', one(T), Swork, BV, zero(T), Twork, compute)
    precision_gemm_batched!('N', 'N', alpha, [view(AU, :, :, 1)],
                            [view(Twork, :, :, 1)], beta, [Cc], compute)
    return C
end
