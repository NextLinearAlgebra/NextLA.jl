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
function tlr_gemm_bpanel_by_int(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T; beta::T=one(T), budget::Int) where {T, BackendT}
    Q = size(A.bottom_U, 3)                    # A bottom-panel tiles (= interior Q)
    Q == 0 && return C                         # no bottom panel (m_A % b == 0)

    mt, _ = tilegrid_size(A)                    # boundary tile-row index (Q+1)
    b = nominal_tile_size(A, 2)
    rA = maxrank(A)
    rB = maxrank(B)
    s = size(A.bottom_U, 1)                     # panel height (= m_A % b)

    # Diagonal i=j:  C_bottom[j] = β·C_bottom + α·U_j (V_jᵀ B_jj)   (first writer, folds β).
    Ndiag = allocate(A.backend, T, rA, b, Q)
    gemm_batched!('T', 'N', one(T),
        [view(A.bottom_V,:,:,j) for j in 1:Q],
        [_diag_tile_view(B, j) for j in 1:Q],
        zero(T), [view(Ndiag,:,:,j) for j in 1:Q])
    gemm_batched!('N', 'N', alpha,
        [view(A.bottom_U,:,:,j) for j in 1:Q],
        [view(Ndiag,:,:,j) for j in 1:Q],
        beta, [_dense_tile_view(C, A, mt, j) for j in 1:Q])

    # Off-diagonal reduction (β = 1), budget-split over output columns.
    (rA == 0 || rB == 0) && return C
    per = Q - 1                                 # off-diagonal tiles per column
    per == 0 && return C
    order = tile_order(B)
    percol = max(rA * per * (rB + b) * sizeof(T), 1)
    maxJ = clamp(div(budget, percol), 1, Q)
    S = allocate(A.backend, T, rA, rB, per, maxJ)
    Tw = allocate(A.backend, T, rA, b, per, maxJ)

    @inbounds for jrange in Iterators.partition(1:Q, maxJ)
        j0 = first(jrange)
        # Stage 1:  S_{j,kk} = V_iᵀ W_ij,  batched over (column j, off-diag position kk).
        gemm_batched!('T', 'N', one(T),
            [view(A.bottom_V,:,:,local_to_col(j, kk)) for j in jrange for kk in 1:per],
            [view(B.int_U,:,:,_offdiag_index(order, Q, Q, local_to_col(j, kk), j)) for j in jrange for kk in 1:per],
            zero(T),
            [view(S,:,:,kk,(j-j0+1)) for j in jrange for kk in 1:per])
        # Stage 2:  T_{j,kk} = S_{j,kk} Z_ijᵀ,  same batch.
        gemm_batched!('N', 'T', one(T),
            [view(S,:,:,kk,(j-j0+1)) for j in jrange for kk in 1:per],
            [view(B.int_V,:,:,_offdiag_index(order, Q, Q, local_to_col(j, kk), j)) for j in jrange for kk in 1:per],
            zero(T),
            [view(Tw,:,:,kk,(j-j0+1)) for j in jrange for kk in 1:per])
        # Stage 3:  C_bottom[j] += α·Σ_i U_i T_{j,kk}.  Loop the reduction (kk); each kk
        # writes distinct columns j, so batch over j and accumulate with β = 1.
        for kk in 1:per
            gemm_batched!('N', 'N', alpha,
                [view(A.bottom_U,:,:,local_to_col(j, kk)) for j in jrange],
                [view(Tw,:,:,kk,(j-j0+1)) for j in jrange],
                one(T),
                [_dense_tile_view(C, A, mt, j) for j in jrange])
        end
    end
    return C
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
function tlr_gemm_corner_by_bpanel(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT, T}, alpha::T; beta::T=one(T)) where {T, BackendT}
    Q = size(B.bottom_U, 3)
    Q == 0 && return C                        # no bottom panel (m_B % b == 0)
    rB = maxrank(B)
    (rB == 0 || size(A.D_corner, 3) == 0) && return C
    mt, _ = tilegrid_size(A)                   # boundary tile-row index (Q+1)
    s_m = size(A.D_corner, 1)                  # corner row extent (= m_A % b)

    # Stage 1:  N_j = γ_A W_j   (s_m×rB), strided over j with γ_A (batch-1) broadcast.
    N = allocate(A.backend, T, s_m, rB, Q)
    gemm_batched!('N', 'N', one(T), A.D_corner, B.bottom_U, zero(T), N)

    # Stage 2:  C_{Q+1,j} += α · N_j Z_jᵀ   (s_m×b), batched over j.
    Nvec = [view(N,:,:,j) for j in 1:Q]
    Zvec = [view(B.bottom_V,:,:,j) for j in 1:Q]
    Cvec = [_dense_tile_view(C, A, mt, j) for j in 1:Q]
    gemm_batched!('N', 'T', alpha, Nvec, Zvec, beta, Cvec)
    return C
end
