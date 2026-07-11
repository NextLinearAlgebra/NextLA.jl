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
    qkA = _bottom_panel_tiles(A)               # A bottom-panel tiles (= interior columns)
    qkA == 0 && return C                       # no bottom panel (m_A % b == 0)

    mt, _ = tilegrid_size(A)                    # boundary tile-row index (Q+1)
    bn = nominal_tile_size(B, 2)
    rA = maxrank(A)
    rB = maxrank(B)
    sm = size(A.bottom_U, 1)                    # panel height (= m_A % b)

    # Diagonal i=j:  C_bottom[j] = β·C_bottom + α·U_j (V_jᵀ B_jj)   (first writer, folds β).
    VDBdiag = allocate(A.backend, T, rA, bn, qkA)
    gemm_batched!('T', 'N', one(T),
        [view(A.bottom_V, :, :, j) for j in 1:qkA],
        [_diag_tile_view(B, j) for j in 1:qkA],
        zero(T), [view(VDBdiag, :, :, j) for j in 1:qkA])
    gemm_batched!('N', 'N', alpha,
        [view(A.bottom_U, :, :, j) for j in 1:qkA],
        [view(VDBdiag, :, :, j) for j in 1:qkA],
        beta, [_dense_tile_view(C, A, mt, j) for j in 1:qkA])

    # Off-diagonal reduction (β = 1), budget-split over output columns.
    (rA == 0 || rB == 0) && return C
    nk_off_per_col = qkA - 1
    nk_off_per_col == 0 && return C
    order = tile_order(B)
    bytes_per_j = max(rA * nk_off_per_col * (rB + bn) * sizeof(T), 1)
    maxJ = clamp(div(budget, bytes_per_j), 1, qkA)
    Swork = allocate(A.backend, T, rA, rB, nk_off_per_col, maxJ)
    Twork = allocate(A.backend, T, rA, bn, nk_off_per_col, maxJ)

    @inbounds for jrange in Iterators.partition(1:qkA, maxJ)
        j0 = first(jrange)
        # Stage 1:  S_{j,kpos} = V_iᵀ W_ij,  batched over column and off-diagonal k position.
        gemm_batched!('T', 'N', one(T),
            [view(A.bottom_V, :, :, local_to_col(j, kpos)) for j in jrange for kpos in 1:nk_off_per_col],
            [view(B.int_U, :, :, _offdiag_index(order, qkA, qkA, local_to_col(j, kpos), j)) for j in jrange for kpos in 1:nk_off_per_col],
            zero(T),
            [view(Swork, :, :, kpos, (j - j0 + 1)) for j in jrange for kpos in 1:nk_off_per_col])
        # Stage 2:  T_{j,kpos} = S_{j,kpos} Z_ijᵀ,  same batch.
        gemm_batched!('N', 'T', one(T),
            [view(Swork, :, :, kpos, (j - j0 + 1)) for j in jrange for kpos in 1:nk_off_per_col],
            [view(B.int_V, :, :, _offdiag_index(order, qkA, qkA, local_to_col(j, kpos), j)) for j in jrange for kpos in 1:nk_off_per_col],
            zero(T),
            [view(Twork, :, :, kpos, (j - j0 + 1)) for j in jrange for kpos in 1:nk_off_per_col])
        # Stage 3:  C_bottom[j] += α·Σ_i U_i T_{j,kpos}.  Loop the reduction; each kpos
        # writes distinct columns j, so batch over j and accumulate with β = 1.
        for kpos in 1:nk_off_per_col
            gemm_batched!('N', 'N', alpha,
                [view(A.bottom_U, :, :, local_to_col(j, kpos)) for j in jrange],
                [view(Twork, :, :, kpos, (j - j0 + 1)) for j in jrange],
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
    qnB = _bottom_panel_tiles(B)
    qnB == 0 && return C                      # no bottom panel (m_B % b == 0)
    rB = maxrank(B)
    (rB == 0 || size(A.D_corner, 3) == 0) && return C
    mt, _ = tilegrid_size(A)                   # boundary tile-row index (Q+1)
    sm = size(A.D_corner, 1)                   # corner row extent (= m_A % b)

    # Stage 1:  N_j = γ_A W_j   (s_m×rB), strided over j with γ_A (batch-1) broadcast.
    Swork = allocate(A.backend, T, sm, rB, qnB)
    gemm_batched!('N', 'N', one(T), A.D_corner, B.bottom_U, zero(T), Swork)

    # Stage 2:  C_{Q+1,j} += α · N_j Z_jᵀ   (s_m×b), batched over j.
    Svec = [view(Swork, :, :, j) for j in 1:qnB]
    Zvec = [view(B.bottom_V, :, :, j) for j in 1:qnB]
    Cvec = [_dense_tile_view(C, A, mt, j) for j in 1:qnB]
    gemm_batched!('N', 'T', alpha, Svec, Zvec, beta, Cvec)
    return C
end

# ── Fully low-rank variants (TLRMatrix) ──────────────────────────────────────
#
# `v_Aᵀ B_int` reduces over ALL contraction tiles `k` (no dense diagonal), and
# `γ_A v_Bᵀ` uses a low-rank corner `γ_A`.

# v_Aᵀ B_int:  C_bottom[j] = Σ_{k=1:q_c} A_{bnd,k} B_kj,  j = 1:q_n^B.  Batched Stages
# 1/2 over (j, k); Stage 3 loops the reduction `k` (batched over j), first `k` folds β.
function tlr_gemm_bpanel_by_int(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T), budget::Int) where {T,BackendT}
    qkA = _bottom_panel_tiles(A)                 # one per interior col k
    qkA == 0 && return C
    qkB, qnB = _full_regular_grid(B)             # B interior grid (q_c × q_n^B)
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row index
    bn = nominal_tile_size(B, 2)                  # output col size
    ord = tile_order(B)

    Swork = allocate(A.backend, T, rA, rB, qkA, qnB)
    Twork = allocate(A.backend, T, rA, bn, qkA, qnB)
    # Stage 1:  S_{j,k} = V^b_k' W_kj
    gemm_batched!('T', 'N', one(T),
        [view(A.bottom_V, :, :, k) for j in 1:qnB for k in 1:qkA],
        [_int_Uview(B, ord, qkB, qnB, k, j) for j in 1:qnB for k in 1:qkA],
        zero(T), [view(Swork, :, :, k, j) for j in 1:qnB for k in 1:qkA])
    # Stage 2:  T_{j,k} = S_{j,k} Z_kj'
    gemm_batched!('N', 'T', one(T),
        [view(Swork, :, :, k, j) for j in 1:qnB for k in 1:qkA],
        [_int_Vview(B, ord, qkB, qnB, k, j) for j in 1:qnB for k in 1:qkA],
        zero(T), [view(Twork, :, :, k, j) for j in 1:qnB for k in 1:qkA])
    # Stage 3:  C_bottom[j] += α Σ_k U^b_k T_{j,k}
    @inbounds for k in 1:qkA
        bb = k == 1 ? beta : one(T)
        gemm_batched!('N', 'N', alpha,
            [view(A.bottom_U, :, :, k) for j in 1:qnB],
            [view(Twork, :, :, k, j) for j in 1:qnB],
            bb, [_output_tile_view(C, A, B, mt, j) for j in 1:qnB])
    end
    return C
end

# γ_A v_Bᵀ:  C_bottom[j] += γ_A B_{bnd,j},  j = 1:q_n^B.  Single contraction (the corner).
function tlr_gemm_corner_by_bpanel(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T)) where {T,BackendT}
    qnB = _bottom_panel_tiles(B)                 # B bottom-panel tiles = q_n^B
    qnB == 0 && return C
    size(A.corner_U, 3) == 0 && return C
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row index
    bn = nominal_tile_size(B, 2)                  # output col size

    Swork = allocate(A.backend, T, rA, rB, qnB)  # S_j = Vc' W^b_j
    gemm_batched!('T', 'N', one(T),
        [view(A.corner_V, :, :, 1) for _ in 1:qnB],
        [view(B.bottom_U, :, :, j) for j in 1:qnB],
        zero(T), [view(Swork, :, :, j) for j in 1:qnB])
    Twork = allocate(A.backend, T, rA, bn, qnB)  # T_j = S_j Z^b_j'
    gemm_batched!('N', 'T', one(T),
        [view(Swork, :, :, j) for j in 1:qnB],
        [view(B.bottom_V, :, :, j) for j in 1:qnB],
        zero(T), [view(Twork, :, :, j) for j in 1:qnB])
    gemm_batched!('N', 'N', alpha,             # C_bottom[j] += α Uc T_j
        [view(A.corner_U, :, :, 1) for _ in 1:qnB],
        [view(Twork, :, :, j) for j in 1:qnB],
        beta, [_output_tile_view(C, A, B, mt, j) for j in 1:qnB])
    return C
end
