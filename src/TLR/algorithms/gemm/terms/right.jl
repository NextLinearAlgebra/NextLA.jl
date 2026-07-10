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
function tlr_gemm_int_by_rpanel(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T; beta::T=one(T), budget::Int) where {T, BackendT}
    Q = size(B.right_U, 3)
    Q == 0 && return C                         # no right panel (n_B % b == 0)
    Q == _full_regular_grid(A)[1] || return C      # non-square

    _, nt = tilegrid_size(A)                    # boundary tile-column index (Q+1)
    b = nominal_tile_size(A, 1)
    rA = maxrank(A)
    rB = maxrank(B)

    s = size(B.right_V, 1)                      # panel width (= n_B % b)

    # diagonal j=i:  C_right[i] = β·C_right + α·(A_ii W_i) Z_iᵀ   (folds β).
    Ndiag = allocate(A.backend, T, b, rB, Q)
    gemm_batched!('N', 'N', one(T),
        [_diag_tile_view(A, i) for i in 1:Q],
        [view(B.right_U, :, :, i) for i in 1:Q],
        zero(T), [view(Ndiag, :, :, i) for i in 1:Q])
    gemm_batched!('N', 'T', alpha,
        [view(Ndiag, :, :, i) for i in 1:Q],
        [view(B.right_V, :, :, i) for i in 1:Q],
        beta, [_dense_tile_view(C, A, i, nt) for i in 1:Q])

    # off-diagonal reduction (β = 1), budget-split over output rows.
    (rA == 0 || rB == 0) && return C
    per = Q - 1                                 # off-diagonal tiles per row
    per == 0 && return C
    order = tile_order(A)
    percol = max(rA * per * (rB + s) * sizeof(T), 1)
    maxI = clamp(div(budget, percol), 1, Q)
    S = allocate(A.backend, T, rA, rB, per, maxI)
    Tw = allocate(A.backend, T, rA, s, per, maxI)

    @inbounds for irange in Iterators.partition(1:Q, maxI)
        i0 = first(irange)
        # Stage 1:  S_{i,kk} = V_ijᵀ W_j,  batched over (row i, off-diag position kk).
        gemm_batched!('T', 'N', one(T),
            [view(A.int_V, :, :, _offdiag_index(order, Q, Q, i, local_to_col(i, kk))) for i in irange for kk in 1:per],
            [view(B.right_U, :, :, local_to_col(i, kk)) for i in irange for kk in 1:per],
            zero(T),
            [view(S, :, :, kk, i - i0 + 1) for i in irange for kk in 1:per])
        # Stage 2:  T_{i,kk} = S_{i,kk} Z_jᵀ,  same batch.
        gemm_batched!('N', 'T', one(T),
            [view(S, :, :, kk, i - i0 + 1) for i in irange for kk in 1:per],
            [view(B.right_V, :, :, local_to_col(i, kk)) for i in irange for kk in 1:per],
            zero(T),
            [view(Tw, :, :, kk, i - i0 + 1) for i in irange for kk in 1:per])
        # Stage 3:  C_right[i] += α·Σ_j U_ij T_{i,kk}.  Loop the reduction (kk); each kk
        # writes distinct rows i, so batch over i and accumulate with β = 1.
        for kk in 1:per
            gemm_batched!('N', 'N', alpha,
                [view(A.int_U, :, :, _offdiag_index(order, Q, Q, i, local_to_col(i, kk))) for i in irange],
                [view(Tw, :, :, kk, i - i0 + 1) for i in irange],
                one(T),
                [_dense_tile_view(C, A, i, nt) for i in irange])
        end
    end
    return C
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
function tlr_gemm_rpanel_by_corner(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T, BackendT}
    Q = size(A.right_U, 3)
    Q == 0 && return C                        # no right panel (n_A % b == 0)
    rA = maxrank(A)
    (rA == 0 || size(B.D_corner, 3) == 0) && return C
    _, nt = tilegrid_size(A)                   # boundary tile-column index (Q+1)
    s_n = size(B.D_corner, 2)                  # corner column extent (= n_B % b)

    # Stage 1:  M_i = V_iᵀ γ_B   (rA×s_n), strided over i with γ_B (batch-1) broadcast.
    M = allocate(A.backend, T, rA, s_n, Q)
    gemm_batched!('T', 'N', one(T), A.right_V, B.D_corner, zero(T), M)

    # Stage 2:  C_{i,Q+1} += α · U_i M_i   (b×s_n), batched over i.
    Uvec = [view(A.right_U, :, :, i) for i in 1:Q]
    Mvec = [view(M, :, :, i) for i in 1:Q]
    Cvec = [_dense_tile_view(C, A, i, nt) for i in 1:Q]
    gemm_batched!('N', 'N', alpha, Uvec, Mvec, beta, Cvec)
    return C
end
