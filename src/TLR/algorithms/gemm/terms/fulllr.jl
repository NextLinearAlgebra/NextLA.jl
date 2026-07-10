# ─── Fully low-rank boundary/corner terms ─────────────────────────────────────
#
# For `TLRMatrix` every tile — interior, boundary panel, corner — is low-rank, so
# the dense-diagonal terms' special cases vanish: the interior×panel reductions run
# over ALL contraction tiles `k` (no diagonal to split out) and the corner is a
# low-rank tile (a 3-stage product) rather than a dense `mul!`.
#
# The caller (`gemm!(::TLRMatrix, …)`) pre-scales `C` by β once, so every term here
# accumulates with β = 1 — no per-region β folding.
#
# These currently assume the boundary is square (A, B same size); the interior term
# already handles rectangular grids. `_dense_tile_view(C, A, i, j)` is valid for the
# output tile because A, B, C share geometry in that regime.

# Interior factor views for full-LR tile (i, j) (stored in `tile_linear_index` order).
@inline _int_Uview(A, ord, qm, qn, i, j) = view(A.int_U, :, :, tile_linear_index(ord, qm, qn, i, j))
@inline _int_Vview(A, ord, qm, qn, i, j) = view(A.int_V, :, :, tile_linear_index(ord, qm, qn, i, j))

# ── Right region:  C_right = A_int u_B + u_A γ_B ──────────────────────────────

# A_int u_B:  C_right[i] = Σ_{k=1:q_c} A_ik B_{k,bnd},  i = 1:q_m^A.  Three stages,
# reduction over ALL interior k (batched Stages 1/2, looped Stage 3).
function tlr_gemm_int_by_rpanel(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T), budget::Int) where {T,BackendT}
    qc = _right_panel_tiles(B)                   # one per interior row k
    qc == 0 && return C
    qmA, qcA = _full_regular_grid(A)

    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    _, ntB = tilegrid_size(B)                    # B boundary tile-col index
    s = size(B.right_V, 1)                       # tail width (n_B % b)
    ord = tile_order(A)

    S = allocate(A.backend, T, rA, rB, qc, qmA)
    Tw = allocate(A.backend, T, rA, s, qc, qmA)
    # Stage 1:  S_{i,k} = V_ik' W_k
    gemm_batched!('T', 'N', one(T),
        [_int_Vview(A, ord, qmA, qcA, i, k) for i in 1:qmA for k in 1:qc],
        [view(B.right_U, :, :, k) for i in 1:qmA for k in 1:qc],
        zero(T), [view(S, :, :, k, i) for i in 1:qmA for k in 1:qc])
    # Stage 2:  T_{i,k} = S_{i,k} Z_k'
    gemm_batched!('N', 'T', one(T),
        [view(S, :, :, k, i) for i in 1:qmA for k in 1:qc],
        [view(B.right_V, :, :, k) for i in 1:qmA for k in 1:qc],
        zero(T), [view(Tw, :, :, k, i) for i in 1:qmA for k in 1:qc])
    # Stage 3:  C_right[i] += α Σ_k U_ik T_{i,k}   (loop the reduction, batch i)
    @inbounds for k in 1:qc
        bb = k == 1 ? beta : one(T)
        gemm_batched!('N', 'N', alpha,
            [_int_Uview(A, ord, qmA, qcA, i, k) for i in 1:qmA],
            [view(Tw, :, :, k, i) for i in 1:qmA],
            bb, [_output_tile_view(C, A, B, i, ntB) for i in 1:qmA])
    end
    return C
end

# u_A γ_B:  C_right[i] += A_{i,bnd} γ_B,  i = 1:q_m^A.  Single contraction (the corner),
# batched over i.  A_{i,bnd} = U^r_i V^r_i',  γ_B = Wc Zc'.
function tlr_gemm_rpanel_by_corner(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T)) where {T,BackendT}
    Q = _right_panel_tiles(A)                    # A right-panel tiles = q_m^A
    Q == 0 && return C

    size(B.corner_U, 3) == 0 && return C
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    _, ntB = tilegrid_size(B)
    s_n = size(B.corner_V, 1)                    # corner col extent (n_B % b)

    P = allocate(A.backend, T, rA, rB, Q)        # P_i = V^r_i' Wc
    gemm_batched!('T', 'N', one(T),
        [view(A.right_V, :, :, i) for i in 1:Q],
        [view(B.corner_U, :, :, 1) for _ in 1:Q],
        zero(T), [view(P, :, :, i) for i in 1:Q])
    Tw = allocate(A.backend, T, rA, s_n, Q)      # T_i = P_i Zc'
    gemm_batched!('N', 'T', one(T),
        [view(P, :, :, i) for i in 1:Q],
        [view(B.corner_V, :, :, 1) for _ in 1:Q],
        zero(T), [view(Tw, :, :, i) for i in 1:Q])
    gemm_batched!('N', 'N', alpha,             # C_right[i] += α U^r_i T_i
        [view(A.right_U, :, :, i) for i in 1:Q],
        [view(Tw, :, :, i) for i in 1:Q],
        beta, [_output_tile_view(C, A, B, i, ntB) for i in 1:Q])
    return C
end

# ── Bottom region:  C_bottom = v_Aᵀ B_int + γ_A v_Bᵀ ──────────────────────────

# v_Aᵀ B_int:  C_bottom[j] = Σ_{k=1:q_c} A_{bnd,k} B_kj,  j = 1:q_n^B.
function tlr_gemm_bpanel_by_int(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T), budget::Int) where {T,BackendT}
    qc = _bottom_panel_tiles(A)                  # one per interior col k
    qc == 0 && return C
    qcB, qnB = _full_regular_grid(B)             # B interior grid (q_c × q_n^B)
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row index
    b = nominal_tile_size(B, 2)                   # output col size
    ord = tile_order(B)

    S = allocate(A.backend, T, rA, rB, qc, qnB)
    Tw = allocate(A.backend, T, rA, b, qc, qnB)
    # Stage 1:  S_{j,k} = V^b_k' W_kj
    gemm_batched!('T', 'N', one(T),
        [view(A.bottom_V, :, :, k) for j in 1:qnB for k in 1:qc],
        [_int_Uview(B, ord, qcB, qnB, k, j) for j in 1:qnB for k in 1:qc],
        zero(T), [view(S, :, :, k, j) for j in 1:qnB for k in 1:qc])
    # Stage 2:  T_{j,k} = S_{j,k} Z_kj'
    gemm_batched!('N', 'T', one(T),
        [view(S, :, :, k, j) for j in 1:qnB for k in 1:qc],
        [_int_Vview(B, ord, qcB, qnB, k, j) for j in 1:qnB for k in 1:qc],
        zero(T), [view(Tw, :, :, k, j) for j in 1:qnB for k in 1:qc])
    # Stage 3:  C_bottom[j] += α Σ_k U^b_k T_{j,k}
    @inbounds for k in 1:qc
        bb = k == 1 ? beta : one(T)
        gemm_batched!('N', 'N', alpha,
            [view(A.bottom_U, :, :, k) for j in 1:qnB],
            [view(Tw, :, :, k, j) for j in 1:qnB],
            bb, [_output_tile_view(C, A, B, mt, j) for j in 1:qnB])
    end
    return C
end

# γ_A v_Bᵀ:  C_bottom[j] += γ_A B_{bnd,j},  j = 1:q_n^B.  Single contraction (the corner).
function tlr_gemm_corner_by_bpanel(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T)) where {T,BackendT}
    Q = _bottom_panel_tiles(B)                   # B bottom-panel tiles = q_n^B
    Q == 0 && return C
    size(A.corner_U, 3) == 0 && return C
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row index
    b = nominal_tile_size(B, 2)                   # output col size

    P = allocate(A.backend, T, rA, rB, Q)        # P_j = Vc' W^b_j
    gemm_batched!('T', 'N', one(T),
        [view(A.corner_V, :, :, 1) for _ in 1:Q],
        [view(B.bottom_U, :, :, j) for j in 1:Q],
        zero(T), [view(P, :, :, j) for j in 1:Q])
    Tw = allocate(A.backend, T, rA, b, Q)        # T_j = P_j Z^b_j'
    gemm_batched!('N', 'T', one(T),
        [view(P, :, :, j) for j in 1:Q],
        [view(B.bottom_V, :, :, j) for j in 1:Q],
        zero(T), [view(Tw, :, :, j) for j in 1:Q])
    gemm_batched!('N', 'N', alpha,             # C_bottom[j] += α Uc T_j
        [view(A.corner_U, :, :, 1) for _ in 1:Q],
        [view(Tw, :, :, j) for j in 1:Q],
        beta, [_output_tile_view(C, A, B, mt, j) for j in 1:Q])
    return C
end

# ── Corner region:  C_corner = γ_A γ_B + v_Aᵀ u_B ────────────────────────────

# γ_A γ_B: low-rank corner × low-rank corner, a single 3-stage product.
function tlr_gemm_corner_by_corner(C, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T)) where {T,BackendT}
    (size(A.corner_U, 3) != 0 && size(B.corner_U, 3) != 0) || return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row
    _, ntB = tilegrid_size(B)                     # B boundary tile-col
    Cc = _output_tile_view(C, A, B, mt, ntB)
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C             # γ contributes 0; C already β-scaled
    s_n = size(B.corner_V, 1)

    P = allocate(A.backend, T, rA, rB, 1)        # P = Vc^A' Wc^B
    gemm_batched!('T', 'N', one(T), A.corner_V, B.corner_U, zero(T), P)
    Tw = allocate(A.backend, T, rA, s_n, 1)      # T = P Zc^B'
    gemm_batched!('N', 'T', one(T), P, B.corner_V, zero(T), Tw)
    mul!(Cc, view(A.corner_U, :, :, 1), view(Tw, :, :, 1), alpha, beta)   # C += α Uc^A T
    return C
end
