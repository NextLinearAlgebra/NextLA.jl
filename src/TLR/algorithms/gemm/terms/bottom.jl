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
function tlr_gemm_bpanel_by_int(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha; beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    qkA = region_tile_count(A, _BOTTOM)        # A bottom-panel tiles (= interior columns)
    qkA == 0 && return C                       # no bottom panel (m_A % b == 0)

    mt, _ = tilegrid_size(A)                    # boundary tile-row index (Q+1)
    bn = nominal_tile_size(B, 2)
    rA = maxrank(A)
    rB = maxrank(B)
    AU = outer_factors(A, _BOTTOM)
    AV = inner_factors(A, _BOTTOM)
    BU = outer_factors(B, _INTERIOR)
    BV = inner_factors(B, _INTERIOR)
    sm = size(AU, 1)                            # panel height (= m_A % b)

    # Diagonal i=j:  C_bottom[j] = β·C_bottom + α·U_j (V_jᵀ B_jj)   (first writer, folds β).
    VDBdiag = allocate(get_backend(A), T, rA, bn, qkA)
    precision_gemm_batched!('T', _opchar(B), one(T),
        [view(AV, :, :, j) for j in 1:qkA],
        [_dense_data(_diag_tile_ref(B, j)) for j in 1:qkA],
        zero(T), [view(VDBdiag, :, :, j) for j in 1:qkA], compute)
    precision_gemm_batched!('N', 'N', alpha,
        [view(AU, :, :, j) for j in 1:qkA],
        [view(VDBdiag, :, :, j) for j in 1:qkA],
        beta, [_output_tile_view(C, A, B, mt, j) for j in 1:qkA], compute)

    # Off-diagonal reduction (β = 1), budget-split over output columns.
    (rA == 0 || rB == 0) && return C
    nk_off_per_col = qkA - 1
    nk_off_per_col == 0 && return C
    order = tile_order(B)
    bytes_per_j = max(rA * nk_off_per_col * (rB + bn) * sizeof(T), 1)
    maxJ = clamp(div(budget, bytes_per_j), 1, qkA)
    Swork = allocate(get_backend(A), T, rA, rB, nk_off_per_col, maxJ)
    Twork = allocate(get_backend(A), T, rA, bn, nk_off_per_col, maxJ)

    @inbounds for jrange in Iterators.partition(1:qkA, maxJ)
        j0 = first(jrange)
        # Stage 1:  S_{j,kpos} = V_iᵀ W_ij,  batched over column and off-diagonal k position.
        precision_gemm_batched!('T', 'N', one(T),
            [view(AV, :, :, local_to_col(j, kpos)) for j in jrange for kpos in 1:nk_off_per_col],
            [view(BU, :, :, _offdiag_index(order, qkA, qkA, local_to_col(j, kpos), j)) for j in jrange for kpos in 1:nk_off_per_col],
            zero(T),
            [view(Swork, :, :, kpos, (j - j0 + 1)) for j in jrange for kpos in 1:nk_off_per_col], compute)
        # Stage 2:  T_{j,kpos} = S_{j,kpos} Z_ijᵀ,  same batch.
        precision_gemm_batched!('N', 'T', one(T),
            [view(Swork, :, :, kpos, (j - j0 + 1)) for j in jrange for kpos in 1:nk_off_per_col],
            [view(BV, :, :, _offdiag_index(order, qkA, qkA, local_to_col(j, kpos), j)) for j in jrange for kpos in 1:nk_off_per_col],
            zero(T),
            [view(Twork, :, :, kpos, (j - j0 + 1)) for j in jrange for kpos in 1:nk_off_per_col], compute)
        # Stage 3:  C_bottom[j] += α·Σ_i U_i T_{j,kpos}.  Loop the reduction; each kpos
        # writes distinct columns j, so batch over j and accumulate with β = 1.
        for kpos in 1:nk_off_per_col
            precision_gemm_batched!('N', 'N', alpha,
                [view(AU, :, :, local_to_col(j, kpos)) for j in jrange],
                [view(Twork, :, :, kpos, (j - j0 + 1)) for j in jrange],
                one(T),
                [_output_tile_view(C, A, B, mt, j) for j in jrange], compute)
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
function tlr_gemm_corner_by_bpanel(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    qnB = region_tile_count(B, _BOTTOM)
    qnB == 0 && return C                      # no bottom panel (m_B % b == 0)
    rB = maxrank(B)
    (rB == 0 || size(physical(A).D_corner, 3) == 0) && return C
    mt, _ = tilegrid_size(A)                   # boundary tile-row index (Q+1)
    corner = _diag_tile_ref(A, ndiag_tiles(A))
    sm = tile_size(A, tilegrid_size(A)...)[1]  # logical corner row extent
    BU = outer_factors(B, _BOTTOM)
    BV = inner_factors(B, _BOTTOM)

    # Stage 1:  N_j = γ_A W_j   (s_m×rB), strided over j with γ_A (batch-1) broadcast.
    Swork = allocate(get_backend(A), T, sm, rB, qnB)
    precision_gemm_batched!(_dense_op(corner), 'N', one(T),
                            physical(A).D_corner, BU, zero(T), Swork, compute)

    # Stage 2:  C_{Q+1,j} += α · N_j Z_jᵀ   (s_m×b), batched over j.
    Svec = [view(Swork, :, :, j) for j in 1:qnB]
    Zvec = [view(BV, :, :, j) for j in 1:qnB]
    Cvec = [_output_tile_view(C, A, B, mt, j) for j in 1:qnB]
    precision_gemm_batched!('N', 'T', alpha, Svec, Zvec, beta, Cvec, compute)
    return C
end

# ── Fully low-rank variants (TLRMatrix) ──────────────────────────────────────
#
# `v_Aᵀ B_int` reduces over ALL contraction tiles `k` (no dense diagonal), and
# `γ_A v_Bᵀ` uses a low-rank corner `γ_A`.

# v_Aᵀ B_int:  C_bottom[j] = Σ_{k=1:q_c} A_{bnd,k} B_kj,  j = 1:q_n^B.  Batched Stages
# 1/2 over (j, k); Stage 3 loops the reduction `k` (batched over j), first `k` folds β.
function tlr_gemm_bpanel_by_int(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    qkA = region_tile_count(A, _BOTTOM)          # one per interior col k
    qkA == 0 && return C
    qkB, qnB = regular_tilegrid_size(B)          # B interior grid (q_c × q_n^B)
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row index
    bn = nominal_tile_size(B, 2)                  # output col size
    ord = tile_order(B)
    AU = outer_factors(A, _BOTTOM)
    AV = inner_factors(A, _BOTTOM)
    Bouter = interior_operand(FullGrid(), outer_factors(B, _INTERIOR), ord, qkB, qnB)
    Binner = interior_operand(FullGrid(), inner_factors(B, _INTERIOR), ord, qkB, qnB)

    Swork = allocate(get_backend(A), T, rA, rB, qkA, qnB)
    Twork = allocate(get_backend(A), T, rA, bn, qkA, qnB)
    # Stage 1:  S_{j,k} = V^b_k' W_kj
    precision_gemm_batched!('T', 'N', one(T),
        [view(AV, :, :, k) for j in 1:qnB for k in 1:qkA],
        [tilefactor(Bouter, k, j) for j in 1:qnB for k in 1:qkA],
        zero(T), [view(Swork, :, :, k, j) for j in 1:qnB for k in 1:qkA], compute)
    # Stage 2:  T_{j,k} = S_{j,k} Z_kj'
    precision_gemm_batched!('N', 'T', one(T),
        [view(Swork, :, :, k, j) for j in 1:qnB for k in 1:qkA],
        [tilefactor(Binner, k, j) for j in 1:qnB for k in 1:qkA],
        zero(T), [view(Twork, :, :, k, j) for j in 1:qnB for k in 1:qkA], compute)
    # Stage 3:  C_bottom[j] += α Σ_k U^b_k T_{j,k}
    @inbounds for k in 1:qkA
        bb = k == 1 ? beta : one(T)
        precision_gemm_batched!('N', 'N', alpha,
            [view(AU, :, :, k) for j in 1:qnB],
            [view(Twork, :, :, k, j) for j in 1:qnB],
            bb, [_output_tile_view(C, A, B, mt, j) for j in 1:qnB], compute)
    end
    return C
end

# γ_A v_Bᵀ:  C_bottom[j] += γ_A B_{bnd,j},  j = 1:q_n^B.  Single contraction (the corner).
function tlr_gemm_corner_by_bpanel(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    qnB = region_tile_count(B, _BOTTOM)          # B bottom-panel tiles = q_n^B
    qnB == 0 && return C
    ACouter = outer_factors(A, _CORNER)
    ACinner = inner_factors(A, _CORNER)
    size(ACouter, 3) == 0 && return C
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    mt, _ = tilegrid_size(A)                      # A boundary tile-row index
    bn = nominal_tile_size(B, 2)                  # output col size
    BU = outer_factors(B, _BOTTOM)
    BV = inner_factors(B, _BOTTOM)

    Swork = allocate(get_backend(A), T, rA, rB, qnB)  # S_j = Vc' W^b_j
    precision_gemm_batched!('T', 'N', one(T),
        [view(ACinner, :, :, 1) for _ in 1:qnB],
        [view(BU, :, :, j) for j in 1:qnB],
        zero(T), [view(Swork, :, :, j) for j in 1:qnB], compute)
    Twork = allocate(get_backend(A), T, rA, bn, qnB)  # T_j = S_j Z^b_j'
    precision_gemm_batched!('N', 'T', one(T),
        [view(Swork, :, :, j) for j in 1:qnB],
        [view(BV, :, :, j) for j in 1:qnB],
        zero(T), [view(Twork, :, :, j) for j in 1:qnB], compute)
    precision_gemm_batched!('N', 'N', alpha,             # C_bottom[j] += α Uc T_j
        [view(ACouter, :, :, 1) for _ in 1:qnB],
        [view(Twork, :, :, j) for j in 1:qnB],
        beta, [_output_tile_view(C, A, B, mt, j) for j in 1:qnB], compute)
    return C
end
