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
    rA = maxrank(A)
    rB = maxrank(B)
    AU = outer_factors(A, _INTERIOR)
    AV = inner_factors(A, _INTERIOR)
    BU = outer_factors(B, _RIGHT)
    BV = inner_factors(B, _RIGHT)

    sn = size(BV, 1)                            # panel width (= n_B % b)

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

    # off-diagonal reduction (β = 1), budget-split over output rows.
    (rA == 0 || rB == 0) && return C
    nk_off_per_row = qmA - 1
    nk_off_per_row == 0 && return C
    order = tile_order(A)
    bytes_per_i = max(rA * nk_off_per_row * (rB + sn) * sizeof(T), 1)
    maxI = clamp(div(budget, bytes_per_i), 1, qmA)
    Swork = allocate(get_backend(A), T, rA, rB, nk_off_per_row, maxI)
    Twork = allocate(get_backend(A), T, rA, sn, nk_off_per_row, maxI)

    @inbounds for irange in Iterators.partition(1:qmA, maxI)
        i0 = first(irange)
        # Stage 1:  S_{i,kpos} = V_ijᵀ W_j,  batched over row and off-diagonal k position.
        precision_gemm_batched!('T', 'N', one(T),
            [view(AV, :, :, _offdiag_index(order, qmA, qmA, i, local_to_col(i, kpos))) for i in irange for kpos in 1:nk_off_per_row],
            [view(BU, :, :, local_to_col(i, kpos)) for i in irange for kpos in 1:nk_off_per_row],
            zero(T),
            [view(Swork, :, :, kpos, i - i0 + 1) for i in irange for kpos in 1:nk_off_per_row], compute)
        # Stage 2:  T_{i,kpos} = S_{i,kpos} Z_jᵀ,  same batch.
        precision_gemm_batched!('N', 'T', one(T),
            [view(Swork, :, :, kpos, i - i0 + 1) for i in irange for kpos in 1:nk_off_per_row],
            [view(BV, :, :, local_to_col(i, kpos)) for i in irange for kpos in 1:nk_off_per_row],
            zero(T),
            [view(Twork, :, :, kpos, i - i0 + 1) for i in irange for kpos in 1:nk_off_per_row], compute)
        # Stage 3:  C_right[i] += α·Σ_j U_ij T_{i,kpos}.  Loop the reduction; each kpos
        # writes distinct rows i, so batch over i and accumulate with β = 1.
        for kpos in 1:nk_off_per_row
            precision_gemm_batched!('N', 'N', alpha,
                [view(AU, :, :, _offdiag_index(order, qmA, qmA, i, local_to_col(i, kpos))) for i in irange],
                [view(Twork, :, :, kpos, i - i0 + 1) for i in irange],
                one(T),
                [_output_tile_view(C, A, B, i, nt) for i in irange], compute)
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
function tlr_gemm_rpanel_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRDenseDiagMatrix}, alpha; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    qmA = region_tile_count(A, _RIGHT)
    qmA == 0 && return C                      # no right panel (n_A % b == 0)
    rA = maxrank(A)
    (rA == 0 || size(physical(B).D_corner, 3) == 0) && return C
    _, nt = tilegrid_size(B)                   # output boundary tile-column index
    corner = _diag_tile_ref(B, ndiag_tiles(B))
    sn = tile_size(B, tilegrid_size(B)...)[2]  # logical corner column extent
    AU = outer_factors(A, _RIGHT)
    AV = inner_factors(A, _RIGHT)

    # Stage 1:  M_i = V_iᵀ γ_B   (rA×s_n), strided over i with γ_B (batch-1) broadcast.
    Twork = allocate(get_backend(A), T, rA, sn, qmA)
    precision_gemm_batched!('T', _dense_op(corner), one(T), AV,
                            physical(B).D_corner, zero(T), Twork, compute)

    # Stage 2:  C_{i,Q+1} += α · U_i M_i   (b×s_n), batched over i.
    Uvec = [view(AU, :, :, i) for i in 1:qmA]
    Tvec = [view(Twork, :, :, i) for i in 1:qmA]
    Cvec = [_output_tile_view(C, A, B, i, nt) for i in 1:qmA]
    precision_gemm_batched!('N', 'N', alpha, Uvec, Tvec, beta, Cvec, compute)
    return C
end

# ── Fully low-rank variants (TLRMatrix) ──────────────────────────────────────
#
# Every tile is low-rank, so there is no dense diagonal to split out: `A_int u_B`
# reduces over ALL contraction tiles `k`, and `u_A γ_B` uses a low-rank corner `γ_B`.

# A_int u_B:  C_right[i] = Σ_{k=1:q_c} A_ik B_{k,bnd},  i = 1:q_m^A.  Batched Stages 1/2
# over (i, k); Stage 3 loops the reduction `k` (batched over i), first `k` folds β.
function tlr_gemm_int_by_rpanel(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T)) where {T}
    qkB = region_tile_count(B, _RIGHT)           # one per interior row k
    qkB == 0 && return C
    qmA, qkA = regular_tilegrid_size(A)

    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    _, ntB = tilegrid_size(B)                    # B boundary tile-col index
    BU = outer_factors(B, _RIGHT)
    BV = inner_factors(B, _RIGHT)
    sn = size(BV, 1)                             # tail width (n_B % b)
    ord = tile_order(A)
    Ainner = interior_operand(FullGrid(), inner_factors(A, _INTERIOR), ord, qmA, qkA)
    Aouter = interior_operand(FullGrid(), outer_factors(A, _INTERIOR), ord, qmA, qkA)

    Swork = allocate(get_backend(A), T, rA, rB, qkB, qmA)
    Twork = allocate(get_backend(A), T, rA, sn, qkB, qmA)
    # Stage 1:  S_{i,k} = V_ik' W_k
    precision_gemm_batched!('T', 'N', one(T),
        [tilefactor(Ainner, i, k) for i in 1:qmA for k in 1:qkB],
        [view(BU, :, :, k) for i in 1:qmA for k in 1:qkB],
        zero(T), [view(Swork, :, :, k, i) for i in 1:qmA for k in 1:qkB], compute)
    # Stage 2:  T_{i,k} = S_{i,k} Z_k'
    precision_gemm_batched!('N', 'T', one(T),
        [view(Swork, :, :, k, i) for i in 1:qmA for k in 1:qkB],
        [view(BV, :, :, k) for i in 1:qmA for k in 1:qkB],
        zero(T), [view(Twork, :, :, k, i) for i in 1:qmA for k in 1:qkB], compute)
    # Stage 3:  C_right[i] += α Σ_k U_ik T_{i,k}   (loop the reduction, batch i)
    @inbounds for k in 1:qkB
        bb = k == 1 ? beta : one(T)
        precision_gemm_batched!('N', 'N', alpha,
            [tilefactor(Aouter, i, k) for i in 1:qmA],
            [view(Twork, :, :, k, i) for i in 1:qmA],
            bb, [_output_tile_view(C, A, B, i, ntB) for i in 1:qmA], compute)
    end
    return C
end

# u_A γ_B:  C_right[i] += A_{i,bnd} γ_B,  i = 1:q_m^A.  Single contraction (the corner),
# batched over i.  A_{i,bnd} = U^r_i V^r_i',  γ_B = Wc Zc'.
function tlr_gemm_rpanel_by_corner(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha;
    beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    qmA = region_tile_count(A, _RIGHT)           # A right-panel tiles = q_m^A
    qmA == 0 && return C

    BCouter = outer_factors(B, _CORNER)
    BCinner = inner_factors(B, _CORNER)
    size(BCouter, 3) == 0 && return C
    rA = maxrank(A); rB = maxrank(B)
    (rA == 0 || rB == 0) && return C
    _, ntB = tilegrid_size(B)
    AU = outer_factors(A, _RIGHT)
    AV = inner_factors(A, _RIGHT)
    sn = size(BCinner, 1)                        # corner col extent (n_B % b)

    Swork = allocate(get_backend(A), T, rA, rB, qmA)  # S_i = V^r_i' Wc
    precision_gemm_batched!('T', 'N', one(T),
        [view(AV, :, :, i) for i in 1:qmA],
        [view(BCouter, :, :, 1) for _ in 1:qmA],
        zero(T), [view(Swork, :, :, i) for i in 1:qmA], compute)
    Twork = allocate(get_backend(A), T, rA, sn, qmA)  # T_i = S_i Zc'
    precision_gemm_batched!('N', 'T', one(T),
        [view(Swork, :, :, i) for i in 1:qmA],
        [view(BCinner, :, :, 1) for _ in 1:qmA],
        zero(T), [view(Twork, :, :, i) for i in 1:qmA], compute)
    precision_gemm_batched!('N', 'N', alpha,             # C_right[i] += α U^r_i T_i
        [view(AU, :, :, i) for i in 1:qmA],
        [view(Twork, :, :, i) for i in 1:qmA],
        beta, [_output_tile_view(C, A, B, i, ntB) for i in 1:qmA], compute)
    return C
end
