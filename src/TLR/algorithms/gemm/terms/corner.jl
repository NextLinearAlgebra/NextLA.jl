# ─── Boundary corner term:  C_corner += α · vᵀx  (only when m%b ≠ 0) ──────────────
#
# Splitting the operands around the partial last tile,
#   A = [A₁₁  u ;  vᵀ  α],   B = [B₁₁  x ;  yᵀ  β],
# the corner block of C = AB is  vᵀx + αβ.  `αβ` is produced by `_diag_diag!`
# (D_corner · D_corner); this term adds
#   vᵀx = Σ_p v_p x_p         (p = interior index 1..Q)
# where v_p is A's bottom-row tile (mt,p) and x_p is B's right-column tile (p,nt).
# Each low-rank product factors as  v_p x_p = U_p (V_pᵀ W_p) Z_pᵀ  with
#   U_p = A.bottom_U[p] (s_m×r),  V_p = A.bottom_V[p] (b×r),
#   W_p = B.right_U[q]  (b×r),    Z_p = B.right_V[q]  (s_n×r),
# lowered as the usual three stages: Stage 1 `VᵀW` and Stage 2 `S·Zᵀ` batched over
# `p`, Stage 3 `U·T` K-stacked over `p` into a single accumulating GEMM.
#
# (Segments the `p`-batch when S/T exceed the budget — TODO; the current form
# does the whole reduction in one pass.)

# Pair A's bottom tiles (mt,p) with B's right tiles (p,nt) by the shared index `p`.
# Returns B-right storage indices in A-bottom storage order (so Stage 3 can reuse
# `A.bottom_U` as a contiguous K-stack); requires the full square pairing.
function _corner_pairing(A::TLRMatrix, B::TLRMatrix)
    brow = Dict{Int,Int}()                       # B right storage index keyed by row p
    for (k, ob) in enumerate(B.obs_right)
        p, _ = _offdiag_coords(B, ob)
        brow[p] = k
    end
    bpair = Vector{Int}(undef, length(A.obs_bottom))
    for (k, ob) in enumerate(A.obs_bottom)
        _, p = _offdiag_coords(A, ob)
        haskey(brow, p) || return nothing         # incomplete pairing (non-square)
        bpair[k] = brow[p]
    end
    return bpair
end

"""
    _corner_term!(C, A, B, alpha; beta=1) -> C

Accumulate `α · vᵀx` into the dense corner block `C[mt,nt]`, i.e.
`C_corner := beta·C_corner + α·Σ_p v_p x_p`. No-op when `m % b == 0`.
"""
function _corner_term!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha; beta=one(T)) where {T}
    size(A.D_corner, 3) != 0 || return C          # no boundary corner (m%b == 0)
    r = A.maxrank
    (r == 0 || B.maxrank == 0) && return C
    bpair = _corner_pairing(A, B)
    (bpair === nothing || isempty(bpair)) && return C

    mt, nt = tilegrid_size(A)
    np = length(bpair)                            # == Q interior indices
    s_m = size(A.bottom_U, 1)
    s_n = size(B.right_V, 1)

    S = allocate(A.backend, T, r, r, np)
    Tw = allocate(A.backend, T, r, np, s_n)

    # Stage 1:  S_p = V_pᵀ W_p   (r×r, contract b), batched over p.
    V1 = [view(A.bottom_V, :, :, k) for k in 1:np]
    W1 = [view(B.right_U, :, :, bpair[k]) for k in 1:np]
    S1 = [view(S, :, :, k) for k in 1:np]
    gemm_batched!('T', 'N', one(T), V1, W1, zero(T), S1)

    # Stage 2:  T_p = S_p Z_pᵀ   (r×s_n), batched over p.
    Z2 = [view(B.right_V, :, :, bpair[k]) for k in 1:np]
    T2 = [view(Tw, :, k, :) for k in 1:np]
    gemm_batched!('N', 'T', one(T), S1, Z2, zero(T), T2)

    # Stage 3:  C_corner += α · Σ_p U_p T_p, K-stacked over p (one GEMM).
    Ustack = reshape(view(A.bottom_U, :, :, 1:np), s_m, np * r)
    Tstack = reshape(Tw, np * r, s_n)
    Ccorner = _dense_tile_view(C, A, mt, nt)
    gemm_batched!('N', 'N', T(alpha), [Ustack], [Tstack], T(beta), [Ccorner])
    return C
end
