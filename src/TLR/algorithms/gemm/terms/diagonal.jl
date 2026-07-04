# ─── The three "easy" terms of  A·B = D_A D_B + O_A D_B + D_A O_B + O_A O_B ──────
#
# These run first and fold `β·C` into their batched GEMMs (the `beta` kwarg): the
# first GEMM to touch a C tile carries `β`, later writers use `β = 1`, and the
# hard term O_A O_B then accumulates last with `β = 1`.  Diagonal tiles are
# written only by D_A D_B; off-diagonal tiles by both O_A D_B and D_A O_B (so the
# caller passes `β` to the first present and `1` to the second).  Each term
# targets distinct C tiles within itself → a single batched pair of GEMMs.
# (Happy path: square, m % b == 0.)

@inline function _dense_tile_view(C, A::TLRMatrix, tile_i::Int, tile_j::Int)
    p0, q0 = tile_origin_coords(A, tile_i, tile_j)
    tm, tn = tile_size(A, tile_i, tile_j)
    return view(C, p0:(p0 + tm - 1), q0:(q0 + tn - 1))
end

function _offdiag_diag_category!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha, obs, U, V; beta=one(T)) where {T}
    n_cat = length(obs)
    rA = A.maxrank
    (n_cat == 0 || rA == 0) && return C

    coords = [_offdiag_coords(A, ob) for ob in obs]
    Mws = KernelAbstractions.allocate(A.backend, T, rA, size(V, 1), n_cat)

    Vvec  = [view(V, :, :, k) for k in 1:n_cat]
    BDvec = [_diag_tile_view(B, coords[k][2]) for k in 1:n_cat]
    Mvec  = [view(Mws, :, :, k) for k in 1:n_cat]
    gemm_batched!('T', 'N', one(T), Vvec, BDvec, zero(T), Mvec)

    Uvec = [view(U, :, :, k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, A, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'N', alpha, Uvec, Mvec, T(beta), Cvec)
    return C
end

function _diag_offdiag_category!(C, A::TLRMatrix, B::TLRMatrix{<:Any,T}, alpha, obs, U, V; beta=one(T)) where {T}
    n_cat = length(obs)
    rB = B.maxrank
    (n_cat == 0 || rB == 0) && return C

    coords = [_offdiag_coords(B, ob) for ob in obs]
    Nws = KernelAbstractions.allocate(B.backend, T, size(U, 1), rB, n_cat)

    ADvec = [_diag_tile_view(A, coords[k][1]) for k in 1:n_cat]
    Wvec  = [view(U, :, :, k) for k in 1:n_cat]
    Nvec  = [view(Nws, :, :, k) for k in 1:n_cat]
    gemm_batched!('N', 'N', one(T), ADvec, Wvec, zero(T), Nvec)

    Zvec = [view(V, :, :, k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, B, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'T', alpha, Nvec, Zvec, T(beta), Cvec)
    return C
end

# D_A D_B :  C_ii := β·C_ii + α · A_ii · B_ii      (diagonal output only)
function _diag_diag!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha; beta=one(T)) where {T}
    b = T(beta)
    n_full_diag = min(_nfull_diag_tiles(A), _nfull_diag_tiles(B))
    if n_full_diag > 0
        Avec = [view(A.D, :, :, i) for i in 1:n_full_diag]
        Bvec = [view(B.D, :, :, i) for i in 1:n_full_diag]
        Cvec = [_dense_tile_view(C, A, i, i) for i in 1:n_full_diag]
        gemm_batched!('N', 'N', alpha, Avec, Bvec, b, Cvec)
    end
    if size(A.D_corner, 3) != 0 && size(B.D_corner, 3) != 0
        tile_k = ndiag_tiles(A)
        mul!(_dense_tile_view(C, A, tile_k, tile_k),
             _diag_tile_view(A, tile_k), _diag_tile_view(B, tile_k),
             alpha, b)
    end
    return C
end

# O_A D_B :  C_ij := β·C_ij + α · A_ij · B_jj  (i≠j) = U_ij (V_ijᵀ B_jj)  over A's off-diag tiles
function _offdiag_diag!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha; beta=one(T)) where {T}
    _offdiag_diag_category!(C, A, B, alpha, A.obs_int, A.int_U, A.int_V; beta=beta)
    _offdiag_diag_category!(C, A, B, alpha, A.obs_right, A.right_U, A.right_V; beta=beta)
    _offdiag_diag_category!(C, A, B, alpha, A.obs_bottom, A.bottom_U, A.bottom_V; beta=beta)
    return C
end

# D_A O_B :  C_ij := β·C_ij + α · A_ii · B_ij  (i≠j) = (A_ii W_ij) Z_ijᵀ  over B's off-diag tiles
function _diag_offdiag!(C, A::TLRMatrix, B::TLRMatrix{<:Any,T}, alpha; beta=one(T)) where {T}
    _diag_offdiag_category!(C, A, B, alpha, B.obs_int, B.int_U, B.int_V; beta=beta)
    _diag_offdiag_category!(C, A, B, alpha, B.obs_right, B.right_U, B.right_V; beta=beta)
    _diag_offdiag_category!(C, A, B, alpha, B.obs_bottom, B.bottom_U, B.bottom_V; beta=beta)
    return C
end
