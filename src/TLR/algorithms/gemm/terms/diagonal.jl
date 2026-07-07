# These run first and fold `β·C` into their batched GEMMs (the `beta` kwarg): the
# first GEMM to touch a C tile carries `β`, later writers use `β = 1`, and the
# hard term O_A O_B then accumulates last with `β = 1`.  Diagonal tiles are
# written only by D_A D_B; off-diagonal tiles by both O_A D_B and D_A O_B (so the
# caller passes `β` to the first present and `1` to the second).  Each term
# targets distinct C tiles within itself → a single batched pair of GEMMs.
# (Happy path: square, m % b == 0.)

function _offdiag_diag_interior_gemm!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha, obs, U, V, ::Stride1Axis{:i}; beta=one(T)) where {T}
    n_cat = length(obs)
    rA = A.maxrank
    (n_cat == 0 || rA == 0) && return C

    Q, _ = _interior_grid(A)             # interior sub-grid is Q×Q
    per = Q - 1                          # off-diagonal tiles per column (panel `j`)
    b_out = size(V, 1)

    # Mws laid out (r, n_cat, b): a column's `per` tiles then merge with r into one
    # contiguous row-block of `reshape(Mws, r*n_cat, b)`.
    Mws = allocate(A.backend, T, rA, n_cat, b_out)
    Mws2 = reshape(Mws, rA * n_cat, b_out)

    Vg = [reshape(view(V,:,:,(((j-1)*per+1):(j*per))), b_out, per * rA) for j in 1:Q]
    Bg = [_diag_tile_view(B, j) for j in 1:Q]
    Mg = [view(Mws2, ((j-1)*per*rA+1):(j*per*rA), :) for j in 1:Q]
    gemm_batched!('T', 'N', one(T), Vg, Bg, zero(T), Mg)

    coords = [_offdiag_coords(A, ob) for ob in obs]
    Uvec = [view(U,:,:,k) for k in 1:n_cat]
    Mvec = [view(Mws,:,k,:) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, A, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'N', alpha, Uvec, Mvec, T(beta), Cvec)
    return C
end

# D_A O_B, fused Stage 1 (B stride-1 axis `:j`): row `i`'s
# `Q-1` off-diagonal tiles are contiguous in `int_U/int_V` and all share the left
# operand `A_ii`, so their `W`s fuse into one wide GEMM per row — batch size `Q`
# instead of `Q(Q-1)`.
function _diag_offdiag_interior_gemm!(C, A::TLRMatrix, B::TLRMatrix{<:Any,T}, alpha, obs, U, V, ::Stride1Axis{:j}; beta=one(T)) where {T}
    n_cat = length(obs)
    rB = B.maxrank
    (n_cat == 0 || rB == 0) && return C

    Q, _ = _interior_grid(B)             # interior sub-grid is Q×Q
    per = Q - 1                          # off-diagonal tiles per row (panel `i`)
    b_in = size(U, 1)

    Nws = allocate(B.backend, T, b_in, rB, n_cat)
    Nws2 = reshape(Nws, b_in, rB * n_cat)

    ADg = [_diag_tile_view(A, i) for i in 1:Q]
    Wg = [reshape(view(U,:,:,(((i-1)*per+1):(i*per))), b_in, per * rB) for i in 1:Q]
    Ng = [view(Nws2, :, ((i-1)*per*rB+1):(i*per*rB)) for i in 1:Q]
    gemm_batched!('N', 'N', one(T), ADg, Wg, zero(T), Ng)

    coords = [_offdiag_coords(B, ob) for ob in obs]
    Zvec = [view(V,:,:,k) for k in 1:n_cat]
    Nvec = [view(Nws,:,:,k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, B, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'T', alpha, Nvec, Zvec, T(beta), Cvec)
    return C
end

function _offdiag_diag_tilebatch_gemm!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha, obs, U, V; beta=one(T)) where {T}
    n_cat = length(obs)
    rA = A.maxrank
    (n_cat == 0 || rA == 0) && return C

    coords = [_offdiag_coords(A, ob) for ob in obs]
    Mws = allocate(A.backend, T, rA, size(V, 1), n_cat)

    Vvec = [view(V,:,:,k) for k in 1:n_cat]
    BDvec = [_diag_tile_view(B, coords[k][2]) for k in 1:n_cat]
    Mvec = [view(Mws,:,:,k) for k in 1:n_cat]
    gemm_batched!('T', 'N', one(T), Vvec, BDvec, zero(T), Mvec)

    Uvec = [view(U,:,:,k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, A, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'N', alpha, Uvec, Mvec, T(beta), Cvec)
    return C
end

function _offdiag_diag_interior_gemm!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha, obs, U, V, ::Stride1Axis; beta=one(T)) where {T}
    return _offdiag_diag_tilebatch_gemm!(C, A, B, alpha, obs, U, V; beta=beta)
end

function _diag_offdiag_tilebatch_gemm!(C, A::TLRMatrix, B::TLRMatrix{<:Any,T}, alpha, obs, U, V; beta=one(T)) where {T}
    n_cat = length(obs)
    rB = B.maxrank
    (n_cat == 0 || rB == 0) && return C

    coords = [_offdiag_coords(B, ob) for ob in obs]
    Nws = allocate(B.backend, T, size(U, 1), rB, n_cat)

    ADvec = [_diag_tile_view(A, coords[k][1]) for k in 1:n_cat]
    Wvec = [view(U,:,:,k) for k in 1:n_cat]
    Nvec = [view(Nws,:,:,k) for k in 1:n_cat]
    gemm_batched!('N', 'N', one(T), ADvec, Wvec, zero(T), Nvec)

    Zvec = [view(V,:,:,k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, B, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'T', alpha, Nvec, Zvec, T(beta), Cvec)
    return C
end

function _diag_offdiag_interior_gemm!(C, A::TLRMatrix, B::TLRMatrix{<:Any,T}, alpha, obs, U, V, ::Stride1Axis; beta=one(T)) where {T}
    return _diag_offdiag_tilebatch_gemm!(C, A, B, alpha, obs, U, V; beta=beta)
end

# D_A D_B :  C_ii := β·C_ii + α · A_ii · B_ii      (diagonal output only)
function _diag_diag_gemm!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha; beta=one(T)) where {T}
    b = T(beta)
    n_full_diag = min(_nfull_diag_tiles(A), _nfull_diag_tiles(B))
    if n_full_diag > 0
        Avec = [view(A.D,:,:,i) for i in 1:n_full_diag]
        Bvec = [view(B.D,:,:,i) for i in 1:n_full_diag]
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
#
# The interior category fuses Stage 1 across each shared column when `A`'s
# stride-1 axis is `:i` (`TileColMajor`).
# Right/bottom boundary tiles keep the per-tile path (mixed tile shapes there).
function _offdiag_diag_gemm!(C, A::TLRMatrix{<:Any,T}, B::TLRMatrix, alpha; beta=one(T)) where {T}
    _offdiag_diag_interior_gemm!(C, A, B, alpha, A.obs_int, A.int_U, A.int_V, stride1_axis_left(A); beta=beta)
    _offdiag_diag_tilebatch_gemm!(C, A, B, alpha, A.obs_right, A.right_U, A.right_V; beta=beta)
    _offdiag_diag_tilebatch_gemm!(C, A, B, alpha, A.obs_bottom, A.bottom_U, A.bottom_V; beta=beta)
    return C
end

# D_A O_B :  C_ij := β·C_ij + α · A_ii · B_ij  (i≠j) = (A_ii W_ij) Z_ijᵀ  over B's off-diag tiles
#
# The interior category fuses Stage 1 across each shared row when `B`'s
# stride-1 axis is `:j` (`TileRowMajor`).
# Right/bottom boundary tiles keep the per-tile path (mixed tile shapes there).
function _diag_offdiag_gemm!(C, A::TLRMatrix, B::TLRMatrix{<:Any,T}, alpha; beta=one(T)) where {T}
    _diag_offdiag_interior_gemm!(C, A, B, alpha, B.obs_int, B.int_U, B.int_V, stride1_axis_right(B); beta=beta)
    _diag_offdiag_tilebatch_gemm!(C, A, B, alpha, B.obs_right, B.right_U, B.right_V; beta=beta)
    _diag_offdiag_tilebatch_gemm!(C, A, B, alpha, B.obs_bottom, B.bottom_U, B.bottom_V; beta=beta)
    return C
end
