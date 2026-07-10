# ─── Interior region of C:  C_int = A_int B_int + u_A v_Bᵀ ────────────────────────
#
# Two terms land in the interior (output rows/cols 1..Q·b, Q = ⌊m/b⌋):
#   `tlr_gemm_int_by_int`       — A_int · B_int, the regular interior product.
#   `tlr_gemm_rpanel_by_bpanel` — u_A v_Bᵀ, the right-panel × bottom-panel outer
#                                 product (boundary factors folding back into the
#                                 interior; fires only when m % b ≠ 0).
#
# `tlr_gemm_int_by_int` splits A_int B_int = (D_A+O_A)(D_B+O_B) into three components:
#   1. diag × diag        D_A D_B              — interior diagonal tiles
#   2. diag × offdiag     O_A D_B + D_A O_B    — interior off-diagonal tiles
#   3. offdiag × offdiag  O_A O_B              — all interior tiles (the hard term)
# Components 1 and 2 write DISJOINT tiles and fold β·C into their first GEMM, so they
# run concurrently on two streams; component 3 accumulates (β = 1) after both sync.

# ─── Component 2 kernels — interior easy off-diagonal terms ───────────────────────
#
# Each carves its intermediate buffer from a caller-supplied flat `scratch`, so the
# O_A D_B pass and the D_A O_B pass reuse ONE allocation (see `_diag_times_offdiag_interior!`).

# O_A D_B, fused Stage 1 (A stride-1 axis `:i`): column `j`'s `Q-1` off-diagonal tiles
# are contiguous in `int_U/int_V` and share the right operand `B_jj`, so their `V`s
# fuse into one wide GEMM per column — batch `Q` instead of `Q(Q-1)`.
function _offdiag_diag_interior_fused_gemm!(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T, slots, U, V, scratch; beta::T=one(T)) where {T,BackendT}
    n_cat = length(slots)
    rA = maxrank(A)
    (n_cat == 0 || rA == 0) && return C

    Q, _ = _full_regular_grid(A) # interior sub-grid is Q×Q
    per = Q - 1  # off-diagonal tiles per column j
    b_out = size(V, 1)
    len = rA * n_cat * b_out
    Mws = reshape(view(scratch, 1:len), rA, n_cat, b_out)
    Mws2 = reshape(view(scratch, 1:len), rA * n_cat, b_out)

    Vg = [reshape(view(V,:,:,(((j-1)*per+1):(j*per))), b_out, per * rA) for j in 1:Q]
    Bg = [_diag_tile_view(B, j) for j in 1:Q]
    Mg = [view(Mws2, ((j-1)*per*rA+1):(j*per*rA), :) for j in 1:Q]
    gemm_batched!('T', 'N', one(T), Vg, Bg, zero(T), Mg)

    coords = [_category_coords(A, _TILE_INT, k) for k in slots]
    Uvec = [view(U,:,:,k) for k in 1:n_cat]
    Mvec = [view(Mws,:,k,:) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, A, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'N', alpha, Uvec, Mvec, beta, Cvec)
    return C
end

# O_A D_B, tilewise Stage 1 (A stride-1 axis `:k`): one r×r batched GEMM per tile.
function _offdiag_diag_tilebatch_gemm!(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T, slots, U, V, scratch; beta::T=one(T)) where {T,BackendT}
    n_cat = length(slots)
    rA = maxrank(A)
    (n_cat == 0 || rA == 0) && return C

    coords = [_category_coords(A, _TILE_INT, k) for k in slots]
    b_out = size(V, 1)
    len = rA * b_out * n_cat
    Mws = reshape(view(scratch, 1:len), rA, b_out, n_cat)

    Vvec = [view(V,:,:,k) for k in 1:n_cat]
    BDvec = [_diag_tile_view(B, coords[k][2]) for k in 1:n_cat]
    Mvec = [view(Mws,:,:,k) for k in 1:n_cat]
    gemm_batched!('T', 'N', one(T), Vvec, BDvec, zero(T), Mvec)

    Uvec = [view(U,:,:,k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, A, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'N', alpha, Uvec, Mvec, beta, Cvec)
    return C
end

# D_A O_B, fused Stage 1 (B stride-1 axis `:j`): row `i`'s `Q-1` off-diagonal tiles
# are contiguous in `int_U/int_V` and share the left operand `A_ii`, so their `W`s
# fuse into one wide GEMM per row.
function _diag_offdiag_interior_fused_gemm!(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T, slots, U, V, scratch; beta::T=one(T)) where {T,BackendT}
    n_cat = length(slots)
    rB = maxrank(B)
    (n_cat == 0 || rB == 0) && return C

    Q, _ = _full_regular_grid(B)             # interior sub-grid is Q×Q
    per = Q - 1                          # off-diagonal tiles per row (panel `i`)
    b_in = size(U, 1)
    len = b_in * rB * n_cat
    Nws = reshape(view(scratch, 1:len), b_in, rB, n_cat)
    Nws2 = reshape(view(scratch, 1:len), b_in, rB * n_cat)

    ADg = [_diag_tile_view(A, i) for i in 1:Q]
    Wg = [reshape(view(U,:,:,(((i-1)*per+1):(i*per))), b_in, per * rB) for i in 1:Q]
    Ng = [view(Nws2, :, ((i-1)*per*rB+1):(i*per*rB)) for i in 1:Q]
    gemm_batched!('N', 'N', one(T), ADg, Wg, zero(T), Ng)

    coords = [_category_coords(B, _TILE_INT, k) for k in slots]
    Zvec = [view(V,:,:,k) for k in 1:n_cat]
    Nvec = [view(Nws,:,:,k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, B, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'T', alpha, Nvec, Zvec, beta, Cvec)
    return C
end

# D_A O_B, tilewise Stage 1 (B stride-1 axis `:k`): one b×r batched GEMM per tile.
function _diag_offdiag_tilebatch_gemm!(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T, slots, U, V, scratch; beta::T=one(T)) where {T,BackendT}
    n_cat = length(slots)
    rB = maxrank(B)
    (n_cat == 0 || rB == 0) && return C

    coords = [_category_coords(B, _TILE_INT, k) for k in slots]
    b_in = size(U, 1)
    len = b_in * rB * n_cat
    Nws = reshape(view(scratch, 1:len), b_in, rB, n_cat)

    ADvec = [_diag_tile_view(A, coords[k][1]) for k in 1:n_cat]
    Wvec = [view(U,:,:,k) for k in 1:n_cat]
    Nvec = [view(Nws,:,:,k) for k in 1:n_cat]
    gemm_batched!('N', 'N', one(T), ADvec, Wvec, zero(T), Nvec)

    Zvec = [view(V,:,:,k) for k in 1:n_cat]
    Cvec = [_dense_tile_view(C, B, coords[k]...) for k in 1:n_cat]
    gemm_batched!('N', 'T', alpha, Nvec, Zvec, beta, Cvec)
    return C
end

# Component 2: O_A D_B then D_A O_B, interior category only, sharing ONE scratch.
# `O_A D_B` is the first off-diagonal writer (folds β); `D_A O_B` then accumulates
# with β = 1.  If A has zero off-diagonal rank, `D_A O_B` becomes the first writer.
function _diag_times_offdiag_interior!(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T,BackendT}
    n_int = size(A.int_U, 3)                     # == Q(Q-1)
    rmax = max(maxrank(A), maxrank(B))
    (n_int == 0 || rmax == 0) && return C
    b = nominal_tile_size(A, 1)
    scratch = allocate(A.backend, T, n_int * b * rmax)

    if maxrank(A) > 0
        if stride1_axis_left(A) isa Stride1Axis{:i}
            _offdiag_diag_interior_fused_gemm!(C, A, B, alpha, axes(A.int_U, 3), A.int_U, A.int_V, scratch; beta=beta)
        else
            _offdiag_diag_tilebatch_gemm!(C, A, B, alpha, axes(A.int_U, 3), A.int_U, A.int_V, scratch; beta=beta)
        end
    end

    beta_B = maxrank(A) > 0 ? one(T) : beta
    if maxrank(B) > 0
        if stride1_axis_right(B) isa Stride1Axis{:j}
            _diag_offdiag_interior_fused_gemm!(C, A, B, alpha, axes(B.int_U, 3), B.int_U, B.int_V, scratch; beta=beta_B)
        else
            _diag_offdiag_tilebatch_gemm!(C, A, B, alpha, axes(B.int_U, 3), B.int_U, B.int_V, scratch; beta=beta_B)
        end
    end
    return C
end

function _diag_diag_gemm!(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T; beta::T=one(T)) where {T,BackendT}
    n_full_diag = min(_nfull_diag_tiles(A), _nfull_diag_tiles(B))
    n_full_diag == 0 && return C

    Avec = [view(A.D,:,:,i) for i in 1:n_full_diag]
    Bvec = [view(B.D,:,:,i) for i in 1:n_full_diag]
    Cvec = [_dense_tile_view(C, A, i, i) for i in 1:n_full_diag]
    gemm_batched!('N', 'N', alpha, Avec, Bvec, beta, Cvec)
    return C
end

"""
    _offdiag_offdiag_gemm!(C, A, B; alpha, budget) -> C

Accumulate `alpha · O_A O_B` into the dense `C`.  Selects the reduction placement
from the operand layouts, then for each budgeted run lowers Stage 1/2/3 straight
to `gemm_batched!` via `execute_stage!`.
"""
function _offdiag_offdiag_gemm!(C, A::AbstractTLRMatrix{<:Any,T}, B::AbstractTLRMatrix{<:Any,T};
    alpha::T, budget) where {T}
    ops = logical_operands(A, B)
    # tiles_per_row(ops.av) == 0 covers both `nt == 1` (SkipDiag: only the diagonal)
    # and an empty grid; zero rank on either side leaves nothing to contract.
    (tiles_per_row(ops.av) == 0 || rankdim(ops.av) == 0 || rankdim(ops.bw) == 0) && return C

    placement = k_axis_schedule(stride1_axis_left(A), stride1_axis_right(B))
    ws = allocate_workspace(placement, ops, C, budget)

    @inbounds for run in runs(placement, ops, budget)
        prepare_run!(placement, run, ws)
        execute_stage!(stage1(placement, run, ops, ws))
        execute_stage!(stage2(placement, run, ops, ws))
        execute_stage!(stage3(placement, run, ops, ws, C, alpha))
    end
    return C
end

"""
    tlr_gemm_int_by_int(C, A, B, alpha, beta; budget) -> C

Compute `C_int := β·C_int + α·A_int B_int` over the interior tile grid, as three
components issued in order on the current stream: diag×diag (component 1) and
diag×offdiag (component 2) fold β into disjoint tile sets, then the hard term
O_A O_B (component 3) accumulates (β = 1). Order-only dependencies — no host sync;
the caller places this whole term on a region stream (see `gemm!`).
"""
function tlr_gemm_int_by_int(C, A::TLRDenseDiagMatrix{BackendT,T}, B::TLRDenseDiagMatrix{BackendT,T}, alpha::T, beta::T; budget::Int) where {T,BackendT}
    Q, _ = _full_regular_grid(A)
    b = nominal_tile_size(A, 1)

    # degenerate case: block-diagonal product
    if maxrank(A) == 0 && maxrank(B) == 0
        # scale the interior block explicitly, then add the diagonal (accumulate, β = 1).
        _scale_output!(view(C, 1:(Q*b), 1:(Q*b)), beta)
        _diag_diag_gemm!(C, A, B, alpha)
        return C
    end

    _diag_diag_gemm!(C, A, B, alpha; beta=beta)                         # component 1 (folds β)
    _diag_times_offdiag_interior!(C, A, B, alpha; beta=beta)            # component 2 (folds β)
    _offdiag_offdiag_gemm!(C, A, B; alpha=alpha, budget=budget)         # component 3 (β = 1)
    return C
end

"""
    tlr_gemm_rpanel_by_bpanel(C, A, B, alpha; beta=1, budget) -> C

Accumulate `α · u_A v_Bᵀ` into the interior of the dense `C`:
`C_int := beta·C_int + α·u_A v_Bᵀ`.  No-op when `n_A % b == 0` (no boundary column
in `A`) or the pairing is incomplete (non-square boundary).
"""
# Pure low-rank panels (A's right column × B's bottom row): identical for both
# container types, so it dispatches on `AbstractTLRMatrix`.
function tlr_gemm_rpanel_by_bpanel(C, A::AbstractTLRMatrix{BackendT,T}, B::AbstractTLRMatrix{BackendT,T}, alpha::T;
    beta::T=one(T), budget::Int) where {T,BackendT}
    Qi = _right_panel_tiles(A)   # A right-panel tiles → output rows
    Qj = _bottom_panel_tiles(B)  # B bottom-panel tiles → output cols
    (Qi == 0 || Qj == 0) && return C

    rA = maxrank(A)
    rB = maxrank(B)
    (rA == 0 || rB == 0) && return C

    s = size(A.right_V, 1) # shared contraction tail (== size(B.bottom_U, 1))
    b = nominal_tile_size(A, 1)

    Vstack = reshape(A.right_V, s, rA * Qi) # [V_1 | … | V_Qi]   (s × Qi·rA)
    Wstack = reshape(B.bottom_U, s, rB * Qj) # [W_1 | … | W_Qj]  (s × Qj·rB)

    # column-block width fitting the budget
    per_col = max(Qi * rA * (rB + b) * sizeof(T), 1)  # S col-block (Qi·rA × rB) + T (Qi·rA × b).
    maxJ = clamp(div(budget, per_col), 1, Qj)

    Sbuf = allocate(A.backend, T, Qi * rA, maxJ * rB)
    Tbuf = allocate(A.backend, T, Qi * rA, b, maxJ)

    s3u = Vector{typeof(view(A.right_U,:,:,1))}()
    s3t = Vector{typeof(view(Tbuf, 1:rA, :, 1))}()
    s3c = Vector{typeof(_output_tile_view(C, A, B, 1, 1))}()

    @inbounds for jrange in Iterators.partition(1:Qj, maxJ)
        j0 = first(jrange)
        j1 = last(jrange)
        Δ = length(jrange)

        # Stage 1
        Wsub = view(Wstack, :, ((j0-1)*rB+1):(j1*rB))
        Ssub = view(Sbuf, :, 1:(Δ*rB))
        gemm_batched!('T', 'N', one(T), [Vstack], [Wsub], zero(T), [Ssub])

        # Stage 2
        S2 = reshape(Ssub, Qi * rA, rB, Δ)
        Z2 = view(B.bottom_V,:,:,jrange)
        T2 = view(Tbuf,:,:,(1:Δ))
        gemm_batched!('N', 'T', one(T), S2, Z2, zero(T), T2)

        # Stage 3
        empty!(s3u);
        empty!(s3t);
        empty!(s3c)
        @inbounds for (jl, j) in enumerate(jrange)
            for i in 1:Qi
                push!(s3u, view(A.right_U,:,:,i))
                push!(s3t, view(Tbuf, ((i-1)*rA+1):(i*rA), :, jl))
                push!(s3c, _output_tile_view(C, A, B, i, j))
            end
        end
        gemm_batched!('N', 'N', alpha, s3u, s3t, beta, s3c)
    end
    return C
end
