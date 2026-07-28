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
# Each carves its intermediate buffer from a caller-supplied flat `workspace`, so the
# O_A D_B pass and the D_A O_B pass reuse ONE allocation (see `_diag_times_offdiag_interior!`).

# O_A D_B, fused Stage 1 (A stride-1 axis `:i`): column `j`'s off-diagonal tiles
# are contiguous in `int_U/int_V` and share the right operand `B_jj`, so the A inner-factor
# stacks fuse into one wide GEMM per column.
function _offdiag_diag_interior_fused_gemm!(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha, tile_slots, a_outer_factors, a_inner_factors, workspace; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    ntiles = length(tile_slots)
    rA = maxrank(A)
    (ntiles == 0 || rA == 0) && return C

    _, qkA = regular_grid_size(A)
    nk_off_per_col = qkA - 1
    bn = nominal_tile_size(B, 2)
    workspace_length = rA * ntiles * bn
    v_dense_work = reshape(view(workspace, 1:workspace_length), rA, ntiles, bn)
    v_dense_stack = reshape(view(workspace, 1:workspace_length), rA * ntiles, bn)

    a_inner_stacks = [reshape(view(a_inner_factors, :, :, (((j - 1) * nk_off_per_col + 1):(j * nk_off_per_col))), bn, nk_off_per_col * rA) for j in 1:qkA]
    b_diag_tiles = [_dense_data(_diag_tile_ref(B, j)) for j in 1:qkA]
    v_dense_stack_views = [view(v_dense_stack, ((j - 1) * nk_off_per_col * rA + 1):(j * nk_off_per_col * rA), :) for j in 1:qkA]
    precision_gemm_batched!('T', _opchar(B), one(T), a_inner_stacks, b_diag_tiles, zero(T), v_dense_stack_views, compute)

    tile_coords = [region_tile_coords(A, _INTERIOR, k) for k in tile_slots]
    a_outer_views = [view(a_outer_factors, :, :, k) for k in 1:ntiles]
    v_dense_tile_views = [view(v_dense_work, :, k, :) for k in 1:ntiles]
    output_views = [_output_tile_view(C, A, B, tile_coords[k]...) for k in 1:ntiles]
    precision_gemm_batched!('N', 'N', alpha, a_outer_views, v_dense_tile_views, beta, output_views, compute)
    return C
end

# O_A D_B, tilewise Stage 1 (A stride-1 axis `:k`): one r×r batched GEMM per tile.
function _offdiag_diag_tilebatch_gemm!(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha, tile_slots, a_outer_factors, a_inner_factors, workspace; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    ntiles = length(tile_slots)
    rA = maxrank(A)
    (ntiles == 0 || rA == 0) && return C

    tile_coords = [region_tile_coords(A, _INTERIOR, k) for k in tile_slots]
    bn = nominal_tile_size(B, 2)
    workspace_length = rA * bn * ntiles
    v_dense_work = reshape(view(workspace, 1:workspace_length), rA, bn, ntiles)

    a_inner_views = [view(a_inner_factors, :, :, k) for k in 1:ntiles]
    b_diag_views = [_dense_data(_diag_tile_ref(B, tile_coords[k][2])) for k in 1:ntiles]
    v_dense_views = [view(v_dense_work, :, :, k) for k in 1:ntiles]
    precision_gemm_batched!('T', _opchar(B), one(T), a_inner_views, b_diag_views, zero(T), v_dense_views, compute)

    a_outer_views = [view(a_outer_factors, :, :, k) for k in 1:ntiles]
    output_views = [_output_tile_view(C, A, B, tile_coords[k]...) for k in 1:ntiles]
    precision_gemm_batched!('N', 'N', alpha, a_outer_views, v_dense_views, beta, output_views, compute)
    return C
end

# D_A O_B, fused Stage 1 (B stride-1 axis `:j`): row `i`'s off-diagonal tiles
# are contiguous in `int_U/int_V` and share the left operand `A_ii`, so the B outer-factor
# stacks fuse into one wide GEMM per row.
function _diag_offdiag_interior_fused_gemm!(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha, tile_slots, b_outer_factors, b_inner_factors, workspace; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    ntiles = length(tile_slots)
    rB = maxrank(B)
    (ntiles == 0 || rB == 0) && return C

    qkB, _ = regular_grid_size(B)
    nk_off_per_row = qkB - 1
    bm = nominal_tile_size(A, 1)
    workspace_length = bm * rB * ntiles
    a_diag_b_outer_work = reshape(view(workspace, 1:workspace_length), bm, rB, ntiles)
    a_diag_b_outer_stack = reshape(view(workspace, 1:workspace_length), bm, rB * ntiles)

    a_diag_tiles = [_dense_data(_diag_tile_ref(A, i)) for i in 1:qkB]
    b_outer_stacks = [reshape(view(b_outer_factors, :, :, (((i - 1) * nk_off_per_row + 1):(i * nk_off_per_row))), bm, nk_off_per_row * rB) for i in 1:qkB]
    a_diag_b_outer_stack_views = [view(a_diag_b_outer_stack, :, ((i - 1) * nk_off_per_row * rB + 1):(i * nk_off_per_row * rB)) for i in 1:qkB]
    precision_gemm_batched!(_opchar(A), 'N', one(T), a_diag_tiles, b_outer_stacks, zero(T), a_diag_b_outer_stack_views, compute)

    tile_coords = [region_tile_coords(B, _INTERIOR, k) for k in tile_slots]
    b_inner_views = [view(b_inner_factors, :, :, k) for k in 1:ntiles]
    a_diag_b_outer_tile_views = [view(a_diag_b_outer_work, :, :, k) for k in 1:ntiles]
    output_views = [_output_tile_view(C, A, B, tile_coords[k]...) for k in 1:ntiles]
    precision_gemm_batched!('N', 'T', alpha, a_diag_b_outer_tile_views, b_inner_views, beta, output_views, compute)
    return C
end

# D_A O_B, tilewise Stage 1 (B stride-1 axis `:k`): one b×r batched GEMM per tile.
function _diag_offdiag_tilebatch_gemm!(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha, tile_slots, b_outer_factors, b_inner_factors, workspace; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    ntiles = length(tile_slots)
    rB = maxrank(B)
    (ntiles == 0 || rB == 0) && return C

    tile_coords = [region_tile_coords(B, _INTERIOR, k) for k in tile_slots]
    bm = nominal_tile_size(A, 1)
    workspace_length = bm * rB * ntiles
    a_diag_b_outer_work = reshape(view(workspace, 1:workspace_length), bm, rB, ntiles)

    a_diag_tiles = [_dense_data(_diag_tile_ref(A, tile_coords[k][1])) for k in 1:ntiles]
    b_outer_views = [view(b_outer_factors, :, :, k) for k in 1:ntiles]
    a_diag_b_outer_tile_views = [view(a_diag_b_outer_work, :, :, k) for k in 1:ntiles]
    precision_gemm_batched!(_opchar(A), 'N', one(T), a_diag_tiles, b_outer_views, zero(T), a_diag_b_outer_tile_views, compute)

    b_inner_views = [view(b_inner_factors, :, :, k) for k in 1:ntiles]
    output_views = [_output_tile_view(C, A, B, tile_coords[k]...) for k in 1:ntiles]
    precision_gemm_batched!('N', 'T', alpha, a_diag_b_outer_tile_views, b_inner_views, beta, output_views, compute)
    return C
end

# Component 2: O_A D_B then D_A O_B, interior category only, sharing ONE workspace.
# `O_A D_B` is the first off-diagonal writer (folds β); `D_A O_B` then accumulates
# with β = 1.  If A has zero off-diagonal rank, `D_A O_B` becomes the first writer.
function _diag_times_offdiag_interior!(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha; beta=one(alpha), compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    a_outer_factors = outer_factors(A, _INTERIOR)
    a_inner_factors = inner_factors(A, _INTERIOR)
    b_outer_factors = outer_factors(B, _INTERIOR)
    b_inner_factors = inner_factors(B, _INTERIOR)
    n_int = size(a_outer_factors, 3)                          # == Q(Q-1)
    rmax = max(maxrank(A), maxrank(B))
    (n_int == 0 || rmax == 0) && return C
    bm = nominal_tile_size(A, 1)
    _arena_reset!(arena)
    workspace = _workspace_array!(
        arena, get_backend(A), T, n_int * bm * rmax)

    if maxrank(A) > 0
        if stride1_axis_left(A) isa Stride1Axis{:i}
            _offdiag_diag_interior_fused_gemm!(C, A, B, alpha, axes(a_outer_factors, 3), a_outer_factors, a_inner_factors, workspace; beta=beta, compute)
        else
            _offdiag_diag_tilebatch_gemm!(C, A, B, alpha, axes(a_outer_factors, 3), a_outer_factors, a_inner_factors, workspace; beta=beta, compute)
        end
    end

    beta_after_A = maxrank(A) > 0 ? one(T) : beta
    if maxrank(B) > 0
        if stride1_axis_right(B) isa Stride1Axis{:j}
            _diag_offdiag_interior_fused_gemm!(C, A, B, alpha, axes(b_outer_factors, 3), b_outer_factors, b_inner_factors, workspace; beta=beta_after_A, compute)
        else
            _diag_offdiag_tilebatch_gemm!(C, A, B, alpha, axes(b_outer_factors, 3), b_outer_factors, b_inner_factors, workspace; beta=beta_after_A, compute)
        end
    end
    return C
end

function _diag_diag_gemm!(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha; beta=one(alpha), compute=default_gemm_compute_mode(T)) where {T}
    n_full_diag = min(_nfull_diag_tiles(A), _nfull_diag_tiles(B))
    n_full_diag == 0 && return C

    a_dense_tiles = [_dense_data(_diag_tile_ref(A, i)) for i in 1:n_full_diag]
    b_dense_tiles = [_dense_data(_diag_tile_ref(B, i)) for i in 1:n_full_diag]
    output_views = [_output_tile_view(C, A, B, i, i) for i in 1:n_full_diag]
    precision_gemm_batched!(_opchar(A), _opchar(B), alpha, a_dense_tiles, b_dense_tiles, beta, output_views, compute)
    return C
end

"""
    _offdiag_offdiag_gemm!(C, A, B; alpha, beta=1, budget) -> C

`C_int := β·C_int + α·O_A O_B` over the interior tile grid. Selects the reduction
placement from the operand layouts, then executes Stage 1/2/3 through the
precision-aware batched GEMM dispatcher.

`O_A O_B` touches *every* interior tile, so it is the natural place to fold β. How β
is folded depends on the layout (see `schedule.jl`): the **row family** writes each
tile exactly once, so Stage 3 applies β directly; the **column family** loops the
reduction across runs, so the interior region is pre-scaled once and Stage 3 then
accumulates with β = 1.
"""
function _offdiag_offdiag_gemm!(C, A::LogicalTLROperand{<:Any,<:AbstractTLRMatrix{<:Any,T}}, B::LogicalTLROperand;
    alpha, beta=one(alpha), budget, fold::Union{Nothing,FoldSide}=nothing,
    compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    ops = logical_operands(A, B)
    geom = interior_geometry(A, B)
    return execute_lowrank_gemm!(C, A, B, ops, geom, 1, 1;
                                 alpha, beta, budget, compute, fold, arena)
end

function _offdiag_offdiag_gemm!(C, A::AbstractTLRMatrix{<:Any,T}, B::AbstractTLRMatrix{<:Any,T};
    alpha, beta=one(alpha), budget, fold::Union{Nothing,FoldSide}=nothing,
    transA::Char='N', transB::Char='N', compute=default_gemm_compute_mode(T),
    arena=nothing) where {T}
    return _offdiag_offdiag_gemm!(C, logical_operand(A, transA), logical_operand(B, transB);
                                  alpha, beta, budget, fold, compute, arena)
end

"""
    tlr_gemm_int_by_int(C, A, B, alpha, beta; budget) -> C

Compute `C_int := β·C_int + α·A_int B_int` over the interior tile grid. `A_int B_int`
splits as `D_A D_B + O_A D_B + D_A O_B + O_A O_B`; the hard term `O_A O_B` touches
*every* interior tile, so it runs first and folds β (write-once row family, or
region pre-scale in the column family — see `_offdiag_offdiag_gemm!`), and the three
diagonal components then accumulate with β = 1. This inversion (β folded in `O_A O_B`
rather than the diagonal terms) is what lets the `PaddedFTLRMatrix` interior — which has
*only* `O_A O_B` — share the same β-folding path. Order-only dependencies — no host
sync; the caller places this whole term on a region stream (see `gemm!`).
"""
function tlr_gemm_int_by_int(C, A::LogicalTLROperand{<:Any,<:TLRMatrix{<:Any,T}}, B::LogicalTLROperand{<:Any,<:TLRMatrix}, alpha, beta; budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    _offdiag_offdiag_gemm!(C, A, B; alpha=alpha, beta=beta, budget=budget, compute, arena)  # O_A O_B (folds β)
    _diag_diag_gemm!(C, A, B, alpha; beta=one(T), compute)                           # D_A D_B
    _diag_times_offdiag_interior!(C, A, B, alpha; beta=one(T), compute, arena)       # O_A D_B + D_A O_B
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
function tlr_gemm_rpanel_by_bpanel(C, A::LogicalTLROperand{<:Any,<:AbstractTLRMatrix{<:Any,T}}, B::LogicalTLROperand, alpha;
    beta=one(alpha), budget::Int, compute=default_gemm_compute_mode(T), arena=nothing) where {T}
    qm = region_tile_count(A, _RIGHT)
    qn = region_tile_count(B, _BOTTOM)
    (qm == 0 || qn == 0) && return C
    return execute_lowrank_term!(C, A, B, _right_pair(A), _bottom_pair(B),
                                 qm, 1, qn, 1, 1;
                                 alpha, beta, budget, compute, arena)
end
