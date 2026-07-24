# End-to-end row-basis driver. It is intentionally narrow: physical
# row-major A and column-major B make both panel stacks zero-copy.

@inline _packed_column_panel(p::InteriorOperand{Kind,OrderT}, j::Int) where {Kind<:GridKind,OrderT<:TileColMajor} = colpanel(p, j)

@inline _packed_row_panel(p::InteriorOperand{Kind,OrderT}, i::Int) where {Kind<:GridKind,OrderT<:TileRowMajor} = rowpanel(p, i)

function _packed_row_panel(p::InteriorOperand, i::Int)
    b, rA, K = size(p.data, 1), size(p.data, 2), p.qn
    panel = allocate(get_backend(p.data), eltype(p.data), b, rA, K)
    @inbounds for k in 1:K
        copyto!(view(panel, :, :, k), tilefactor(p, i, k))
    end
    return panel
end

function _packed_column_panel(p::InteriorOperand, j::Int)
    b, rB, K = size(p.data, 1), size(p.data, 2), p.qm
    panel = allocate(get_backend(p.data), eltype(p.data), b, rB, K)
    @inbounds for k in 1:K
        copyto!(view(panel, :, :, k), tilefactor(p, k, j))
    end
    return panel
end

# Batched Stage 2 for a whole output row (beta == 0). Fills `Vm[:, :, j]` with
# `alpha * M[i,j]` for every output column `j`, where `M[i,j] = Σ_k Z[k,j] (P[k]
# V[k]' W[k,j])'`. All qn columns of the row share `Vrow` and `Pblocks`, so the
# per-(k,j) contraction GEMMs and the terminal `Zstack * Rstack` GEMM run as three
# batched calls (batch dims K*qn, K*qn, qn) instead of qn sequential per-tile calls.
# This is the concurrency the shared row basis exists to enable.
function _accumulate_row_block!(Vm::AbstractArray{T,3}, Vrow::AbstractArray{T,3},
                                Pblocks::AbstractArray{T,3}, BpU::InteriorOperand,
                                BpV::InteriorOperand, qn::Int, alpha::T, compute) where {T}
    kd, rA, K = size(Vrow)
    t = size(Pblocks, 1)
    rB = size(tilefactor(BpU, 1, 1), 2)
    bn = size(Vm, 1)
    backend = get_backend(Vrow)
    adj = _adjoint_blas_char(T)
    Rstack = allocate(backend, T, K * rB, t, qn)
    rslab(k, j) = view(Rstack, (k - 1) * rB + 1:k * rB, :, j)   # R[k,j] within slab j

    if t <= rA
        # T[k] = V[k] * P[k]'  (kd × t), independent of j — computed once.
        Tpanel = allocate(backend, T, kd, t, K)
        precision_gemm_batched!('N', adj, one(T),
            [view(Vrow, :, :, k) for k in 1:K],
            [view(Pblocks, :, :, k) for k in 1:K], zero(T),
            [view(Tpanel, :, :, k) for k in 1:K], compute)
        # R[k,j] = W[k,j]' * T[k]  (rB × t).
        precision_gemm_batched!(adj, 'N', one(T),
            [tilefactor(BpU, k, j) for j in 1:qn for k in 1:K],
            [view(Tpanel, :, :, k) for j in 1:qn for k in 1:K], zero(T),
            [rslab(k, j) for j in 1:qn for k in 1:K], compute)
    else
        # S[k,j] = V[k]' * W[k,j]  (rA × rB), one Sbuf slab per (k,j) enumeration.
        Sbuf = allocate(backend, T, rA, rB, K * qn)
        Sslab = [view(Sbuf, :, :, n) for n in 1:(K * qn)]
        precision_gemm_batched!(adj, 'N', one(T),
            [view(Vrow, :, :, k) for j in 1:qn for k in 1:K],
            [tilefactor(BpU, k, j) for j in 1:qn for k in 1:K], zero(T),
            Sslab, compute)
        # R[k,j] = S[k,j]' * P[k]'  (rB × t).
        precision_gemm_batched!(adj, adj, one(T),
            Sslab,
            [view(Pblocks, :, :, k) for j in 1:qn for k in 1:K], zero(T),
            [rslab(k, j) for j in 1:qn for k in 1:K], compute)
    end
    # Terminal: M[i,j] = Zstack_j * Rstack_j  (bn × t), α folded in, batched over j.
    precision_gemm_batched!('N', 'N', alpha,
        [reshape(_packed_column_panel(BpV, j), bn, K * rB) for j in 1:qn],
        [view(Rstack, :, :, j) for j in 1:qn], zero(T),
        [view(Vm, :, :, j) for j in 1:qn], compute)
    return Vm
end

# ── Saturation guard (alg.md §6.4) ───────────────────────────────────────────
# The shared row basis only pays off while `t ≪ b`. A saturated row (`t ≥ θ·b`)
# compresses nothing: it would do dense-sized coefficient work *plus* the basis
# overhead, measurably slower than the M4 dense-slab sink. Such rows are routed
# through the M4 machinery via single-row runs. The fallback exists only for
# `beta == 0` and the row family (`KAsGemmK`); otherwise rows stay on the
# row-basis path, which remains correct, just slower.
#
# Two devices keep the detection cost negligible:
#   * the sketch is capped at `S = ⌈θ·b⌉` when the guard is armed — a row that
#     needs more than the cap is routed to dense anyway, so the probe never
#     builds a wider basis than the row-basis path could use;
#   * after `SAT_STREAK_CUTOFF` consecutive saturated rows the remaining rows
#     are routed to dense without probing at all (routing is a performance
#     choice — both paths meet the requested tolerance).
const SAT_STREAK_CUTOFF = 2

# Build the M4 lowering context once per call, or `nothing` when the layout
# lowers to the column family (no M4 TLR-output support).
function _m4_row_context(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix, budget::Int)
    LA = logical_operand(A, 'N')
    LB = logical_operand(B, 'N')
    ops = logical_operands(LA, LB)
    geom = interior_geometry(LA, LB)
    fold = choose_fold(ops)
    placement = placement_for_fold(fold, ops)
    placement isa KAsGemmK || return nothing
    ws = _alloc_tlr_output_workspace(C, geom, placement, ops, budget, fold)
    return (; ops, geom, placement, fold, ws)
end

# Compute one output row through the M4 dense-slab sink: single-row runs chunked
# to the workspace's column capacity. `_compress_run_into_factors!` writes
# `C.ranks`/`C.resid` directly, so the driver's host mirrors are resynced.
function _m4_row!(C::TLRMatrix, m4, i::Int, alpha, compute,
                  ranks_host, resid_host; eps_sq::Float64, rel::Bool)
    (; ops, geom, placement, fold, ws) = m4
    qn = geom.q_n
    @inbounds for j0 in 1:ws.maxJ:qn
        run = RowRun(i, i, j0, min(j0 + ws.maxJ - 1, qn))
        prepare_run!(placement, run, ws.stage_ws)
        execute_stage1!(placement, run, ops, ws.stage_ws, compute)
        execute_stage2!(placement, fold, run, ops, ws.stage_ws, compute)
        execute_slab_stage3!(placement, fold, run, ops, ws.stage_ws, ws.slab,
                             alpha, zero(alpha), compute)
        _compress_run_into_factors!(C, ws, run, geom.bm, geom.bn; eps_sq, rel)
    end
    @inbounds for j in 1:qn
        ridx = _rank_index(C, i, j)
        ranks_host[ridx] = C.ranks[ridx]
        resid_host[ridx] = C.resid[ridx]
    end
    return C
end

# Write one merged output tile back into C: clear its factor slots, copy the first
# `rank` columns of the new factors in, and record the rank and residual (from the
# squared error `resid_sq`). Shared by the batched (beta==0) and per-tile paths.
@inline function _scatter_tile!(C::TLRMatrix{<:Any,T}, slot::Int, ridx::Int, rank::Int,
                                Qsrc, Vsrc, resid_sq, ranks_host, resid_host) where {T}
    fill!(view(C.int_U, :, :, slot), zero(T)); fill!(view(C.int_V, :, :, slot), zero(T))
    copyto!(view(C.int_U, :, 1:rank, slot), Qsrc)
    copyto!(view(C.int_V, :, 1:rank, slot), Vsrc)
    ranks_host[ridx] = eltype(C.ranks)(rank)
    resid_host[ridx] = sqrt(max(resid_sq, 0.0))
    return nothing
end

function _row_basis_gemm!(C::TLRMatrix{BackendT,T},
                             A::TLRMatrix{BackendT,T},
                             B::TLRMatrix{BackendT,T};
                             alpha::T=one(T), beta::T=zero(T), tol::Real=0.0,
                             rel::Bool=false, compute=default_gemm_compute_mode(T),
                             max_workspace::Int=DEFAULT_GEMM_BUDGET,
                             sat_threshold::Real=0.5) where {BackendT,T}
    qm, K = regular_tilegrid_size(A)
    _, qn = regular_tilegrid_size(B)
    b = size(A.int_U, 1)          # output row tile size (bm)
    bn = size(C.int_V, 1)         # output column tile size
    rA, rB = A.maxrank, B.maxrank
    S_full = min(b, K * rA)
    eps_basis = tol == 0 ? zero(Float64) : Float64(tol) / 4
    eps_sq = Float64(tol)^2
    ApU = interior_operand(FullGrid(), A.int_U, A.order, qm, K)
    ApV = interior_operand(FullGrid(), A.int_V, A.order, qm, K)
    BpU = interior_operand(FullGrid(), B.int_U, B.order, K, qn)
    BpV = interior_operand(FullGrid(), B.int_V, B.order, K, qn)

    ranks_host = Array(C.ranks)
    resid_host = Array(C.resid)
    iszero(beta) && begin
        fill!(C.int_U, zero(T)); fill!(C.int_V, zero(T));
        fill!(ranks_host, zero(eltype(C.ranks))); fill!(resid_host, 0.0)
    end

    # On a regular grid every row shares the panel shape (b × K*rA), so the sketch
    # buffers and the row-basis workspace are row-independent: allocate them once
    # and reuse across all rows. `build_row_basis!` overwrites all of its fields.
    # Saturation guard: build the M4 fallback context only when a row could
    # actually saturate (`t ≤ S_full`, so `S_full < θ·b` rules it out a priori)
    # and the fallback applies (beta == 0; row-family layout — checked inside).
    m4 = iszero(beta) && S_full >= sat_threshold * b ?
         _m4_row_context(C, A, B, max_workspace) : nothing
    # With the fallback armed, rows needing t ≥ θ·b are routed to dense anyway,
    # so the sketch (and every downstream buffer, which scales with t) is capped
    # at the threshold; hitting the cap is the saturation signal, and the basis
    # build is told (`tguard`) to return early at it rather than finish a basis
    # the guard will discard.
    S = m4 === nothing ? S_full : min(S_full, ceil(Int, sat_threshold * b))
    tguard = m4 === nothing ? typemax(Int) : ceil(Int, sat_threshold * b)

    backend = get_backend(C.int_U)
    omega = allocate(backend, T, K * rA, S)
    # The random test matrix is independent of the output row.  Reuse one draw
    # for the whole GEMM call: every row still builds its own basis from its own
    # Ubar, while avoiding an RNG launch/fill per row.
    randn!(omega)
    gamma = allocate(backend, T, K); fill!(gamma, one(T))
    basis_ws = RowBasisWorkspace(reshape(_packed_row_panel(ApU, 1), b, K * rA), S)
    basis_err_host = Vector{Float64}(undef, 1)
    sat_streak = 0

    for i in 1:qm
        if m4 !== nothing && sat_streak >= SAT_STREAK_CUTOFF
            _m4_row!(C, m4, i, alpha, compute, ranks_host, resid_host; eps_sq, rel)
            continue
        end
        Urow = _packed_row_panel(ApU, i)
        Vrow = _packed_row_panel(ApV, i)
        Ubar = reshape(Urow, b, K * rA)
        basis = build_row_basis!(basis_ws, Ubar, omega, gamma;
                                 eps_basis=eps_basis, tmax=S, tguard, compute=compute)
        t = basis.t
        if m4 !== nothing && t >= sat_threshold * b
            sat_streak += 1
            _m4_row!(C, m4, i, alpha, compute, ranks_host, resid_host; eps_sq, rel)
            continue
        end
        sat_streak = 0
        # Basis truncation error for this row (nonzero only when eps_basis > 0);
        # folded into every tile residual below as a diagnostic upper-add so
        # `residuals(C)` does not silently under-report the shared-basis error.
        basis_err_sq = if eps_basis > 0 && t > 0
            copyto!(basis_err_host, basis.residual_sq)
            @inbounds basis_err_host[1]
        else
            0.0
        end
        if t == 0
            if !iszero(beta)
                @inbounds for j in 1:qn
                    slot = tile_linear_index(C.order, qm, qn, i, j)
                    ridx = _rank_index(C, i, j)
                    rC = Int(ranks_host[ridx])
                    view(C.int_V, :, 1:rC, slot) .*= beta
                end
            end
            continue
        end
        # `basis.P` is a row-subset view of the sketch buffer, hence strided when
        # t < S. Host BLAS tolerates that, but CUBLAS needs a contiguous operand for
        # the batched coefficient GEMMs, so compact it into a dense t × (K*rA) block.
        Pc = allocate(backend, T, t, K * rA)
        copyto!(Pc, basis.P)
        Pblocks = reshape(Pc, t, rA, K)
        if iszero(beta) || C.maxrank == 0
            # Batched row: coefficients for every output column at once (one set of
            # batched GEMMs), then a single batched prune since every tile shares the
            # left basis Q. This removes both the per-tile coefficient calls and the
            # per-tile merge synchronizations that otherwise dominate the GPU cost.
            # (A zero-capacity C stores nothing to fold, so beta != 0 degenerates
            # to the same path.)
            Qm = allocate(backend, T, b, t, qn)
            Vm = allocate(backend, T, bn, t, qn)
            rvec = allocate(backend, Int32, qn)
            evec = allocate(backend, Float64, qn)
            _accumulate_row_block!(Vm, Vrow, Pblocks, BpU, BpV, qn, alpha, compute)
            Qm .= reshape(basis.Q, b, t, 1)              # broadcast the shared basis into every slab
            fill!(evec, 0.0)
            prune_orthogonal_columns!(Qm, Vm, rvec, evec, t, min(C.maxrank, t), eps_sq, rel)
            rvec_h = Array(rvec); evec_h = Array(evec)
            @inbounds for j in 1:qn
                slot = tile_linear_index(C.order, qm, qn, i, j)
                rank = Int(rvec_h[j])
                _scatter_tile!(C, slot, _rank_index(C, i, j), rank,
                               view(Qm, :, 1:rank, j), view(Vm, :, 1:rank, j),
                               evec_h[j] + basis_err_sq, ranks_host, resid_host)
            end
        else
            # beta != 0: coefficients from the same batched Stage 2 as beta == 0
            # (alpha folded there), then the whole row merges in one batched pass
            # (C2a): old factors enter at the uniform `maxrank` width — padded
            # columns are zero by the container invariant, so full-width batching
            # plus one uniform-width prune is algebraically equivalent to the
            # per-tile merge, with zero device→host reads in the merge body.
            Vm = allocate(backend, T, bn, t, qn)
            _accumulate_row_block!(Vm, Vrow, Pblocks, BpU, BpV, qn, alpha, compute)
            rcap = C.maxrank
            rvec = allocate(backend, Int32, qn)
            evec = allocate(backend, Float64, qn)
            slots = [tile_linear_index(C.order, qm, qn, i, j) for j in 1:qn]
            mws = BatchedMergeWorkspace(basis.Q, rcap, bn, qn)
            merge_row_block!(mws, basis.Q, Vm,
                             [view(C.int_U, :, 1:rcap, s) for s in slots],
                             [view(C.int_V, :, 1:rcap, s) for s in slots],
                             beta, eps_sq, rel, C.maxrank, rvec, evec, compute)
            rvec_h = Array(rvec); evec_h = Array(evec)
            @inbounds for j in 1:qn
                rank = Int(rvec_h[j])
                _scatter_tile!(C, slots[j], _rank_index(C, i, j), rank,
                               view(mws.Qmerge, :, 1:rank, j),
                               view(mws.Vmerge, :, 1:rank, j),
                               evec_h[j] + basis_err_sq, ranks_host, resid_host)
            end
        end
    end
    copyto!(C.ranks, ranks_host)
    copyto!(C.resid, resid_host)
    return C
end
