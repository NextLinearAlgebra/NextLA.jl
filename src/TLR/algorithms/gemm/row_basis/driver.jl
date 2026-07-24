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
                             rel::Bool=false, compute=default_gemm_compute_mode(T)) where {BackendT,T}
    qm, K = regular_tilegrid_size(A)
    _, qn = regular_tilegrid_size(B)
    b = size(A.int_U, 1)          # output row tile size (bm)
    bn = size(C.int_V, 1)         # output column tile size
    rA, rB = A.maxrank, B.maxrank
    S = min(b, K * rA)
    eps_basis = tol == 0 ? zero(Float64) : Float64(tol) / 4
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
    backend = get_backend(C.int_U)
    omega = allocate(backend, T, K * rA, S)
    gamma = allocate(backend, T, K); fill!(gamma, one(T))
    basis_ws = RowBasisWorkspace(reshape(_packed_row_panel(ApU, 1), b, K * rA), S)

    for i in 1:qm
        Urow = _packed_row_panel(ApU, i)
        Vrow = _packed_row_panel(ApV, i)
        Ubar = reshape(Urow, b, K * rA)
        randn!(omega)
        basis = build_row_basis!(basis_ws, Ubar, omega, gamma;
                                 eps_basis=eps_basis, tmax=S, compute=compute)
        t = basis.t
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
        if iszero(beta)
            # Batched row: coefficients for every output column at once (one set of
            # batched GEMMs), then a single batched prune since every tile shares the
            # left basis Q. This removes both the per-tile coefficient calls and the
            # per-tile merge synchronizations that otherwise dominate the GPU cost.
            Qm = allocate(backend, T, b, t, qn)
            Vm = allocate(backend, T, bn, t, qn)
            rvec = allocate(backend, Int32, qn)
            evec = allocate(backend, Float64, qn)
            _accumulate_row_block!(Vm, Vrow, Pblocks, BpU, BpV, qn, alpha, compute)
            Qm .= reshape(basis.Q, b, t, 1)              # broadcast the shared basis into every slab
            fill!(evec, 0.0)
            prune_orthogonal_columns!(Qm, Vm, rvec, evec, t, min(C.maxrank, t), Float64(tol)^2, rel)
            rvec_h = Array(rvec); evec_h = Array(evec)
            @inbounds for j in 1:qn
                slot = tile_linear_index(C.order, qm, qn, i, j)
                rank = Int(rvec_h[j])
                _scatter_tile!(C, slot, _rank_index(C, i, j), rank,
                               view(Qm, :, 1:rank, j), view(Vm, :, 1:rank, j),
                               evec_h[j], ranks_host, resid_host)
            end
        else
            # beta != 0: each tile folds its existing factor through a residual
            # CholQR2, whose rank is data-dependent, so the tiles are merged one at
            # a time. Batching this path is future work.
            coeff_ws = CoefficientWorkspace(Vrow, Pblocks, _packed_column_panel(BpU, 1), t; q=K, bn=bn)
            for j in 1:qn
                Wcol = _packed_column_panel(BpU, j)
                Zcol = _packed_column_panel(BpV, j)
                M = accumulate_row_coefficients!(coeff_ws, Vrow, Pblocks, Wcol, Zcol; compute)
                slot = tile_linear_index(C.order, qm, qn, i, j)
                ridx = _rank_index(C, i, j)
                rC = Int(ranks_host[ridx])
                Uold = view(C.int_U, :, 1:rC, slot)
                Vold = view(C.int_V, :, 1:rC, slot)
                merge_ws = OrthogonalMergeWorkspace(basis.Q, Uold; bn=bn)
                merged = merge_row_basis_tile!(merge_ws, basis.Q, M, Uold, Vold;
                                               alpha, beta, eps_sq=Float64(tol)^2,
                                               rel, maxrank=C.maxrank, compute)
                _scatter_tile!(C, slot, ridx, merged.rank, merged.Q, merged.V,
                               _merge_error_sq(merge_ws), ranks_host, resid_host)
            end
        end
    end
    copyto!(C.ranks, ranks_host)
    copyto!(C.resid, resid_host)
    return C
end
