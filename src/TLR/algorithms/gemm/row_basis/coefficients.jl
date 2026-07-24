# Row-basis coefficient accumulation. The final Zstack GEMM is deliberately
# explicit: it is the large, contiguous terminal operation for column-major B.

struct CoefficientWorkspace{TT,ST,RT,MT,BT}
    Tpanel::TT
    Sbuf::ST
    Rstack::RT
    M::MT
    batches::BT
    q::Int
end

"""
    CoefficientWorkspace(Vrow, Pblocks, Wcol, t; q=size(Vrow,3))

Allocate reusable Stage-2 storage for one row and one output-column factor
panel.  `Vrow`, `Wcol`, and `Zcol` are `b × rank × K` views; for the preferred
layout they are zero-copy row/column panels of A and B respectively.
"""
function CoefficientWorkspace(Vrow::AbstractArray{T,3},
                                Pblocks::AbstractArray{T,3},
                                Wcol::AbstractArray{T,3}, t::Int;
                                q::Int=size(Vrow, 3),
                                bn::Int=size(Wcol, 1)) where {T}
    # `kd` is the contraction tile size (rows of V_A and W_B); `bn` is the output
    # column tile size (rows of Z_B and of the coefficient M). They differ on a
    # rectangular grid, so they are tracked separately.
    kd, rA, K = size(Vrow)
    size(Pblocks) == (t, rA, K) ||
        throw(DimensionMismatch("Pblocks must have size ($t, $rA, $K)"))
    size(Wcol, 1) == kd && size(Wcol, 3) == K ||
        throw(DimensionMismatch("Wcol must be kd × rB × K"))
    1 <= q <= K || throw(ArgumentError("q must be in 1:K"))
    rB = size(Wcol, 2)
    backend = get_backend(Vrow)
    Tpanel = t <= rA ? allocate(backend, T, kd, t, q) : nothing
    Sbuf = t <= rA ? nothing : allocate(backend, T, rA, rB, q)
    Rstack = allocate(backend, T, q * rB, t)
    M = allocate(backend, T, bn, t)
    sample_v = view(Vrow, :, :, 1)
    sample_p = view(Pblocks, :, :, 1)
    sample_w = view(Wcol, :, :, 1)
    sample_r = view(Rstack, 1:rB, :)
    batches = t <= rA ? (
        vp = _batchvec(sample_v, q), pp = _batchvec(sample_p, q),
        tp = _batchvec(view(Tpanel, :, :, 1), q),
        wt = _batchvec(sample_w, q), rt = _batchvec(sample_r, q),
    ) : (
        vs = _batchvec(sample_v, q), ws = _batchvec(sample_w, q),
        ss = _batchvec(view(Sbuf, :, :, 1), q),
        sp = _batchvec(sample_p, q), rp = _batchvec(sample_r, q),
    )
    return CoefficientWorkspace(Tpanel, Sbuf, Rstack, M, batches, q)
end

@inline function _clear_coefficient_batches!(batches)
    foreach(empty!, values(batches))
    return nothing
end

"""
    accumulate_row_coefficients!(ws, Vrow, Pblocks, Wcol, Zcol; compute) -> ws.M

Compute `M = Σₖ Z[k] (P[k] V[k]' W[k])'` without dense tile products.  The
`t <= rA` branch first forms the reusable compressed right factor `V*P'`; the
other branch retains `S = V'W` only until it is consumed.  `Zcol` is stacked in
one terminal GEMM per k panel and is zero-copy for logical column-major B.
"""
function accumulate_row_coefficients!(ws::CoefficientWorkspace,
                                      Vrow::AbstractArray{T,3},
                                      Pblocks::AbstractArray{T,3},
                                      Wcol::AbstractArray{T,3},
                                      Zcol::AbstractArray{T,3};
                                      compute=default_gemm_compute_mode(T)) where {T}
    kd, rA, K = size(Vrow)
    bn = size(Zcol, 1)
    t = size(Pblocks, 1)
    rB = size(Wcol, 2)
    size(Pblocks) == (t, rA, K) || throw(DimensionMismatch("incompatible Pblocks"))
    size(Wcol) == (kd, rB, K) ||
        throw(DimensionMismatch("Wcol must be kd × rB × K"))
    size(Zcol) == (bn, rB, K) ||
        throw(DimensionMismatch("Zcol must be bn × rB × K"))
    size(ws.M) == (bn, t) || throw(DimensionMismatch("workspace M does not match inputs"))
    fill!(ws.M, zero(T))
    backend = get_backend(Vrow)
    adj = _adjoint_blas_char(T)
    first_panel = true

    for k0 in 1:ws.q:K
        len = min(ws.q, K - k0 + 1)
        krange = k0:(k0 + len - 1)
        Rsub = view(ws.Rstack, 1:(len * rB), :)
        _clear_coefficient_batches!(ws.batches)
        if t <= rA
            @inbounds for (slot, k) in enumerate(krange)
                push!(ws.batches.vp, view(Vrow, :, :, k))
                push!(ws.batches.pp, view(Pblocks, :, :, k))
                push!(ws.batches.tp, view(ws.Tpanel, :, :, slot))
            end
            precision_gemm_batched!('N', adj, one(T), ws.batches.vp, ws.batches.pp,
                                    zero(T), ws.batches.tp, compute)
            @inbounds for slot in 1:len
                push!(ws.batches.wt, view(Wcol, :, :, k0 + slot - 1))
                push!(ws.batches.rt, view(Rsub, (slot - 1) * rB + 1:slot * rB, :))
            end
            precision_gemm_batched!(adj, 'N', one(T), ws.batches.wt, ws.batches.tp,
                                    zero(T), ws.batches.rt, compute)
        else
            @inbounds for (slot, k) in enumerate(krange)
                push!(ws.batches.vs, view(Vrow, :, :, k))
                push!(ws.batches.ws, view(Wcol, :, :, k))
                push!(ws.batches.ss, view(ws.Sbuf, :, :, slot))
            end
            precision_gemm_batched!(adj, 'N', one(T), ws.batches.vs, ws.batches.ws,
                                    zero(T), ws.batches.ss, compute)
            @inbounds for slot in 1:len
                push!(ws.batches.sp, view(Pblocks, :, :, k0 + slot - 1))
                push!(ws.batches.rp, view(Rsub, (slot - 1) * rB + 1:slot * rB, :))
            end
            precision_gemm_batched!(adj, adj, one(T), ws.batches.ss, ws.batches.sp,
                                    zero(T), ws.batches.rp, compute)
        end
        Zstack = reshape(view(Zcol, :, :, krange), bn, len * rB)
        precision_gemm!('N', 'N', one(T), Zstack, Rsub,
                        first_panel ? zero(T) : one(T), ws.M, compute)
        first_panel = false
    end
    return ws.M
end
