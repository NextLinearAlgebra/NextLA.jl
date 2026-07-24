# One-tile orthogonal merge. The residual split keeps the shared row basis intact and
# gives the final pruning kernel genuinely orthogonal coordinates.

struct OrthogonalMergeWorkspace{QT,VT,DT,UT,RT,VTT,CT,IV,EV}
    Qmerge::QT
    Vmerge::VT
    D::DT
    D2::DT
    Dtmp::DT
    Ures::UT
    V0::RT
    Vtmp::VTT
    chol::CT
    ranks::IV
    error_sq::EV
    rank_host::Vector{Int32}   # reusable host mirror of `ranks[1]`
    err_host::Vector{Float64}  # reusable host mirror of `error_sq[1]`
end

function OrthogonalMergeWorkspace(Q::AbstractMatrix{T}, U::AbstractMatrix{T};
                                 bn::Int=size(Q, 1)) where {T}
    # `bm` is the output row tile size (left factor Q/U); `bn` is the output column
    # tile size (right factor M/V). They differ on a rectangular grid.
    bm, t = size(Q)
    size(U, 1) == bm || throw(DimensionMismatch("Q and U must have equal heights"))
    rC = size(U, 2)
    p = t + rC
    backend = get_backend(Q)
    Qmerge = allocate(backend, T, bm, p, 1)
    Vmerge = allocate(backend, T, bn, p, 1)
    D = allocate(backend, T, t, rC)
    D2 = allocate(backend, T, t, rC)
    Dtmp = allocate(backend, T, t, rC)
    Ures = allocate(backend, T, bm, rC, 1)
    V0 = allocate(backend, T, rC, rC, 1)
    Vtmp = allocate(backend, T, bn, rC)
    Thi = tlr_orthogonalization_type(T)
    Yhi = allocate(backend, Thi, bm, rC, 1)
    Ghi = allocate(backend, Thi, rC, rC, 1)
    R1 = allocate(backend, T, rC, rC, 1)
    R2 = allocate(backend, T, rC, rC, 1)
    multipliers = allocate(backend, real(Thi), 1)
    Vchol = allocate(backend, T, rC, rC, 1)
    chol = CholQR2FactorWorkspace(Ures, Vchol, Yhi, Ghi, R1, R2, multipliers)
    ranks = allocate(backend, Int32, 1)
    error_sq = allocate(backend, Float64, 1)
    return OrthogonalMergeWorkspace(Qmerge, Vmerge, D, D2, Dtmp, Ures, V0, Vtmp,
                            chol, ranks, error_sq, Vector{Int32}(undef, 1),
                            Vector{Float64}(undef, 1))
end

# Transport a device scalar to the host without a per-call allocation: `copyto!`
# is a bulk copy (single D2H on CUDA, memcpy on CPU) and never scalar-indexes.
@inline function _merge_scalar_rank(ws::OrthogonalMergeWorkspace)
    copyto!(ws.rank_host, ws.ranks)
    return Int(@inbounds ws.rank_host[1])
end

@inline function _merge_error_sq(ws::OrthogonalMergeWorkspace)
    copyto!(ws.err_host, ws.error_sq)
    return @inbounds ws.err_host[1]
end

"""
    merge_row_basis_tile!(ws, Q, M, U, V; alpha, beta, eps_sq, rel, maxrank, compute)

Represent `alpha*Q*M' + beta*U*V'` in an orthogonal left coordinate system,
then prune once. The output lives in `ws.Qmerge`/`ws.Vmerge`; callers scatter
the first returned `rank` columns into their TLR tile storage.
"""
function merge_row_basis_tile!(ws::OrthogonalMergeWorkspace,
                               Q::AbstractMatrix{T}, M::AbstractMatrix{T},
                               U::AbstractMatrix{T}, V::AbstractMatrix{T};
                               alpha::T=one(T), beta::T=zero(T),
                               eps_sq::Float64=0.0, rel::Bool=false,
                               maxrank::Int=size(ws.Qmerge, 2),
                               compute=default_gemm_compute_mode(T)) where {T}
    bm, t = size(Q)
    bn = size(ws.Vmerge, 1)
    rC = size(U, 2)
    size(M) == (bn, t) || throw(DimensionMismatch("M must be bn × t"))
    size(U, 1) == bm || throw(DimensionMismatch("Q and U must have equal heights"))
    size(V) == (bn, rC) || throw(DimensionMismatch("V must be bn × rC"))
    size(ws.Qmerge, 1) == bm && size(ws.Qmerge, 2) >= t + rC ||
        throw(DimensionMismatch("merge workspace is too small"))
    maxrank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))

    Qout = ws.Qmerge
    Vout = ws.Vmerge
    fill!(Qout, zero(T)); fill!(Vout, zero(T))
    # Every exit prunes the `active` merged columns, reads back the rank, and
    # returns the kept factor views. `maxrank` is the caller's output-rank cap; the
    # merged system has only `active` columns, so the effective cap is their min.
    _finish!(active) = begin
        fill!(ws.error_sq, 0.0)
        prune_orthogonal_columns!(Qout, Vout, ws.ranks, ws.error_sq,
                                  active, min(maxrank, active), eps_sq, rel)
        rank = _merge_scalar_rank(ws)
        (; Q=view(Qout, :, 1:rank, 1), V=view(Vout, :, 1:rank, 1), rank)
    end

    if t == 0
        iszero(beta) && return (; Q=view(Qout, :, 1:0, 1), V=view(Vout, :, 1:0, 1), rank=0)
        copyto!(view(Qout, :, 1:rC, 1), U)
        copyto!(view(Vout, :, 1:rC, 1), V)
        return _finish!(rC)
    end

    copyto!(view(Qout, :, 1:t, 1), Q)
    copyto!(view(Vout, :, 1:t, 1), M)
    Vout[:, 1:t, 1] .*= alpha
    (iszero(beta) || rC == 0) && return _finish!(t)

    adj = _adjoint_blas_char(T)
    precision_gemm!(adj, 'N', one(T), Q, U, zero(T), ws.D, compute)
    copyto!(view(ws.Ures, :, :, 1), U)
    precision_gemm!('N', 'N', -one(T), Q, ws.D, one(T), view(ws.Ures, :, :, 1), compute)

    rank_tol = cholqr_rank_rtol_sq(T, tlr_orthogonalization_type(T), bm, rC)
    fill!(ws.error_sq, 0.0)
    mixed_cholqr2_compress!(ws.chol, ws.ranks, ws.error_sq, rC, rank_tol)
    rho0 = _merge_scalar_rank(ws)
    if rho0 == 0
        precision_gemm!('N', adj, beta, V, ws.D, one(T), view(Vout, :, 1:t, 1), compute)
        return _finish!(t)
    end

    copyto!(ws.V0, ws.chol.V)
    Qres0 = view(ws.Ures, :, :, 1)
    precision_gemm!(adj, 'N', one(T), Q, Qres0, zero(T), ws.D2, compute)
    precision_gemm!('N', 'N', -one(T), Q, ws.D2, one(T), Qres0, compute)
    fill!(ws.error_sq, 0.0)
    mixed_cholqr2_compress!(ws.chol, ws.ranks, ws.error_sq, rC, rank_tol)
    rho = _merge_scalar_rank(ws)

    # U = Q*(D + D2*V0') + Qres*(V1'*V0').
    precision_gemm!('N', adj, one(T), ws.D2, view(ws.V0, :, :, 1), zero(T), ws.Dtmp, compute)
    ws.D .+= ws.Dtmp
    precision_gemm!('N', 'N', one(T), V, view(ws.V0, :, :, 1), zero(T), ws.Vtmp, compute)
    precision_gemm!('N', 'N', beta, ws.Vtmp, view(ws.chol.V, :, :, 1),
                    zero(T), view(Vout, :, (t + 1):(t + rC), 1), compute)
    precision_gemm!('N', adj, beta, V, ws.D, one(T), view(Vout, :, 1:t, 1), compute)
    rho > 0 && copyto!(view(Qout, :, (t + 1):(t + rho), 1),
                        view(ws.Ures, :, 1:rho, 1))

    return _finish!(t + rho)
end
