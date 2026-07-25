# Standalone numerical Stage 1 for the global row basis. This intentionally
# owns its fused sequence; it is not a composition of generic sink callbacks.

@kernel function _row_basis_scale_copy_blocks!(dest, src, gamma, rA::Int)
    row, col = @index(Global, NTuple)
    k = (col - 1) ÷ rA + 1
    @inbounds dest[row, col] = eltype(dest)(src[row, col]) * eltype(dest)(gamma[k])
end

@kernel function _row_basis_copy_descending_eigenvectors!(dest, src, t::Int)
    row, col = @index(Global, NTuple)
    @inbounds dest[row, col] = eltype(dest)(src[row, size(src, 2) - col + 1])
end

"""Workspace for one standalone global-row-basis construction."""
struct RowBasisWorkspace{FA,HA,QT,PT,ET,OT,HT,KT,RT,YT,GT,RWT,MT,QVT,RVT,EV}
    factor_arena::FA
    high_arena::HA
    Q::QT                 # b × S × 1: sketch Y, then Q0 and final Q
    Pfull::PT             # S × (K*rA): unweighted projection, then compact P
    Ebar::ET              # b × (K*rA): rotate scratch, then explicit residual
    omega::OT             # weighted copy of caller-provided Ω
    Pweighted::HT         # promoted, block-weighted copy of Pfull
    Ksmall::KT            # promoted covariance, overwritten by eigenvectors
    R::RT                 # storage-precision descending eigenvectors
    Yhi::YT
    Ghi::GT
    Rwork::RWT
    multipliers::MT
    Qtiles::QVT
    Rtiles::RVT
    residual_sq::EV
end

"""
    RowBasisWorkspace(Ubar, S)

Allocate reusable scratch for `build_row_basis!`, where `Ubar` is a contiguous
`b × (K*rA)` row panel.  `S` is the sketch capacity.  The caller owns the
random sketch and k-weights, allowing deterministic tests and avoiding hidden
RNG policy in the hot path.
"""
function RowBasisWorkspace(Ubar::AbstractMatrix{T}, S::Int) where {T}
    b, width = size(Ubar)
    0 <= S <= min(b, width) ||
        throw(ArgumentError("S must satisfy 0 <= S <= min(size(Ubar)...)"))
    backend = get_backend(Ubar)
    Thi = tlr_orthogonalization_type(T)
    factor_count = b * S + S * width + b * width + width * S + S * S + S * S
    high_count = S * width + S * S + b * S + S * S + 1
    factor_arena = allocate(backend, T, factor_count)
    high_arena = allocate(backend, Thi, high_count)
    off = 0
    Q, off = _take(factor_arena, off, b, S, 1)
    Pfull, off = _take(factor_arena, off, S, width)
    Ebar, off = _take(factor_arena, off, b, width)
    omega, off = _take(factor_arena, off, width, S)
    R, off = _take(factor_arena, off, S, S)
    Rwork, off = _take(factor_arena, off, S, S, 1)
    hoff = 0
    Pweighted, hoff = _take(high_arena, hoff, S, width)
    Ksmall, hoff = _take(high_arena, hoff, S, S)
    Yhi, hoff = _take(high_arena, hoff, b, S, 1)
    Ghi, hoff = _take(high_arena, hoff, S, S, 1)
    multipliers, hoff = _take(high_arena, hoff, 1)
    residual_sq = allocate(backend, Float64, 1)
    return RowBasisWorkspace(
        factor_arena, high_arena, Q, Pfull, Ebar, omega, Pweighted, Ksmall, R, Yhi, Ghi, Rwork,
        multipliers, _batch_views(Q), _batch_views(Rwork), residual_sq,
    )
end

"""Backend-specialized symmetric eigensolve; eigenvectors overwrite `A`."""
function _row_basis_eigh!(A::StridedMatrix{T}) where {T<:Union{Float32,Float64}}
    return LinearAlgebra.LAPACK.syevd!('V', 'U', A)
end

@inline function _row_basis_choose_rank(values, eps_basis::Real, tmax::Int)
    isempty(values) && return 0
    total = sum(values)
    total <= 0 && return 0
    limit = min(tmax, length(values))
    tail = total
    threshold = Float64(eps_basis)^2 * total
    @inbounds for t in 1:limit
        tail -= values[end - t + 1]
        tail <= threshold && return t
    end
    return limit
end

"""
    build_row_basis!(ws, Ubar, omega, gamma; eps_basis, tmax, compute)
        -> (; Q, P, t, residual_sq)

Build a shared orthogonal left basis for the already-contiguous row panel
`Ubar = [U[i,1] … U[i,K]]`. `omega` has shape `(K*rA, S)` and `gamma` has one
weight per k block.  Returned `P` is unweighted, so importance weights choose
directions but do not rescale the represented factors.
"""
function build_row_basis!(ws::RowBasisWorkspace, Ubar::AbstractMatrix{T},
                          omega::AbstractMatrix{T}, gamma::AbstractVector{T};
                          eps_basis::Real, tmax::Int=size(ws.Q, 2),
                          compute=default_gemm_compute_mode(T)) where {T}
    b, width = size(Ubar)
    S = size(ws.Q, 2)
    size(ws.Q, 1) == b && size(ws.Pfull) == (S, width) &&
        size(ws.Ebar) == (b, width) || throw(DimensionMismatch("workspace does not match Ubar"))
    size(omega) == (width, S) || throw(DimensionMismatch("omega must have size ($width, $S)"))
    isempty(gamma) && width != 0 && throw(DimensionMismatch("gamma must partition Ubar columns"))
    !isempty(gamma) && width % length(gamma) != 0 &&
        throw(DimensionMismatch("gamma must partition Ubar columns"))
    0 <= tmax <= S || throw(ArgumentError("tmax must satisfy 0 <= tmax <= S"))
    eps_basis >= 0 || throw(ArgumentError("eps_basis must be nonnegative"))
    S == 0 && return (; Q=view(ws.Q, :, 1:0, 1), P=view(ws.Pfull, 1:0, :),
                       t=0, residual_sq=ws.residual_sq)

    rA = div(width, length(gamma))
    copyto!(ws.omega, omega)
    backend = get_backend(Ubar)
    _row_basis_scale_copy_blocks!(backend)(ws.omega, ws.omega, gamma, rA;
                                    ndrange=size(ws.omega))

    Q0 = view(ws.Q, :, :, 1)
    precision_gemm!('N', 'N', one(T), Ubar, ws.omega, zero(T), Q0, compute)
    mixed_cholqr2_basis!(ws.Q, ws.Yhi, ws.Ghi, ws.Rwork, ws.Rtiles,
                          ws.Qtiles, ws.multipliers)

    adj = _adjoint_blas_char(T)
    precision_gemm!(adj, 'N', one(T), Q0, Ubar, zero(T), ws.Pfull, compute)

    _row_basis_scale_copy_blocks!(backend)(ws.Pweighted, ws.Pfull, gamma, rA;
                                    ndrange=size(ws.Pweighted))
    Thi = eltype(ws.Ksmall)
    high_compute = default_gemm_compute_mode(Thi)
    precision_gemm!('N', _adjoint_blas_char(Thi), one(Thi), ws.Pweighted,
                    ws.Pweighted, zero(Thi), ws.Ksmall, high_compute)

    values, vectors = _row_basis_eigh!(ws.Ksmall)
    values_host = Array(values)
    t = _row_basis_choose_rank(values_host, eps_basis, tmax)
    if t == 0
        fill!(ws.Q, zero(T)); fill!(ws.Pfull, zero(T)); fill!(ws.residual_sq, 0.0)
        return (; Q=view(ws.Q, :, 1:0, 1), P=view(ws.Pfull, 1:0, :),
                t=0, residual_sq=ws.residual_sq)
    end
    vectors === ws.Ksmall || copyto!(ws.Ksmall, vectors)
    Rsel = view(ws.R, :, 1:t)
    _row_basis_copy_descending_eigenvectors!(backend)(Rsel, ws.Ksmall, t;
                                                ndrange=size(Rsel))

    Qtmp = view(ws.Ebar, :, 1:t)
    precision_gemm!('N', 'N', one(T), Q0, Rsel, zero(T), Qtmp, compute)
    copyto!(view(ws.Q, :, 1:t, 1), Qtmp)
    Ptmp = view(ws.Ebar, 1:t, :)
    precision_gemm!(adj, 'N', one(T), Rsel, ws.Pfull, zero(T), Ptmp, compute)
    copyto!(view(ws.Pfull, 1:t, :), Ptmp)

    copyto!(ws.Ebar, Ubar)
    precision_gemm!('N', 'N', -one(T), view(ws.Q, :, 1:t, 1),
                    view(ws.Pfull, 1:t, :), one(T), ws.Ebar, compute)
    batch_frobenius_norms_sq!(ws.residual_sq, reshape(ws.Ebar, b, width, 1))
    return (; Q=view(ws.Q, :, 1:t, 1), P=view(ws.Pfull, 1:t, :),
            t, residual_sq=ws.residual_sq)
end
