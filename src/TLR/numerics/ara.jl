# Support kernels for the blocked adaptive randomized approximation (ARA) of
# Boukaram, Turkiyyah & Keyes, "Randomized GPU algorithms for the construction
# of hierarchical matrices from matrix-vector operations", SIAM J. Sci. Comput.
# 41(4):C339–C366, 2019 (Algorithm 2.3).
#
# The loop samples a block, orthogonalizes it against the existing basis with
# BCGS2, orthonormalizes it with mixed-precision CholQR2, and stops once
# `r_required` consecutive projected columns have negligible norm. Two pieces of
# bookkeeping make that possible and live here:
#
#   * `ara_block_norms!` recovers the projected column norms from the triangular
#     factors that CholQR2 already produced (`dR` in the reference), and
#   * `ara_update_convergence!` maintains, per batch member, the running maximum
#     norm, the consecutive-small-vector count, the detected rank, and how many
#     columns that member should sample next.
#
# Why the diagonal and not the column norms of the projected block: at
# convergence the block is a set of random combinations of the few directions
# that remain, so every *column* still has O(1) norm while the *block* is
# rank-deficient. `R[j,j]` is the residual of column `j` against both the basis
# and the earlier columns of the same block, so it collapses exactly when there
# is nothing new left. Column norms alone would miss the generic stopping case.

"""
    cholqr2_relative_shift_floor(Tgram, m, s) -> Float64

Smallest value `R[j,j] / max_j‖y_j‖` that shifted CholQR2 can produce.

`_shifted_cholesky!` adds `coeff · max_i G[i,i] · multiplier` to the Gram
diagonal, and `max_i G[i,i]` is the largest squared column norm of the panel, so
every triangular diagonal entry is bounded below by `√(coeff·multiplier)`
*relative to that column norm* — independent of the data.

This is a hard floor under any rank or convergence test built on `diag(R)`: a
relative tolerance below it can never be met, and a stopping loop would run to
`maxrank` on every tile with no error raised. At `m=256, s=32` in `Float64` it
is `3.4e-6` — well inside the range callers ask for.

The ARA loop therefore factors **unshifted**, which removes the floor entirely
and extends the usable range to `√u ≈ 1.05e-8` (see [`ara_stopping_floor`](@ref)).
This function remains the guard for paths that do shift, such as the wide
one-shot panels in the compression code. The value assumes `multiplier == 1`;
escalation raises the true floor further.
"""
@inline cholqr2_relative_shift_floor(::Type{Tgram}, m::Int, s::Int) where {Tgram} =
    sqrt(Float64(_cholqr_shift_coeff(Tgram, m, s)))

# `dR[j,b] = √(max(R1[j,j]² − δ_b, 0)) · |R2[j,j]|`, the shift-corrected diagonal
# of the composite factor `R₂R₁`. One thread per (column, batch member): no
# barriers, so this is safe on the KA CPU backend.
@kernel function _ara_block_norms_kernel!(dR,
                                          R1::AbstractArray{T,3},
                                          R2::AbstractArray{T,3},
                                          colmax_sq,
                                          coeff::Float64,
) where {T}
    j, b = @index(Global, NTuple)
    @inbounds begin
        delta = coeff * Float64(colmax_sq[b])
        d1_sq = _abs2_f64(R1[j, j, b]) - delta
        d1 = d1_sq > 0.0 ? sqrt(d1_sq) : 0.0
        dR[j, b] = d1 * sqrt(_abs2_f64(R2[j, j, b]))
    end
end

"""
    ara_block_norms!(dR, ws::CholQR2FactorWorkspace, colmax_sq;
                     shift_coeff=nothing) -> dR

Projected column norms of the block just orthonormalized by
[`mixed_cholqr2_factor!`](@ref): `dR[j,b] = |R[j,j]|` for the composite factor
`R₂R₁`, which is the residual of column `j` against both the existing basis and
the earlier columns of the same block.

`shift_coeff` must be the coefficient passed to `mixed_cholqr2_factor!`. The ARA
loop factors unshifted (`0`), so no correction is applied and `dR` is exact. For
a shifted factorization the shift is subtracted, but note the subtraction is
*conservative rather than exact*: the shift also perturbs the off-diagonal
entries, and working the recurrence through gives
`R[j,j]² = R̃[j,j]² + δ(1 + G₁₂²/(G₁₁(G₁₁+δ)) + …) ≥ R̃[j,j]² + δ`, so removing
`δ` leaves `dR` slightly too large — which costs at most an extra pass and never
an early stop.

`colmax_sq[b]` must be the largest squared column norm of member `b`'s panel
*as handed to* CholQR2 (after BCGS2, before the Gram); it reproduces the
`max_i G[i,i]` that scaled the shift, which the factorization has overwritten.
It is ignored when `shift_coeff == 0`.
"""
function ara_block_norms!(dR, ws::CholQR2FactorWorkspace, colmax_sq;
                          shift_coeff=nothing)
    R1 = ws.R1
    s, _, count = size(R1)
    size(dR) == (s, count) ||
        throw(DimensionMismatch("dR must have size ($s, $count)"))
    length(colmax_sq) == count ||
        throw(DimensionMismatch("colmax_sq must have length $count"))
    count == 0 && return dR
    coeff = shift_coeff === nothing ?
            Float64(_cholqr_shift_coeff(eltype(ws.G_hi), size(ws.Q, 1), s)) :
            Float64(shift_coeff)
    _ara_block_norms_kernel!(get_backend(R1))(
        dR, R1, ws.R2, colmax_sq, coeff; ndrange=(s, count),
    )
    return dR
end

# `cn[j,b] = ‖Y[:,j,b]‖²`. One thread walks one column, so the reads are
# contiguous per thread but not coalesced across threads; the panel is narrow
# and this runs once per pass, so it has not been worth a tiled variant yet.
@kernel function _ara_colnorms_sq_kernel!(cn, Y::AbstractArray{T,3}) where {T}
    j, b = @index(Global, NTuple)
    acc = 0.0
    @inbounds for i in axes(Y, 1)
        acc += _abs2_f64(Y[i, j, b])
    end
    @inbounds cn[j, b] = acc
end

@kernel function _ara_colmax_kernel!(colmax_sq, cn, width::Int)
    b = @index(Global, Linear)
    mx = 0.0
    @inbounds for j in 1:width
        v = cn[j, b]
        mx = ifelse(v > mx, v, mx)
    end
    @inbounds colmax_sq[b] = mx
end

"""
    ara_column_norms_sq!(cn, colmax_sq, Y, width) -> (cn, colmax_sq)

Squared column norms of the first `width` columns of each panel in `Y`, and
their per-member maximum.

`colmax_sq` is what [`ara_block_norms!`](@ref) needs to undo the Cholesky
shift, so this must be called on the panel *as handed to* CholQR2 — after
BCGS2, before the Gram, which overwrites it.
"""
function ara_column_norms_sq!(cn, colmax_sq, Y::AbstractArray{T,3}, width::Int) where {T}
    s, count = size(cn)
    count == length(colmax_sq) == size(Y, 3) ||
        throw(DimensionMismatch("cn, colmax_sq and Y disagree on the batch count"))
    0 <= width <= min(s, size(Y, 2)) ||
        throw(ArgumentError("width must satisfy 0 <= width <= $(min(s, size(Y, 2)))"))
    count == 0 && return (cn, colmax_sq)
    backend = get_backend(Y)
    if width > 0
        _ara_colnorms_sq_kernel!(backend)(cn, Y; ndrange=(width, count))
    end
    _ara_colmax_kernel!(backend)(colmax_sq, cn, width; ndrange=count)
    return (cn, colmax_sq)
end

"""
    ARAConvergenceState(backend, count)

Per-batch-member bookkeeping for the ARA loop over `count` members.

`samples[b]` is how many columns member `b` should draw on the next pass and is
the single source of truth for whether it is still active: `0` means converged
(or capped). `ranks[b]` is the index of the last column that carried
significant energy, `svec[b]` the number of consecutive negligible columns seen
so far, `jcount[b]` the columns accumulated, and `rmax[b]` the running maximum
projected norm against which the relative test is made.

Members advance independently, so `jcount` is tracked per member rather than as
a shared loop counter.
"""
struct ARAConvergenceState{IV,FV}
    samples::IV
    ranks::IV
    svec::IV
    jcount::IV
    rmax::FV
    samples_host::Vector{Int32}
end

function ARAConvergenceState(backend, count::Int)
    count >= 0 || throw(ArgumentError("count must be nonnegative"))
    ints() = allocate(backend, Int32, count)
    return ARAConvergenceState(
        ints(), ints(), ints(), ints(), allocate(backend, Float64, count),
        Vector{Int32}(undef, count),
    )
end

"""
    ara_reset!(state, block, maxrank) -> state

Arm every member for a first pass of width `min(block, maxrank)`.
"""
function ara_reset!(state::ARAConvergenceState, block::Int, maxrank::Int)
    fill!(state.samples, Int32(max(min(block, maxrank), 0)))
    fill!(state.ranks, Int32(0))
    fill!(state.svec, Int32(0))
    fill!(state.jcount, Int32(0))
    fill!(state.rmax, 0.0)
    return state
end

# One thread per batch member; the scan over the block is serial within a
# member, which keeps `rmax` a true running maximum (the relative test compares
# each column against the largest norm seen *so far*, as in the reference).
@kernel function _ara_convergence_kernel!(samples, ranks, svec, jcount, rmax,
                                          dR,
                                          eps_rel::Float64,
                                          r_required::Int,
                                          block::Int,
                                          maxrank::Int,
)
    b = @index(Global, Linear)
    @inbounds begin
        drawn = Int(samples[b])
        if drawn > 0
            m = rmax[b]
            sv = Int(svec[b])
            rk = Int(ranks[b])
            j0 = Int(jcount[b])
            for j in 1:min(drawn, block)
                d = dR[j, b]
                m = ifelse(d > m, d, m)
                if d <= eps_rel * m
                    sv += 1
                else
                    sv = 0
                    rk = j0 + j
                end
            end
            j0 += min(drawn, block)
            rmax[b] = m
            svec[b] = Int32(sv)
            ranks[b] = Int32(rk)
            jcount[b] = Int32(j0)
            done = (sv >= r_required) | (j0 >= maxrank)
            samples[b] = done ? Int32(0) : Int32(min(block, maxrank - j0))
        end
    end
end

"""
    ara_update_convergence!(state, dR, eps_rel, r_required, block, maxrank)
        -> active_count

Fold one block of projected norms into `state` and return how many members are
still sampling.

A member stops when `r_required` consecutive columns satisfy
`dR[j] ≤ eps_rel · rmax`, or when it reaches `maxrank`. Members already at
`samples == 0` are skipped, so a converged member costs nothing but its slot in
the batch — packing the active members contiguously is a separate concern.

The returned count comes from a device-to-host copy of `samples`, one small
transfer per pass. That is the loop's only synchronization point.
"""
function ara_update_convergence!(state::ARAConvergenceState,
                                 dR,
                                 eps_rel::Real,
                                 r_required::Int,
                                 block::Int,
                                 maxrank::Int,
)
    count = length(state.samples)
    count == 0 && return 0
    size(dR, 2) == count ||
        throw(DimensionMismatch("dR must have $count columns"))
    block <= size(dR, 1) ||
        throw(ArgumentError("block exceeds the rows of dR"))
    eps_rel >= 0 || throw(ArgumentError("eps_rel must be nonnegative"))
    r_required >= 1 || throw(ArgumentError("r_required must be positive"))
    _ara_convergence_kernel!(get_backend(dR))(
        state.samples, state.ranks, state.svec, state.jcount, state.rmax, dR,
        Float64(eps_rel), r_required, block, maxrank; ndrange=count,
    )
    copyto!(state.samples_host, state.samples)
    return count_active(state)
end

"""Number of members still drawing samples, from the host mirror."""
@inline count_active(state::ARAConvergenceState) =
    count(>(Int32(0)), state.samples_host)

# ---------------------------------------------------------------------------
# The blocked ARA loop (Algorithm 2.3).
# ---------------------------------------------------------------------------

"""
    ara_stopping_floor(Tgram) -> Float64

Smallest relative tolerance the unshifted loop can honour, `√u_hi`.

At the stopping block the projected panel has `κ(Y_Δ) ≈ 1/ε_rel`, and
CholeskyQR2 attains `O(u)` orthogonality only for `κ ≤ u^{-1/2}` (Yamamoto,
Nakatsukasa, Yanagisawa & Fukaya, ETNA 44:306–326, 2015). Equating the two gives
`ε_rel ≥ √u_hi` — about `1.05e-8` for a `Float64` Gram. Boukaram, Turkiyyah &
Keyes report the same limit empirically (§2.3.1: below it, quad precision is
needed to stabilise the Cholesky QR).

This is a property of the orthogonalizer, not a tuning knob: it is the exact
point at which `potrf` also starts to break down, which is why breakdown is a
sound convergence signal rather than a failure.
"""
@inline ara_stopping_floor(::Type{Tgram}) where {Tgram} =
    sqrt(Float64(eps(real(Tgram))) / 2)

"""Scratch for [`ara_build_basis!`](@ref) over a batch of `count` panels."""
struct ARAWorkspace{QT,YT,DT,CT,FM,FV,IV,SV}
    Q::QT                  # m × maxrank × count: the basis, grown in place
    Yblk::YT               # m × block × count: current sample block
    Dproj::DT              # maxrank × block × count: BCGS2 coefficients
    chol::CT               # CholQR2 over `Yblk`
    cn::FM                 # block × count: squared column norms
    colmax::FV             # count: max squared column norm (undoes nothing here,
                           #        but pins the scale the shift would have used)
    dR::FM
    status::IV             # per-member potrf info from the first CholQR pass
    status_host::Vector{Int32}
    state::SV
    block::Int
end

"""
    ARAWorkspace(::Type{T}, backend, m, maxrank, count; block=32)

Allocate the ARA loop's scratch. `block` is the number of columns sampled per
pass; it trades kernel efficiency against oversampling and does not affect
correctness. The reference uses 32 (the warp size).
"""
function ARAWorkspace(::Type{T}, backend, m::Int, maxrank::Int, count::Int;
                      block::Int=32) where {T}
    maxrank >= 0 && count >= 0 && m >= 0 ||
        throw(ArgumentError("m, maxrank and count must be nonnegative"))
    block >= 1 || throw(ArgumentError("block must be positive"))
    blk = min(block, max(maxrank, 1))
    Thi = tlr_orthogonalization_type(T)
    Q = allocate(backend, T, m, maxrank, count)
    Yblk = allocate(backend, T, m, blk, count)
    Dproj = allocate(backend, T, max(maxrank, 1), blk, count)
    chol = CholQR2FactorWorkspace(
        Yblk, allocate(backend, T, blk, blk, count),
        allocate(backend, Thi, m, blk, count),
        allocate(backend, Thi, blk, blk, count),
        allocate(backend, T, blk, blk, count),
        allocate(backend, T, blk, blk, count),
        allocate(backend, real(Thi), count),
    )
    return ARAWorkspace(
        Q, Yblk, Dproj, chol,
        allocate(backend, Float64, blk, count), allocate(backend, Float64, count),
        allocate(backend, Float64, blk, count),
        allocate(backend, Int32, count), Vector{Int32}(undef, count),
        ARAConvergenceState(backend, count), blk,
    )
end

"""
    ara_build_basis!(ws, sample_right!; eps_rel, r_required=10, compute)
        -> (; Q, ranks, passes)

Grow an orthonormal basis for the column space of a batch of implicit operators
to relative tolerance `eps_rel` (Boukaram, Turkiyyah & Keyes, Algorithm 2.3).

`sample_right!(Y, width)` is the black box: it must overwrite the first `width`
columns of the `m × block × count` array `Y` with `Xᵦ Ω` for a freshly drawn
Gaussian `Ω`, independently per batch member. Freshness is required — the
stopping rule is the Halko–Martinsson–Tropp a posteriori bound, whose failure
probability decays like `10^{-r_required}` only for independent samples.

Each pass projects the block against the existing basis with two-pass block
classical Gram–Schmidt, orthonormalizes it with **unshifted** mixed-precision
CholQR2, and folds `diag(R)` into the convergence state. A member stops when
`r_required` consecutive columns are negligible, when `potrf` breaks down (its
block is numerically singular, so nothing new remains), or at `maxrank`.

`eps_rel` below [`ara_stopping_floor`](@ref) is rejected rather than silently
run to `maxrank`.
"""
function ara_build_basis!(ws::ARAWorkspace, sample_right!;
                          eps_rel::Real,
                          r_required::Int=10,
                          compute=nothing,
)
    T = eltype(ws.Q)
    Thi = tlr_orthogonalization_type(T)
    maxrank, count = size(ws.Q, 2), size(ws.Q, 3)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    blk = ws.block

    eps_rel > 0 || throw(ArgumentError("eps_rel must be positive"))
    floor_rel = ara_stopping_floor(Thi)
    eps_rel >= floor_rel || throw(ArgumentError(
        "eps_rel = $eps_rel is below the CholeskyQR2 orthogonality limit " *
        "√u = $floor_rel for Gram type $Thi; the stopping test cannot be met " *
        "and the loop would run to maxrank. Use a coarser tolerance or a " *
        "higher-precision Gram."))
    (count == 0 || maxrank == 0) &&
        return (; Q=view(ws.Q, :, 1:0, :), ranks=ws.state.ranks, passes=0)

    adj = _adjoint_blas_char(T)
    ara_reset!(ws.state, blk, maxrank)
    fill!(ws.Q, zero(T))
    passes = 0
    grown = 0                      # columns of `Q` written so far (batch-uniform)

    while grown < maxrank
        width = min(blk, maxrank - grown)
        sample_right!(ws.Yblk, width)
        width < blk && fill!(view(ws.Yblk, :, (width + 1):blk, :), zero(T))

        # Two-pass block classical Gram-Schmidt against the existing basis.
        if grown > 0
            Qc = view(ws.Q, :, 1:grown, :)
            D = view(ws.Dproj, 1:grown, :, :)
            for _ in 1:2
                precision_gemm_batched!(adj, 'N', one(T), Qc, ws.Yblk,
                                        zero(T), D, mode)
                precision_gemm_batched!('N', 'N', -one(T), Qc, D,
                                        one(T), ws.Yblk, mode)
            end
        end

        # The scale the stopping test is relative to, captured before the Gram
        # overwrites the panel.
        ara_column_norms_sq!(ws.cn, ws.colmax, ws.Yblk, width)

        # Unshifted: `potrf` breakdown is the rank signal, not an error.
        fill!(ws.status, Int32(0))
        mixed_cholqr2_factor!(ws.chol; shift_coeff=0, escalate=false,
                              status=ws.status)
        copyto!(ws.status_host, ws.status)

        ara_block_norms!(ws.dR, ws.chol, ws.colmax; shift_coeff=0)
        ara_mask_breakdown!(ws.Yblk, ws.dR, ws.status, width)

        view(ws.Q, :, (grown + 1):(grown + width), :) .=
            view(ws.Yblk, :, 1:width, :)
        grown += width
        passes += 1

        ara_update_convergence!(ws.state, ws.dR, eps_rel, r_required,
                                width, maxrank)
        active = _ara_retire_broken!(ws.state, ws.status_host)
        active == 0 && break
    end

    return (; Q=view(ws.Q, :, 1:grown, :), ranks=ws.state.ranks, passes)
end

# `potrf` reports `info = k` when the leading minor of order `k` is not positive
# definite, so columns `1..k-1` of the block were validly factored and only
# `k..width` are meaningless. Because `R` is upper triangular, column `j` of
# `Y R⁻¹` depends only on the leading `j×j` block of `R`, so the valid prefix is
# untouched by whatever the failed tail contains — the contamination is
# forward-only. Zero the tail of the basis block and its `dR` entries: the
# zeros then read as "no new content", which is exactly what they are.
@kernel function _ara_mask_breakdown_kernel!(Y::AbstractArray{T,3}, dR, status,
                                             width::Int) where {T}
    j, b = @index(Global, NTuple)
    @inbounds begin
        k = Int(status[b])
        if k > 0 && j >= k && j <= width
            for i in axes(Y, 1)
                Y[i, j, b] = zero(T)
            end
            dR[j, b] = 0.0
        end
    end
end

"""
    ara_mask_breakdown!(Y, dR, status, width) -> nothing

Discard the columns a broken-down `potrf` never factored, keeping its valid
prefix. See [`ara_build_basis!`](@ref) for why breakdown is information.
"""
function ara_mask_breakdown!(Y, dR, status, width::Int)
    width == 0 && return nothing
    _ara_mask_breakdown_kernel!(get_backend(Y))(
        Y, dR, status, width; ndrange=(width, size(Y, 3)),
    )
    return nothing
end

# Breakdown means the block of `width` fresh Gaussian samples produced only
# `k-1 < width` independent directions. For `Y_Δ = (I-QQᵀ)XΩ_Δ` with Gaussian
# `Ω_Δ`, the samples are in general position, so `rank(Y_Δ) = rank((I-QQᵀ)X)`
# almost surely: the residual range has dimension `k-1` and this pass has just
# captured all of it. The member is therefore converged, and forcing it is not a
# heuristic but the conclusion of that argument. Runs *after* the convergence
# update so the rank from this pass is recorded first.
function _ara_retire_broken!(state::ARAConvergenceState, status_host)
    any(!iszero, status_host) || return count_active(state)
    samples = state.samples_host
    copyto!(samples, state.samples)
    @inbounds for b in eachindex(status_host)
        iszero(status_host[b]) || (samples[b] = Int32(0))
    end
    copyto!(state.samples, samples)
    return count_active(state)
end
