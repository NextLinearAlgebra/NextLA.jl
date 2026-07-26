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
                                 count::Int=length(state.samples),
)
    count == 0 && return 0
    0 <= count <= length(state.samples) ||
        throw(ArgumentError("count must lie in 0:$(length(state.samples))"))
    size(dR, 2) >= count ||
        throw(DimensionMismatch("dR must have at least $count columns"))
    block <= size(dR, 1) ||
        throw(ArgumentError("block exceeds the rows of dR"))
    eps_rel >= 0 || throw(ArgumentError("eps_rel must be nonnegative"))
    r_required >= 1 || throw(ArgumentError("r_required must be positive"))
    _ara_convergence_kernel!(get_backend(dR))(
        state.samples, state.ranks, state.svec, state.jcount, state.rmax, dR,
        Float64(eps_rel), r_required, block, maxrank; ndrange=count,
    )
    copyto!(state.samples_host, state.samples)
    return Base.count(>(Int32(0)), view(state.samples_host, 1:count))
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
    kcut::IV               # per-member first numerically invalid column
    kcut_host::Vector{Int32}
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
    return ARAWorkspace(allocate(backend, T, m, maxrank, count); block)
end

"""
    ARAWorkspace(Q; block=32)

Wrap a caller-owned basis panel `Q` (`m × maxrank × count`), so the loop can
write straight into storage the caller already has — `compress!` keeps its basis
inside the output factor panels this way. Everything else is allocated on `Q`'s
backend.
"""
function ARAWorkspace(Q::AbstractArray{T,3}; block::Int=32) where {T}
    block >= 1 || throw(ArgumentError("block must be positive"))
    m, maxrank, count = size(Q)
    backend = get_backend(Q)
    blk = min(block, max(maxrank, 1))
    Thi = tlr_orthogonalization_type(T)
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

        # Block Gram-Schmidt with one reorthogonalization, INTERLEAVED with the
        # two Cholesky-QR passes: `(P·CholQR)²`, exactly as Algorithm 2.3 lines
        # 22-27 (the Gram/potrf/trsm sit inside the `for i = 1:2` loop).
        #
        # The interleaving is load-bearing, not stylistic. `P²·CholQR²` -- both
        # projections, then both normalizations -- is a different algorithm in
        # finite precision. A column that is numerically dependent on the rest
        # of the block has a within-block residual of size `O(u‖Y‖)`, the same
        # order as the `O(u‖Y‖)` overlap with `Q` that BCGS leaves behind; so
        # that residual's direction has an `O(1)` *fraction* lying in `span(Q)`.
        # The first Cholesky normalizes it to unit length and bakes that
        # fraction in at `O(1)`. Interleaved, the second projection runs *after*
        # that amplification and removes it; batched together, both projections
        # happen before it and nothing ever does.
        #
        # Measured on a rank-6 operator at block width 4: `P²CholQR²` produced
        # `‖QᵀQ−I‖ > 1e-10` on 97/300 seeds (worst 0.998); `(P·CholQR)²` on
        # 0/300 (worst 3.7e-12). See docs/TODO.md worklog item 9.
        Qc = grown > 0 ? view(ws.Q, :, 1:grown, :) : nothing
        D = grown > 0 ? view(ws.Dproj, 1:grown, :, :) : nothing
        fill!(ws.status, Int32(0))
        for pass in 1:2
            if Qc !== nothing
                precision_gemm_batched!(adj, 'N', one(T), Qc, ws.Yblk,
                                        zero(T), D, mode)
                precision_gemm_batched!('N', 'N', -one(T), Qc, D,
                                        one(T), ws.Yblk, mode)
            end
            if pass == 1
                # The scale the stopping test is relative to, captured before
                # the Gram overwrites the panel.
                ara_column_norms_sq!(ws.cn, ws.colmax, ws.Yblk, width)
                # Unshifted: `potrf` breakdown is the rank signal, not an error.
                mixed_cholqr_pass!(ws.chol, ws.chol.R1, ws.chol.R1_tiles;
                                   shift_coeff=0, escalate=false,
                                   status=ws.status)
            else
                mixed_cholqr_pass!(ws.chol, ws.chol.R2, ws.chol.R2_tiles;
                                   shift_coeff=0, escalate=false)
            end
        end
        copyto!(ws.status_host, ws.status)

        ara_block_norms!(ws.dR, ws.chol, ws.colmax; shift_coeff=0)
        ara_mask_breakdown!(ws.Yblk, ws.dR, ws.kcut, ws.chol.R1, ws.status,
                            width, floor_rel)
        copyto!(ws.kcut_host, ws.kcut)

        view(ws.Q, :, (grown + 1):(grown + width), :) .=
            view(ws.Yblk, :, 1:width, :)
        grown += width
        passes += 1

        ara_update_convergence!(ws.state, ws.dR, eps_rel, r_required,
                                width, maxrank)
        active = _ara_retire_broken!(ws.state, ws.kcut_host, width)
        active == 0 && break
    end

    return (; Q=view(ws.Q, :, 1:grown, :), ranks=ws.state.ranks, passes)
end

# Where a block stops being numerically usable.
#
# Two things can end it, and `potrf`'s status alone is not enough:
#
#   * `potrf` reports `info = k` when the leading minor of order `k` is not
#     positive definite, so columns `1..k-1` were validly factored.
#   * `potrf` can also *succeed* on a pivot that is positive but far too small
#     to divide by. Measured on a rank-4 Float32 panel of width 12: status 0,
#     yet pivots 5–12 sat at `~5e-7` against a leading `6.6`. The triangular
#     solve then produces garbage (or, on an exactly zero panel, `NaN` from
#     `0/0`).
#
# The second cut is the CholeskyQR2 validity condition itself, applied per
# column: the `O(u)` orthogonality guarantee holds for `κ(Y) ≤ u^{-1/2}`, and
# `κ(R) = κ(Y)`, so a pivot below `√u · R_max` is outside the regime where the
# factorization means anything. It is the same `√u` that bounds the stopping
# tolerance — not a separate constant.
#
# Everything from the cut onward is zeroed. That is not over-eager: `R` is upper
# triangular, so column `j` of `Y R⁻¹` is built from the leading `j×j` block,
# and once a pivot is invalid every later column inherits it. The zeros then
# read as "no new content", which is what they are.
# The cut must be read off the **first** CholQR pass. `dR` is the composite
# `R₂R₁` diagonal, and once pass 1's triangular solve has produced garbage
# columns, pass 2's Gram is computed from them — so a contaminated `R₂` can drag
# the composite below threshold on columns that were perfectly valid. Measured:
# a rank-20 operator sampled at width 8 gave `potrf` status 5 (four genuine new
# directions) while the composite-based cut fired at column 1, losing them.
@kernel function _ara_cut_kernel!(kcut, R1::AbstractArray{T,3}, status,
                                  width::Int, floor_rel::Float64) where {T}
    b = @index(Global, Linear)
    @inbounds begin
        mx = 0.0
        for j in 1:width
            d = sqrt(_abs2_f64(R1[j, j, b]))
            (isfinite(d) && d > mx) && (mx = d)
        end
        k = width + 1
        info = Int(status[b])
        (info > 0 && info < k) && (k = info)
        thresh = floor_rel * mx
        for j in 1:width
            d = sqrt(_abs2_f64(R1[j, j, b]))
            if !isfinite(d) || d < thresh
                (j < k) && (k = j)
                break
            end
        end
        kcut[b] = Int32(k)
    end
end

@kernel function _ara_mask_breakdown_kernel!(Y::AbstractArray{T,3}, dR, kcut,
                                             width::Int) where {T}
    j, b = @index(Global, NTuple)
    @inbounds if j >= Int(kcut[b]) && j <= width
        for i in axes(Y, 1)
            Y[i, j, b] = zero(T)
        end
        dR[j, b] = 0.0
    end
end

"""
    ara_mask_breakdown!(Y, dR, kcut, R1, status, width, floor_rel) -> kcut

Discard the columns of a block that the orthonormalization could not deliver,
keeping its valid prefix. `R1` is the **first**-pass triangular factor and
`floor_rel` the CholeskyQR2 validity threshold `√u_hi`; see the commentary above
for why `potrf`'s status alone is insufficient and why the composite factor
cannot be used here.
"""
function ara_mask_breakdown!(Y, dR, kcut, R1, status, width::Int, floor_rel::Float64)
    width == 0 && return kcut
    backend = get_backend(Y)
    count = size(Y, 3)
    _ara_cut_kernel!(backend)(kcut, R1, status, width, floor_rel; ndrange=count)
    _ara_mask_breakdown_kernel!(backend)(Y, dR, kcut, width; ndrange=(width, count))
    return kcut
end

# Breakdown means the block of `width` fresh Gaussian samples produced only
# `k-1 < width` independent directions. For `Y_Δ = (I-QQᵀ)XΩ_Δ` with Gaussian
# `Ω_Δ`, the samples are in general position, so `rank(Y_Δ) = rank((I-QQᵀ)X)`
# almost surely: the residual range has dimension `k-1` and this pass has just
# captured all of it. The member is therefore converged, and forcing it is not a
# heuristic but the conclusion of that argument. Runs *after* the convergence
# update so the rank from this pass is recorded first.
function _ara_retire_broken!(state::ARAConvergenceState, kcut_host, width::Int)
    any(k -> k <= width, kcut_host) || return count_active(state)
    samples = state.samples_host
    copyto!(samples, state.samples)
    @inbounds for b in eachindex(kcut_host)
        (kcut_host[b] <= width) && (samples[b] = Int32(0))
    end
    copyto!(state.samples, samples)
    return count_active(state)
end

# Swap two batch members in the persistent basis and convergence state. Packed
# ARA keeps live members in `1:nactive`; a retiring member is exchanged with the
# last live slot, leaving a dense active prefix and a retired suffix. The
# accumulated basis is moved once, at retirement, rather than inactive members
# occupying every later batched factorization.
@kernel function _ara_swap_basis_kernel!(Q, p::Int, q::Int)
    i, j = @index(Global, NTuple)
    @inbounds begin
        x = Q[i, j, p]
        Q[i, j, p] = Q[i, j, q]
        Q[i, j, q] = x
    end
end

@kernel function _ara_swap_state_kernel!(samples, ranks, svec, jcount, rmax,
                                         p::Int, q::Int)
    _ = @index(Global, Linear)
    @inbounds begin
        x = samples[p]; samples[p] = samples[q]; samples[q] = x
        x = ranks[p]; ranks[p] = ranks[q]; ranks[q] = x
        x = svec[p]; svec[p] = svec[q]; svec[q] = x
        x = jcount[p]; jcount[p] = jcount[q]; jcount[q] = x
        y = rmax[p]; rmax[p] = rmax[q]; rmax[q] = y
    end
end

function _ara_swap_members!(ws::ARAWorkspace, p::Int, q::Int)
    p == q && return ws
    backend = get_backend(ws.Q)
    _ara_swap_basis_kernel!(backend)(ws.Q, p, q;
                                     ndrange=(size(ws.Q, 1), size(ws.Q, 2)))
    _ara_swap_state_kernel!(backend)(
        ws.state.samples, ws.state.ranks, ws.state.svec, ws.state.jcount,
        ws.state.rmax, p, q; ndrange=(1,),
    )
    ws.state.samples_host[p], ws.state.samples_host[q] =
        ws.state.samples_host[q], ws.state.samples_host[p]
    return ws
end

"""
    ara_build_basis_packed!(ws, sample_right!, member_ids;
                            eps_rel, r_required=10, compute,
                            swap_member! = (p,q)->nothing)

Packed-active variant of [`ara_build_basis!`](@ref). `member_ids[slot]` records
which logical operator occupies a physical batch slot. After every pass,
converged members are swap-removed into a retired suffix, so all subsequent
GEMM/SYRK/POTRF/TRSM calls operate on the contiguous prefix `1:nactive`.

`sample_right!(Y, width, active_ids)` overwrites the active prefix of `Y`.
`swap_member!(p,q)` lets an implicit-operator implementation apply the same
slot permutation to retained run-local data such as coupling matrices.

The returned `member_ids` is the final slot-to-logical-member permutation.
"""
function ara_build_basis_packed!(ws::ARAWorkspace, sample_right!,
                                 member_ids::Vector{Int};
                                 eps_rel::Real,
                                 r_required::Int=10,
                                 compute=nothing,
                                 swap_member!::Function=(p, q) -> nothing)
    T = eltype(ws.Q)
    Thi = tlr_orthogonalization_type(T)
    maxrank, count = size(ws.Q, 2), size(ws.Q, 3)
    length(member_ids) == count ||
        throw(DimensionMismatch("member_ids must have length $count"))
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    blk = ws.block

    eps_rel > 0 || throw(ArgumentError("eps_rel must be positive"))
    floor_rel = ara_stopping_floor(Thi)
    eps_rel >= floor_rel || throw(ArgumentError(
        "eps_rel = $eps_rel is below the CholeskyQR2 orthogonality limit " *
        "√u = $floor_rel for Gram type $Thi"))
    (count == 0 || maxrank == 0) &&
        return (; Q=view(ws.Q, :, 1:0, :), ranks=ws.state.ranks,
                passes=0, member_ids, active_counts=Int[])

    adj = _adjoint_blas_char(T)
    ara_reset!(ws.state, blk, maxrank)
    fill!(ws.Q, zero(T))
    passes = 0
    grown = 0
    nactive = count
    active_counts = Int[]

    while grown < maxrank && nactive > 0
        push!(active_counts, nactive)
        width = min(blk, maxrank - grown)
        Y = view(ws.Yblk, :, :, 1:nactive)
        sample_right!(Y, width, view(member_ids, 1:nactive))
        width < blk && fill!(view(Y, :, (width + 1):blk, :), zero(T))

        Qc = grown > 0 ? view(ws.Q, :, 1:grown, 1:nactive) : nothing
        D = grown > 0 ? view(ws.Dproj, 1:grown, :, 1:nactive) : nothing
        chol = CholQR2FactorWorkspace(
            Y,
            view(ws.chol.V, :, :, 1:nactive),
            view(ws.chol.Y_hi, :, :, 1:nactive),
            view(ws.chol.G_hi, :, :, 1:nactive),
            view(ws.chol.R1, :, :, 1:nactive),
            view(ws.chol.R2, :, :, 1:nactive),
            view(ws.chol.multipliers, 1:nactive),
        )
        status = view(ws.status, 1:nactive)
        fill!(status, Int32(0))
        for pass in 1:2
            if Qc !== nothing
                precision_gemm_batched!(adj, 'N', one(T), Qc, Y,
                                        zero(T), D, mode)
                precision_gemm_batched!('N', 'N', -one(T), Qc, D,
                                        one(T), Y, mode)
            end
            if pass == 1
                ara_column_norms_sq!(
                    view(ws.cn, :, 1:nactive), view(ws.colmax, 1:nactive),
                    Y, width,
                )
                mixed_cholqr_pass!(chol, chol.R1, chol.R1_tiles;
                                   shift_coeff=0, escalate=false, status)
            else
                mixed_cholqr_pass!(chol, chol.R2, chol.R2_tiles;
                                   shift_coeff=0, escalate=false)
            end
        end
        copyto!(ws.status_host, ws.status)

        dR = view(ws.dR, :, 1:nactive)
        ara_block_norms!(dR, chol, view(ws.colmax, 1:nactive); shift_coeff=0)
        kcut = view(ws.kcut, 1:nactive)
        ara_mask_breakdown!(Y, dR, kcut, chol.R1, status, width, floor_rel)
        copyto!(ws.kcut_host, ws.kcut)

        view(ws.Q, :, (grown + 1):(grown + width), 1:nactive) .=
            view(Y, :, 1:width, :)
        grown += width
        passes += 1

        ara_update_convergence!(ws.state, ws.dR, eps_rel, r_required,
                                width, maxrank, nactive)
        _ara_retire_broken!(ws.state, view(ws.kcut_host, 1:nactive), width)

        # Partition in place: active prefix, retired suffix. Targets are chosen
        # from the shrinking tail, so swap pairs are disjoint within a pass.
        p = 1
        while p <= nactive
            if ws.state.samples_host[p] > 0
                p += 1
                continue
            end
            while nactive > p && ws.state.samples_host[nactive] == 0
                nactive -= 1
            end
            if p < nactive
                _ara_swap_members!(ws, p, nactive)
                swap_member!(p, nactive)
                member_ids[p], member_ids[nactive] =
                    member_ids[nactive], member_ids[p]
            end
            nactive -= 1
        end
    end

    return (; Q=view(ws.Q, :, 1:grown, :), ranks=ws.state.ranks,
            passes, member_ids, active_counts)
end

# ---------------------------------------------------------------------------
# A2: optimal truncation of X ≈ Q Zᵀ.
# ---------------------------------------------------------------------------
#
# With `Q` orthonormal and `Z = P Σ Wᵀ` the thin SVD of `Z`,
#
#     Q Zᵀ = Q W Σ Pᵀ = (Q W) Σ Pᵀ ,
#
# and `QW` has orthonormal columns, so this *is* the SVD of the represented
# matrix: its singular values are those of `Z`. Truncating at rank `r` is
# therefore optimal in the Eckart–Young sense, with squared error `Σ_{k>r} σ_k²`
# known exactly and for free. The outputs are `U = Q·W[:,1:r]` and
# `V = P[:,1:r]·diag(σ_{1:r})` — the latter a column scaling, not a product.
#
# Two invariants make this safe when `Q` carries zero columns (which it does:
# `ara_mask_breakdown!` zeroes the tail of a broken-down block).
#
#   * Column `j` of `Z = XᵀQ` is zero whenever column `j` of `Q` is. Then
#     `Z[:,j] = P Σ W[j,:]ᵀ = 0` and, since `P` has orthonormal columns,
#     `σ_k W[j,k] = 0` for every `k`. So the retained right singular vectors
#     (those with `σ_k > 0`) vanish on exactly the rows that index zero columns
#     of `Q`. Hence `UᵀU = W[:,1:r]ᵀ (QᵀQ) W[:,1:r] = I` even though `QᵀQ ≠ I`.
#   * Members truncate to different ranks. The batch runs at the row-wise
#     maximum with `W`'s tail zeroed per member, so every product stays a single
#     uniform batched GEMM and the surplus columns come out exactly zero.

"""
    batched_thin_svd!(A) -> (U, S, V)

Thin SVD `A[:,:,b] = U[:,:,b] * Diagonal(S[:,b]) * V[:,:,b]'` of every member of
a batch of tall-skinny matrices, with `size(A,1) ≥ size(A,2)`. `A` may be
overwritten.

This is the one backend hook of the truncation step. The generic method loops
LAPACK on the host and is correct for every array type; backends override it:

  * **CUDA** — `gesvdaStridedBatched`. Measured on this repository's panels it
    is 3.4× faster than a Gram plus `syevjBatched` at `s = 64` and recovers the
    true rank where the Gram route over-retains, because it never squares the
    condition number. Its *left* factor is not orthonormal when returned
    untruncated, but the caller consumes the *right* factor, which measures
    clean at `~1e-14`.
  * **AMD** — `rocsolver_?gesvdj_strided_batched`, the batched one-sided Jacobi
    SVD, which is the closest rocSOLVER equivalent.
"""
function batched_thin_svd!(A::AbstractArray{T,3}) where {T}
    m, n, count = size(A)
    m >= n || throw(ArgumentError("batched_thin_svd! requires size(A,1) >= size(A,2)"))
    RT = real(T)
    Ah = A isa Array ? A : Array(A)
    U = Array{T,3}(undef, m, n, count)
    S = Array{RT,2}(undef, n, count)
    V = Array{T,3}(undef, n, n, count)
    @inbounds for b in 1:count
        F = svd!(view(Ah, :, :, b); full=false)
        copyto!(view(U, :, :, b), F.U)
        copyto!(view(S, :, b), F.S)
        copyto!(view(V, :, :, b), F.V)
    end
    A isa Array && return (U, S, V)
    dev = similar(A, T, 0)                       # backend-typed constructor
    return (_to_backend(dev, U), _to_backend(dev, S), _to_backend(dev, V))
end

@inline function _to_backend(proto, H::Array)
    D = similar(proto, eltype(H), size(H))
    copyto!(D, H)
    return D
end

# Per member: the smallest `r` whose discarded energy fits the budget, capped at
# `maxrank`. Scanning from the tail accumulates the discarded energy directly,
# so the reported error is the achieved one, not a bound.
#
# When `has_energy`, `energy[b] = ‖X‖²_F` of the *original* operator is supplied
# and the range-capture error is charged against the budget too. `Q` is
# orthonormal, so `‖QQᵀX‖² = ‖Z‖² = Σ_k σ_k²` and Pythagoras gives
# `‖X − QQᵀX‖² = ‖X‖² − Σ_k σ_k²` exactly; the total error of the returned
# factors is then `‖X‖² − Σ_{k≤r} σ_k²`. This is the randQB_EI indicator, and it
# carries its cancellation: both terms are `O(‖X‖²)` while the difference may be
# far smaller, so it is clamped to zero below the rounding floor of the
# subtraction — otherwise noise there is charged as real error and the rank
# is inflated.
@kernel function _ara_truncation_rank_kernel!(ranks, err_sq, S, energy, tol_sq,
                                              relative::Bool, has_energy::Bool,
                                              maxrank::Int, width::Int, rows::Int)
    b = @index(Global, Linear)
    @inbounds begin
        total = 0.0
        for k in 1:width
            total += Float64(S[k, b])^2
        end
        reference = has_energy ? Float64(energy[b]) : total
        capture = 0.0
        if has_energy
            capture = reference - total
            floor_sq = Float64(rows) * Float64(eps(Float64)) * reference
            capture = (capture < floor_sq) ? 0.0 : capture
        end
        budget = (relative ? Float64(tol_sq) * reference : Float64(tol_sq)) - capture
        dropped = 0.0
        r = width
        while r > 0
            e = Float64(S[r, b])^2
            dropped + e > budget && break
            dropped += e
            r -= 1
        end
        if r > maxrank                     # capacity is authoritative
            for k in (maxrank + 1):r
                dropped += Float64(S[k, b])^2
            end
            r = maxrank
        end
        ranks[b] = eltype(ranks)(r)
        err_sq[b] = capture + dropped
    end
end

# Zero `W[:, r_b+1 : width]` and scale `V[:, k] = P[:, k]·σ_k` (zero past `r_b`),
# so the subsequent GEMM can run at one uniform width for the whole batch.
@kernel function _ara_apply_truncation_kernel!(W::AbstractArray{T,3},
                                               V::AbstractArray{T,3},
                                               P::AbstractArray{T,3},
                                               S, ranks, width::Int) where {T}
    k, b = @index(Global, NTuple)
    @inbounds begin
        keep = k <= Int(ranks[b])
        for i in axes(W, 1)
            W[i, k, b] = keep ? W[i, k, b] : zero(T)
        end
        scale = keep ? T(S[k, b]) : zero(T)
        for i in axes(V, 1)
            V[i, k, b] = P[i, k, b] * scale
        end
    end
end

"""
    ara_truncate!(U, V, ranks, err_sq, Q, Z; tol, relative=true, maxrank)
        -> (; U, V, ranks, err_sq)

Optimally truncate `X ≈ Q Zᵀ` to per-member rank, writing `U` (`b_m × maxrank ×
count`, orthonormal columns) and `V` (`b_n × maxrank × count`).

`tol` is the Frobenius error budget, squared internally; with `relative=true` it
is taken against each member's reference energy. `err_sq[b]` returns the
**achieved** squared error, and `ranks[b] == maxrank` flags a member whose
spectrum did not fit — no separate residual pass is needed to report either (see
`docs/TODO.md`, worklog item 4).

`energy[b] = ‖X‖_F²` of the original operator may be supplied when it is known
independently, as it is when compressing an explicit matrix. The range-capture
error `‖X‖² − ‖Z‖²` is then charged against the budget as well, so `err_sq` is
the total error of the returned factors rather than the truncation part alone.
Without it the reference is `‖Z‖²` and capture is assumed zero, which is correct
when `Q` was built to tolerance by [`ara_build_basis!`](@ref).

`Z` is overwritten.
"""
function ara_truncate!(U, V, ranks, err_sq, Q, Z;
                       tol::Real, relative::Bool=true, maxrank::Int,
                       energy=nothing, compute=nothing)
    T = eltype(Q)
    bm, sQ, count = size(Q)
    bn = size(Z, 1)
    size(Z) == (bn, sQ, count) ||
        throw(DimensionMismatch("Z must be ($bn, $sQ, $count)"))
    size(U) == (bm, maxrank, count) ||
        throw(DimensionMismatch("U must be ($bm, $maxrank, $count)"))
    size(V) == (bn, maxrank, count) ||
        throw(DimensionMismatch("V must be ($bn, $maxrank, $count)"))
    maxrank <= sQ ||
        throw(ArgumentError("maxrank must not exceed the basis width $sQ"))
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    count == 0 && return (; U, V, ranks, err_sq)

    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    backend = get_backend(Q)

    P, S, W = batched_thin_svd!(Z)
    _ara_truncation_rank_kernel!(backend)(
        ranks, err_sq, S, energy === nothing ? err_sq : energy,
        Float64(tol)^2, relative, energy !== nothing, maxrank, sQ, bn;
        ndrange=count,
    )
    _ara_apply_truncation_kernel!(backend)(
        W, V, P, S, ranks, maxrank; ndrange=(maxrank, count),
    )
    # U = Q · W[:, 1:maxrank]; columns past each member's rank are zero because
    # the kernel above zeroed the matching columns of W.
    precision_gemm_batched!('N', 'N', one(T), Q, view(W, :, 1:maxrank, :),
                            zero(T), U, mode)
    return (; U, V, ranks, err_sq)
end
