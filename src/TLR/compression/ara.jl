# Support kernels for the blocked adaptive randomized approximation (ARA) of
# Boukaram, Turkiyyah & Keyes, "Randomized GPU algorithms for the construction
# of hierarchical matrices from matrix-vector operations", SIAM J. Sci. Comput.
# 41(4):C339–C366, 2019 (Algorithm 2.3).
#
# The loop samples a block, orthogonalizes it against the existing basis with
# BCGS2, normalizes it with two mixed-precision Cholesky-QR passes, and stops once
# `r_required` consecutive projected columns have negligible norm. Two pieces of
# bookkeeping make that possible and live here:
#
#   * `ara_block_norms!` recovers the projected column norms from the two
#     triangular factors (`dR` in the reference), and
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

"""Minimal reusable workspace for ARA's Cholesky-QR passes."""
struct ARACholeskyWorkspace{QT,YT,GT,RT,QViews,RViews}
    Q::QT
    Y_hi::YT
    G_hi::GT
    R1::RT
    R2::RT
    Q_tiles::QViews
    R1_tiles::RViews
    R2_tiles::RViews
end

function ARACholeskyWorkspace(Q::AbstractArray{T,3},
                              Y_hi::AbstractArray{Thi,3},
                              G_hi::AbstractArray{Thi,3},
                              R1::AbstractArray{T,3},
                              R2::AbstractArray{T,3}) where {T,Thi}
    _, width, count = size(Q)
    size(Y_hi) == size(Q) ||
        throw(DimensionMismatch("Y_hi must have the same dimensions as Q"))
    size(G_hi) == (width, width, count) ||
        throw(DimensionMismatch("G_hi must have size ($width, $width, $count)"))
    size(R1) == size(G_hi) == size(R2) ||
        throw(DimensionMismatch("R1 and R2 must match G_hi"))
    return ARACholeskyWorkspace(
        Q, Y_hi, G_hi, R1, R2,
        _batch_views(Q), _batch_views(R1), _batch_views(R2),
    )
end

@kernel function _ara_copy_upper_triangle_kernel!(dest::AbstractArray{T,3},
                                                  src::AbstractArray{Thi,3}) where {T,Thi}
    row, col, batch = @index(Global, NTuple)
    @inbounds dest[row, col, batch] =
        row <= col ? T(src[row, col, batch]) : zero(T)
end

"""
    ara_cholesky_pass!(ws, R, R_tiles; status=nothing, count=size(ws.Q, 3))

Apply one mixed-precision Cholesky-QR pass to the first `count`
members of `ws.Q`. The Gram matrix and Cholesky factorization use the promoted
type; the triangular solve and stored factor use the basis storage type.

ARA deliberately treats a failed Cholesky factorization as a rank signal.
When provided, `status` receives the per-member factorization status.
"""
function ara_cholesky_pass!(ws::ARACholeskyWorkspace, Rdest, Rdest_tiles;
                            status=nothing,
                            count::Int=size(ws.Q, 3))
    total = size(ws.Q, 3)
    0 <= count <= total ||
        throw(ArgumentError("count must lie in 0:$total"))
    status === nothing || length(status) >= count ||
        throw(DimensionMismatch("status must have at least $count entries"))
    count == 0 && return ws

    Q = count == total ? ws.Q : view(ws.Q, :, :, 1:count)
    Y_hi = count == total ? ws.Y_hi : view(ws.Y_hi, :, :, 1:count)
    G_hi = count == total ? ws.G_hi : view(ws.G_hi, :, :, 1:count)
    R = count == total ? Rdest : view(Rdest, :, :, 1:count)
    Q_tiles = count == total ? ws.Q_tiles : view(ws.Q_tiles, 1:count)
    R_tiles = count == total ? Rdest_tiles : view(Rdest_tiles, 1:count)

    Y_hi .= Q
    Thi = eltype(Y_hi)
    gemm_batched!(_adjoint_blas_char(Thi), 'N',
                  one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
    result = potrf_batched!('U', G_hi)
    if status !== nothing && result isa Tuple
        copyto!(view(status, 1:count), result[2])
    end
    _ara_copy_upper_triangle_kernel!(get_backend(Q))(
        R, G_hi; ndrange=size(R),
    )
    trsm_batched!('R', 'U', 'N', 'N', R_tiles, Q_tiles, one(eltype(Q)))
    return ws
end

# Diagonal of the composite factor R₂R₁. One thread per (column, member):
# no barriers, so this is safe on the KernelAbstractions CPU backend.
@kernel function _ara_block_norms_kernel!(dR,
                                          R1::AbstractArray{T,3},
                                          R2::AbstractArray{T,3}) where {T}
    j, b = @index(Global, NTuple)
    @inbounds begin
        dR[j, b] = sqrt(_abs2_f64(R1[j, j, b])) *
                   sqrt(_abs2_f64(R2[j, j, b]))
    end
end

"""
    ara_block_norms!(dR, ws::ARACholeskyWorkspace) -> dR

Projected column norms after two Cholesky-QR passes:
`dR[j,b] = |R₁[j,j]R₂[j,j]|`. This is the residual of column `j` against both
the existing basis and the earlier columns of the same block.
"""
function ara_block_norms!(dR, ws::ARACholeskyWorkspace;
                          count::Int=size(ws.R1, 3))
    s, _, total = size(ws.R1)
    0 <= count <= total ||
        throw(ArgumentError("count must lie in 0:$total"))
    size(dR) == (s, count) ||
        throw(DimensionMismatch("dR must have size ($s, $count)"))
    count == 0 && return dR
    R1 = count == total ? ws.R1 : view(ws.R1, :, :, 1:count)
    R2 = count == total ? ws.R2 : view(ws.R2, :, :, 1:count)
    _ara_block_norms_kernel!(get_backend(R1))(
        dR, R1, R2; ndrange=(s, count),
    )
    return dR
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

function ARAConvergenceState(samples, ranks, svec, jcount, rmax, samples_host)
    count = length(samples)
    all(length(x) == count for x in (ranks, svec, jcount, rmax)) ||
        throw(DimensionMismatch("ARA convergence arrays must have equal length"))
    length(samples_host) == count ||
        throw(DimensionMismatch("ARA samples host mirror must have length $count"))
    return ARAConvergenceState(
        samples, ranks, svec, jcount, rmax, samples_host)
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

Smallest relative tolerance the loop can honour, `√u_hi`.

At the stopping block the projected panel has `κ(Y_Δ) ≈ 1/ε_rel`, and
two-pass Cholesky-QR attains `O(u)` orthogonality only for `κ ≤ u^{-1/2}` (Yamamoto,
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
struct ARAWorkspace{QT,YT,DT,CT,FM,IV,SV}
    Q::QT                  # m × maxrank × count: the basis, grown in place
    Yblk::YT               # m × block × count: current sample block
    Dproj::DT              # maxrank × block × count: BCGS2 coefficients
    chol::CT               # two-pass Cholesky-QR workspace over `Yblk`
    dR::FM
    status::IV             # per-member potrf info from the first Cholesky-QR pass
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
                      block::Int=32, arena=nothing,
                      state_storage=nothing) where {T}
    maxrank >= 0 && count >= 0 && m >= 0 ||
        throw(ArgumentError("m, maxrank and count must be nonnegative"))
    Q = _workspace_array!(
        _run_persistent_t_arena(arena), backend, T, m, maxrank, count)
    return ARAWorkspace(Q; block, arena, state_storage)
end

"""
    ARAWorkspace(Q; block=32, arena=nothing)

Wrap a caller-owned basis panel `Q` (`m × maxrank × count`), so the loop can
write straight into numerical staging supplied by dense compression or GEMM.
Everything else is allocated on `Q`'s
backend, drawn from `arena` when one is supplied (see `ARARunArena` in
`gemm/compressed_accumulation/workspace.jl`) and via plain `allocate` otherwise.
"""
function ARAWorkspace(Q::AbstractArray{T,3}; block::Int=32, arena=nothing,
                      state_storage=nothing) where {T}
    block >= 1 || throw(ArgumentError("block must be positive"))
    m, maxrank, count = size(Q)
    backend = get_backend(Q)
    blk = min(block, max(maxrank, 1))
    Thi = tlr_orthogonalization_type(T)
    tarena = _run_t_arena(arena)
    thi_arena = _run_thi_arena(arena)
    Yblk = _workspace_array!(tarena, backend, T, m, blk, count)
    Dproj = _workspace_array!(tarena, backend, T, max(maxrank, 1), blk, count)
    R1 = _workspace_array!(tarena, backend, T, blk, blk, count)
    R2 = _workspace_array!(tarena, backend, T, blk, blk, count)
    chol = ARACholeskyWorkspace(
        Yblk,
        _workspace_array!(thi_arena, backend, Thi, m, blk, count),
        _workspace_array!(thi_arena, backend, Thi, blk, blk, count),
        R1, R2,
    )
    if state_storage === nothing
        dR = allocate(backend, Float64, blk, count)
        status = allocate(backend, Int32, count)
        status_host = Vector{Int32}(undef, count)
        kcut = allocate(backend, Int32, count)
        kcut_host = Vector{Int32}(undef, count)
        state = ARAConvergenceState(backend, count)
    else
        size(state_storage.dR) == (blk, count) ||
            throw(DimensionMismatch("ARA dR storage must have size ($blk, $count)"))
        dR = state_storage.dR
        status = state_storage.status
        status_host = state_storage.status_host
        kcut = state_storage.kcut
        kcut_host = state_storage.kcut_host
        state = ARAConvergenceState(
            state_storage.samples, state_storage.ranks, state_storage.svec,
            state_storage.jcount, state_storage.rmax,
            state_storage.samples_host)
    end
    return ARAWorkspace(
        Q, Yblk, Dproj, chol, dR, status, status_host, kcut, kcut_host,
        state, blk,
    )
end

"""
    rebind_ara_phase(ws, arena)

Recreate only the phase-backed sampling/orthogonalization views after a rolling
scheduler has reused the phase arena for admission or retirement. Persistent
`Q` and every convergence/status buffer retain their identities and contents.
"""
function rebind_ara_phase(ws::ARAWorkspace, arena)
    _arena_reset_phase!(arena)
    T = eltype(ws.Q)
    backend = get_backend(ws.Q)
    m, maxrank, count = size(ws.Q)
    blk = ws.block
    Thi = tlr_orthogonalization_type(T)
    a = _run_t_arena(arena)
    ahi = _run_thi_arena(arena)
    Yblk = _workspace_array!(a, backend, T, m, blk, count)
    Dproj = _workspace_array!(a, backend, T, max(maxrank, 1), blk, count)
    R1 = _workspace_array!(a, backend, T, blk, blk, count)
    R2 = _workspace_array!(a, backend, T, blk, blk, count)
    chol = ARACholeskyWorkspace(
        Yblk,
        _workspace_array!(ahi, backend, Thi, m, blk, count),
        _workspace_array!(ahi, backend, Thi, blk, blk, count),
        R1, R2,
    )
    return ARAWorkspace(
        ws.Q, Yblk, Dproj, chol, ws.dR, ws.status, ws.status_host,
        ws.kcut, ws.kcut_host, ws.state, ws.block,
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
classical Gram–Schmidt, normalizes it with two mixed-precision
Cholesky-QR passes, and folds `diag(R)` into the convergence state. A member stops when
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
        "eps_rel = $eps_rel is below the Cholesky-QR orthogonality limit " *
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
        # two Cholesky-QR passes: `(P·CQR)²`, exactly as Algorithm 2.3 lines
        # 22-27 (the Gram/potrf/trsm sit inside the `for i = 1:2` loop).
        #
        # The interleaving is load-bearing, not stylistic. `P²·CQR²` -- both
        # projections, then both normalizations -- is a different algorithm in
        # finite precision. A column that is numerically dependent on the rest
        # of the block has a within-block residual of size `O(u‖Y‖)`, the same
        # order as the `O(u‖Y‖)` overlap with `Q` that BCGS leaves behind; so
        # that residual's direction has an `O(1)` *fraction* lying in `span(Q)`.
        # The first Cholesky normalizes it to unit length and bakes that
        # fraction in at `O(1)`. Interleaved, the second projection runs *after*
        # that amplification and removes it; batched together, both projections
        # happen before it and nothing ever does.
        Qc = grown > 0 ? view(ws.Q, :, 1:grown, :) : nothing
        D = grown > 0 ? view(ws.Dproj, 1:grown, :, :) : nothing
        fill!(ws.status, Int32(0))
        @unroll for pass in 1:2
            if Qc !== nothing
                precision_gemm_batched!(adj, 'N', one(T), Qc, ws.Yblk,
                                        zero(T), D, mode)
                precision_gemm_batched!('N', 'N', -one(T), Qc, D,
                                        one(T), ws.Yblk, mode)
            end
            if pass == 1
                ara_cholesky_pass!(ws.chol, ws.chol.R1, ws.chol.R1_tiles;
                                   status=ws.status)
            else
                ara_cholesky_pass!(ws.chol, ws.chol.R2, ws.chol.R2_tiles)
            end
        end
        copyto!(ws.status_host, ws.status)

        ara_block_norms!(ws.dR, ws.chol)
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

@kernel function _ara_append_local_kernel!(Q, Y, samples, jcount)
    i, j, b = @index(Global, NTuple)
    @inbounds begin
        drawn = Int(samples[b])
        if j <= drawn
            Q[i, Int(jcount[b]) + j, b] = Y[i, j, b]
        end
    end
end

@kernel function _ara_reset_slot_kernel!(Q, samples, ranks, svec, jcount, rmax,
                                         firstslot::Int, count::Int,
                                         first_width::Int)
    i, j, slotlocal = @index(Global, NTuple)
    if slotlocal <= count
        slot = firstslot + slotlocal - 1
        @inbounds Q[i, j, slot] = zero(eltype(Q))
        if i == 1 && j == 1
            @inbounds begin
                samples[slot] = Int32(first_width)
                ranks[slot] = Int32(0)
                svec[slot] = Int32(0)
                jcount[slot] = Int32(0)
                rmax[slot] = 0.0
            end
        end
    end
end

"""
    ara_reset_slots!(ws, firstslot, count, maxrank)

Reset a contiguous slot range for rolling admission without disturbing basis
or convergence state retained by older active slots.
"""
function ara_reset_slots!(ws::ARAWorkspace, firstslot::Int, count::Int,
                          maxrank::Int=size(ws.Q, 2))
    count == 0 && return ws
    1 <= firstslot <= size(ws.Q, 3) &&
        firstslot + count - 1 <= size(ws.Q, 3) ||
        throw(BoundsError(ws.Q, (:, :, firstslot:(firstslot + count - 1))))
    first_width = min(ws.block, maxrank)
    _ara_reset_slot_kernel!(get_backend(ws.Q))(
        ws.Q, ws.state.samples, ws.state.ranks, ws.state.svec,
        ws.state.jcount, ws.state.rmax, firstslot, count, first_width;
        ndrange=(size(ws.Q, 1), size(ws.Q, 2), count),
    )
    @inbounds fill!(
        view(ws.state.samples_host, firstslot:(firstslot + count - 1)),
        Int32(first_width),
    )
    return ws
end

function _ara_orthogonalize_group!(ws::ARAWorkspace, slots::UnitRange{Int},
                                   width::Int, projection_width::Int,
                                   mode, floor_rel)
    isempty(slots) && return
    T = eltype(ws.Q)
    adj = _adjoint_blas_char(T)
    Y = view(ws.Yblk, :, 1:width, slots)
    Qc = projection_width > 0 ?
        view(ws.Q, :, 1:projection_width, slots) : nothing
    D = projection_width > 0 ?
        view(ws.Dproj, 1:projection_width, 1:width, slots) : nothing
    status = view(ws.status, slots)
    fill!(status, Int32(0))
    chol = ARACholeskyWorkspace(
        Y,
        view(ws.chol.Y_hi, :, 1:width, slots),
        view(ws.chol.G_hi, 1:width, 1:width, slots),
        view(ws.chol.R1, 1:width, 1:width, slots),
        view(ws.chol.R2, 1:width, 1:width, slots),
    )
    for pass in 1:2
        if Qc !== nothing
            precision_gemm_batched!(adj, 'N', one(T), Qc, Y,
                                    zero(T), D, mode)
            precision_gemm_batched!('N', 'N', -one(T), Qc, D,
                                    one(T), Y, mode)
        end
        if pass == 1
            ara_cholesky_pass!(
                chol, chol.R1, chol.R1_tiles; status, count=length(slots))
        else
            ara_cholesky_pass!(
                chol, chol.R2, chol.R2_tiles; count=length(slots))
        end
    end
    dR = view(ws.dR, 1:width, slots)
    ara_block_norms!(dR, chol)
    kcut = view(ws.kcut, slots)
    ara_mask_breakdown!(Y, dR, kcut, chol.R1, status, width, floor_rel)
    return
end

"""
    ara_packed_pass!(ws, sample_right!, nactive, progress_host; ...)

Execute one complete rolling-ARA pass on the active prefix. Basis progress is
slot-local: projection uses the largest active prefix (younger slots contain
zeros beyond their own progress), and each normalized block is appended at
that slot's `jcount`. The returned retired suffix has already been
swap-compacted through `swap_member!`.
"""
function ara_packed_pass!(ws::ARAWorkspace, sample_right!, nactive::Int,
                          member_ids::Vector{Int}, progress_host::Vector{Int};
                          eps_rel::Real, r_required::Int=10,
                          compute=nothing,
                          swap_member!::Function=(p, q) -> nothing)
    nactive == 0 && return (; nactive=0, retired=1:0, width=0,
                            projection_width=0, discarded=0)
    nactive <= size(ws.Q, 3) ||
        throw(DimensionMismatch("active count exceeds workspace capacity"))
    T = eltype(ws.Q)
    Thi = tlr_orthogonalization_type(T)
    mode = compute === nothing ? default_gemm_compute_mode(T) :
           gemm_compute_mode(compute)
    floor_rel = ara_stopping_floor(Thi)
    eps_rel >= floor_rel || throw(ArgumentError(
        "eps_rel = $eps_rel is below the Cholesky-QR limit $floor_rel"))
    drawn = collect(Int, view(ws.state.samples_host, 1:nactive))
    width = maximum(drawn)
    width > 0 || return (; nactive=0, retired=1:nactive, width=0,
                         projection_width=maximum(view(progress_host, 1:nactive)),
                         discarded=0)
    projection_width = maximum(view(progress_host, 1:nactive))

    # Keep the common full block in front and the final partial block in one
    # suffix. They share the fused full-k sampling launch below, but use
    # separate Cholesky-QR batches so padded terminal columns never enter a
    # factorization.
    nfull = count(==(width), drawn)
    if nfull < nactive
        p, q = 1, nactive
        while p <= nfull
            if drawn[p] == width
                p += 1
                continue
            end
            while q > nfull && drawn[q] != width
                q -= 1
            end
            q <= nfull && break
            _ara_swap_members!(ws, p, q)
            swap_member!(p, q)
            member_ids[p], member_ids[q] = member_ids[q], member_ids[p]
            progress_host[p], progress_host[q] =
                progress_host[q], progress_host[p]
            drawn[p], drawn[q] = drawn[q], drawn[p]
            p += 1
            q -= 1
        end
    end
    Y = view(ws.Yblk, :, :, 1:nactive)
    sample_right!(Y, width, view(member_ids, 1:nactive))
    width < ws.block && fill!(view(Y, :, (width + 1):ws.block, :), zero(T))

    nfull > 0 && _ara_orthogonalize_group!(
        ws, 1:nfull, width, projection_width, mode, floor_rel)
    if nfull < nactive
        terminal_width = drawn[nfull + 1]
        _ara_orthogonalize_group!(
            ws, (nfull + 1):nactive, terminal_width, projection_width,
            mode, floor_rel)
    end
    copyto!(ws.status_host, ws.status)
    copyto!(ws.kcut_host, ws.kcut)

    _ara_append_local_kernel!(get_backend(ws.Q))(
        ws.Q, Y, ws.state.samples, ws.state.jcount;
        ndrange=(size(ws.Q, 1), width, nactive),
    )
    @inbounds for p in 1:nactive
        progress_host[p] += drawn[p]
    end
    discarded = sum(width - d for d in drawn)
    ara_update_convergence!(
        ws.state, ws.dR, eps_rel, r_required, width, size(ws.Q, 2), nactive)

    # Breakdown is terminal only when it occurs inside the member's valid
    # prefix; a terminal member's deliberately discarded tail is irrelevant.
    samples = ws.state.samples_host
    copyto!(samples, ws.state.samples)
    @inbounds for p in 1:nactive
        if ws.kcut_host[p] <= drawn[p]
            samples[p] = Int32(0)
        end
    end
    copyto!(ws.state.samples, samples)

    oldactive = nactive
    p = 1
    while p <= nactive
        if samples[p] > 0
            p += 1
            continue
        end
        while nactive > p && samples[nactive] == 0
            nactive -= 1
        end
        if p < nactive
            _ara_swap_members!(ws, p, nactive)
            swap_member!(p, nactive)
            member_ids[p], member_ids[nactive] =
                member_ids[nactive], member_ids[p]
            progress_host[p], progress_host[nactive] =
                progress_host[nactive], progress_host[p]
        end
        nactive -= 1
    end
    return (; nactive, retired=(nactive + 1):oldactive, width,
            projection_width, discarded)
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
    n == 0 && return (
        similar(A, T, m, 0, count),
        similar(A, RT, 0, count),
        similar(A, T, 0, 0, count),
    )
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
spectrum did not fit — no separate residual pass is needed to report either

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
