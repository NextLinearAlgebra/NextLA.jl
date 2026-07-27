# RangeFind for one output tile (algorithm.tex Algorithm 1) -- the batched ARA
# loop driven through single_tile_coupling.jl's implicit ApplyRight/ApplyLeft,
# with no exact residual on the hot path (docs/TODO.md, worklog item 4).

"""
    range_find_tile!(U, V, ranks, err_sq, ops, i, j;
                     alpha, beta=false, C=nothing,
                     eps_rel, r_required=10, tol, rel=true,
                     block=32, compute=nothing) -> (; passes)

Compress one output tile `X_ij = α Σ_ℓ A_iℓB_ℓj + βC_ij` into the low-rank pair
`(U, V)`, `U'U = I`, `X_ij ≈ U V'`, without ever forming `X_ij` as a dense tile.

`U` is `b_m × maxrank × 1`, `V` is `b_n × maxrank × 1`, `ranks`/`err_sq` length
`1` -- `maxrank = size(U,2)` is the tile's *storage* capacity (`maxrank(C)` in
the caller's container), the hard cap on the returned rank. Columns beyond the
achieved rank are zeroed, matching the format's zero-pad invariant.

Three phases, matching algorithm.tex's `RangeFind`:

 1. Grow an orthonormal basis `Q` for `range(X_ij)`, to relative tolerance
    `eps_rel`, via the blocked ARA loop of [`ara_build_basis!`](@ref), sampling
    through [`apply_right!`](@ref). The basis never grows past `maxrank`
    -- there is no point capturing more range than the tile can store.
 2. The co-range `Z = X_ijᵀQ`, one [`apply_left!`](@ref) call.
 3. Optimal (Eckart-Young) truncation of `(Q,Z)` via [`ara_truncate!`](@ref),
to `tol` and the basis's own achieved sketch width -- never to `maxrank` directly,
    since [`ara_truncate!`](@ref) cannot truncate wider than the basis it was
    given (see the note below).

`eps_rel` and `tol` are different knobs answering different questions:
`eps_rel` is how well the *sketch* must capture the range before the loop
stops (bounded below by [`ara_stopping_floor`](@ref)); `tol` is the Frobenius
error the *returned factors* must meet after truncation, and carries no such
floor (truncation is exact, `docs/TODO.md` worklog item 6).

## Why truncation is capped at the achieved basis sketch width, not `maxrank`

If the tile's true rank is below `maxrank`, the ARA loop converges (via the
`r_required`-consecutive-small-columns rule) at some sketch width `s_Q < maxrank`,
and `ara_truncate!` can only select a rank ≤ the basis actually built --
asking it to truncate "up to `maxrank`" when the basis is narrower is not
meaningful. So truncation runs at cap `s_Q`, into a temporary `s_Q`-wide
buffer, which is then copied into the first `s_Q` columns of `U`/`V` with the
remaining `maxrank - s_Q` columns zeroed. `ranks[1] == maxrank` after this
therefore means the achieved basis *itself* saturated at `maxrank` and the
spectrum still didn't fit -- genuine saturation, reported at zero extra cost
(`docs/TODO.md`, worklog item 4), not an artifact of this two-step cap.

## `err_sq` is not a trustworthy error bound under saturation

`ara_truncate!` is called here without its `energy` keyword, so it measures
error only *within* the basis `Q` actually built, assuming range-capture error
is zero. That assumption is what `ara_build_basis!`'s own stopping rule
justifies in the normal case: it will not stop while capture is large. But when
the loop is cut off by `maxrank` before it converges (`ranks[1] == maxrank`),
that justification no longer holds, and `err_sq` can underreport the true error
by orders of magnitude -- the basis is generically a near-perfect fit for
itself.

The fix would be to supply `energy = ‖X_ij‖²_F` exactly, but that is precisely
the `O(q_k²)` cross-term sum (algorithm.tex eq:energy) worklog item 4 rejects
on the hot path -- unlike `compress!`'s dense-tile sampler (A3), which gets an
exact tile norm for free from a single elementwise reduction, the factor-list
case has no cheap exact energy to offer. So: treat `ranks[1] == maxrank` itself
as the actionable saturation signal, not `err_sq`, whenever it fires.
"""
function range_find_tile!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                          ranks, err_sq,
                          ops::LogicalTLROperands, i::Integer, j::Integer;
                          alpha, beta=false, C=nothing,
                          eps_rel::Real, r_required::Int=10,
                          tol::Real, rel::Bool=true,
                          block::Int=32, compute=nothing) where {T}
    bm, maxrank, _ = size(U)
    bn = size(V, 1)
    size(U, 3) == 1 && size(V, 3) == 1 && length(ranks) == 1 && length(err_sq) == 1 ||
        throw(ArgumentError("range_find_tile! is scoped to a single tile: " *
                            "U, V, ranks, err_sq must all carry batch size 1"))

    coupling, beta_term = tile_factor_list(ops, i, j; alpha, beta, C, compute)
    backend = get_backend(coupling.S)

    if maxrank == 0
        fill!(ranks, zero(eltype(ranks)))
        fill!(err_sq, 0.0)
        return (; passes=0)
    end

    ws = ARAWorkspace(T, backend, bm, maxrank, 1; block)
    Omega = allocate(backend, T, bn, ws.block, 1)
    sampler = function (Y, sketch_width)
        Om = view(Omega, :, 1:sketch_width, 1)
        Random.randn!(Om)
        apply_right!(view(Y, :, 1:sketch_width, 1), coupling, beta_term, Om; compute)
        return Y
    end
    basis = ara_build_basis!(ws, sampler; eps_rel, r_required, compute)
    sQ = size(basis.Q, 2)

    if sQ == 0
        fill!(U, zero(T)); fill!(V, zero(T))
        fill!(ranks, zero(eltype(ranks))); fill!(err_sq, 0.0)
        return (; passes=basis.passes)
    end

    Z = allocate(backend, T, bn, sQ, 1)
    apply_left!(view(Z, :, :, 1), coupling, beta_term, view(basis.Q, :, :, 1);
               compute)

    # Truncate at the basis's own sketch width; see the docstring for why `maxrank`
    # itself would be the wrong cap whenever the basis converged early.
    Uh = allocate(backend, T, bm, sQ, 1)
    Vh = allocate(backend, T, bn, sQ, 1)
    ara_truncate!(Uh, Vh, ranks, err_sq, basis.Q, Z;
                 tol, relative=rel, maxrank=sQ, compute)

    view(U, :, 1:sQ, :) .= Uh
    view(V, :, 1:sQ, :) .= Vh
    sQ < maxrank && (fill!(view(U, :, (sQ + 1):maxrank, :), zero(T));
                     fill!(view(V, :, (sQ + 1):maxrank, :), zero(T)))
    return (; passes=basis.passes)
end
