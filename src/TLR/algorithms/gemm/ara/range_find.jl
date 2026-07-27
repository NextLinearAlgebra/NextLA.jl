# R2: RangeFind for one output tile (algorithm.tex Algorithm 1) -- A1's ARA
# loop driven through R1's implicit ApplyRight/ApplyLeft, with no exact
# residual on the hot path (docs/TODO.md, worklog item 4).

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
    through [`apply_right!`](@ref) (R1). The basis never grows past `maxrank`
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

@kernel function _scatter_run_factor_kernel!(dest, src, ranks_slot,
                                             slot_to_member)
    i, k, slot = @index(Global, NTuple)
    @inbounds dest[i, k, Int(slot_to_member[slot])] =
        k <= Int(ranks_slot[slot]) ? src[i, k, slot] : zero(eltype(dest))
end

@kernel function _scatter_run_diagnostic_kernel!(ranks, err_sq, ranks_slot,
                                                 err_slot, slot_to_member)
    slot = @index(Global, Linear)
    @inbounds begin
        member = Int(slot_to_member[slot])
        ranks[member] = ranks_slot[slot]
        err_sq[member] = err_slot[slot]
    end
end

"""
    range_find_column_run!(U, V, ranks, err_sq, ops, rows, j; ...)

R3 fixed-column RangeFind. The tiles `(rows, j)` share one `H_ℓj` per sampling
pass and occupy a physically contiguous active prefix. Members that converge
are swap-removed into a retired suffix; the complete run is truncated as one
batch and scattered once back to the caller's original row order.

The canonical row-major API uses this family for the right-selected `NT` case,
where a fixed logical-B column shares `H_ℓj`. The `NN` right path and the
left-sampling paths use the symmetric fixed-row driver below.
"""
function range_find_column_run!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                                ranks, err_sq,
                                ops::LogicalTLROperands, rows, j::Integer;
                                alpha, beta=false, C=nothing,
                                eps_rel::Real, r_required::Int=10,
                                tol::Real, rel::Bool=true,
                                block::Int=32,
                                rA::Int=rankdim(ops.au),
                                rB::Int=rankdim(ops.bv),
                                compute=nothing) where {T}
    row_ids = collect(Int, rows)
    nmember = length(row_ids)
    bm, maxrank, countU = size(U)
    bn, maxrankV, countV = size(V)
    countU == countV == nmember == length(ranks) == length(err_sq) ||
        throw(DimensionMismatch("run outputs and diagnostics must match length(rows)"))
    maxrank == maxrankV ||
        throw(DimensionMismatch("U and V must have the same rank capacity"))
    nmember == 0 && return (; passes=0, active_counts=Int[], member_ids=Int[])
    fill!(U, zero(T)); fill!(V, zero(T))

    if maxrank == 0
        fill!(ranks, zero(eltype(ranks)))
        fill!(err_sq, 0.0)
        return (; passes=0, active_counts=Int[], member_ids=collect(1:nmember))
    end

    backend = get_backend(U)
    ws = ARAWorkspace(T, backend, bm, maxrank, nmember; block)
    run = ColumnRunCoupling(ops, row_ids, j;
                            alpha, beta, C, block=ws.block, maxrank,
                            rA, rB, compute)
    member_ids = collect(1:nmember)
    sampler = function (Y, sketch_width, active_ids)
        apply_right_run!(Y, run, sketch_width, length(active_ids); beta, compute)
    end
    basis = ara_build_basis_packed!(
        ws, sampler, member_ids; eps_rel, r_required, compute,
        swap_member! = (p, q) -> _swap_column_run_members!(run, p, q),
    )
    sQ = size(basis.Q, 2)

    if sQ == 0
        fill!(ranks, zero(eltype(ranks)))
        fill!(err_sq, 0.0)
        return (; passes=basis.passes, active_counts=basis.active_counts,
                member_ids=basis.member_ids)
    end

    Z = allocate(backend, T, bn, sQ, nmember)
    apply_left_run!(Z, run, basis.Q, sQ; beta, compute)
    Uh = allocate(backend, T, bm, sQ, nmember)
    Vh = allocate(backend, T, bn, sQ, nmember)
    ranks_slot = allocate(backend, eltype(ranks), nmember)
    err_slot = allocate(backend, eltype(err_sq), nmember)
    ara_truncate!(Uh, Vh, ranks_slot, err_slot, basis.Q, Z;
                  tol, relative=rel, maxrank=sQ, compute)

    slot_to_member = copyto!(allocate(backend, Int32, nmember),
                             Int32.(basis.member_ids))
    _scatter_run_factor_kernel!(backend)(
        view(U, :, 1:sQ, :), Uh, ranks_slot, slot_to_member;
        ndrange=(bm, sQ, nmember),
    )
    _scatter_run_factor_kernel!(backend)(
        view(V, :, 1:sQ, :), Vh, ranks_slot, slot_to_member;
        ndrange=(bn, sQ, nmember),
    )
    _scatter_run_diagnostic_kernel!(backend)(
        ranks, err_sq, ranks_slot, err_slot, slot_to_member; ndrange=nmember,
    )
    return (; passes=basis.passes, active_counts=basis.active_counts,
            member_ids=basis.member_ids)
end

@inline function _validate_row_run_outputs(U, V, ranks, err_sq, cols)
    nmember = length(cols)
    size(U, 3) == size(V, 3) == nmember == length(ranks) == length(err_sq) ||
        throw(DimensionMismatch("run outputs and diagnostics must match length(cols)"))
    size(U, 2) == size(V, 2) ||
        throw(DimensionMismatch("U and V must have the same rank capacity"))
    return nmember
end

function _scatter_range_run!(U, V, ranks, err_sq, Uh, Vh, ranks_slot, err_slot,
                             member_ids, sQ)
    backend = get_backend(U)
    nmember = length(member_ids)
    slot_to_member = copyto!(allocate(backend, Int32, nmember), Int32.(member_ids))
    _scatter_run_factor_kernel!(backend)(
        view(U, :, 1:sQ, :), Uh, ranks_slot, slot_to_member;
        ndrange=(size(U, 1), sQ, nmember),
    )
    _scatter_run_factor_kernel!(backend)(
        view(V, :, 1:sQ, :), Vh, ranks_slot, slot_to_member;
        ndrange=(size(V, 1), sQ, nmember),
    )
    _scatter_run_diagnostic_kernel!(backend)(
        ranks, err_sq, ranks_slot, err_slot, slot_to_member; ndrange=nmember,
    )
    return nothing
end

"""
    range_find_row_right_run!(...)

Fixed-output-row R3 path using repeated `XΩ` samples. This is the canonical
`NN` path: A's logical tile row is the zero-copy terminal stack.
"""
function range_find_row_right_run!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                                   ranks, err_sq, ops::LogicalTLROperands,
                                   i::Integer, cols;
                                   alpha, beta=false, C=nothing,
                                   eps_rel::Real, r_required::Int=10,
                                   tol::Real, rel::Bool=true,
                                   block::Int=32,
                                   rA::Int=rankdim(ops.au),
                                   rB::Int=rankdim(ops.bv),
                                   compute=nothing) where {T}
    col_ids = collect(Int, cols)
    nmember = _validate_row_run_outputs(U, V, ranks, err_sq, col_ids)
    nmember == 0 &&
        return (; passes=0, active_counts=Int[], member_ids=Int[], side=:right)
    fill!(U, zero(T)); fill!(V, zero(T))
    maxrank = size(U, 2)
    if maxrank == 0
        fill!(ranks, zero(eltype(ranks))); fill!(err_sq, 0.0)
        return (; passes=0, active_counts=Int[], member_ids=collect(1:nmember),
                side=:right)
    end

    backend = get_backend(U)
    ws = ARAWorkspace(T, backend, size(U, 1), maxrank, nmember; block)
    run = RowRightRunCoupling(
        ops, i, col_ids; alpha, beta, C, block=ws.block, maxrank,
        rA, rB, compute,
    )
    member_ids = collect(1:nmember)
    sampler = function (Y, sketch_width, active_ids)
        apply_right_row_run!(Y, run, sketch_width, length(active_ids); beta, compute)
    end
    basis = ara_build_basis_packed!(
        ws, sampler, member_ids; eps_rel, r_required, compute,
        swap_member! = (p, q) -> _swap_row_right_members!(run, p, q),
    )
    sQ = size(basis.Q, 2)
    sQ == 0 &&
        return (; passes=basis.passes, active_counts=basis.active_counts,
                member_ids=basis.member_ids, side=:right)

    Z = allocate(backend, T, size(V, 1), sQ, nmember)
    apply_left_row_run!(Z, run, basis.Q, sQ; beta, compute)
    Uh = allocate(backend, T, size(U, 1), sQ, nmember)
    Vh = allocate(backend, T, size(V, 1), sQ, nmember)
    ranks_slot = allocate(backend, eltype(ranks), nmember)
    err_slot = allocate(backend, eltype(err_sq), nmember)
    ara_truncate!(Uh, Vh, ranks_slot, err_slot, basis.Q, Z;
                  tol, relative=rel, maxrank=sQ, compute)
    _scatter_range_run!(U, V, ranks, err_sq, Uh, Vh, ranks_slot, err_slot,
                        basis.member_ids, sQ)
    return (; passes=basis.passes, active_counts=basis.active_counts,
            member_ids=basis.member_ids, side=:right)
end

"""
    range_find_row_left_run!(...)

Fixed-output-row R3 path using repeated `XᵀΩ` samples. The ARA basis is the
right output basis; truncation is applied to `Xᵀ` and its factors are swapped
back to the conventional `X ≈ U Vᵀ` orientation.
"""
function range_find_row_left_run!(U::AbstractArray{T,3}, V::AbstractArray{T,3},
                                  ranks, err_sq, ops::LogicalTLROperands,
                                  i::Integer, cols;
                                  alpha, beta=false, C=nothing,
                                  eps_rel::Real, r_required::Int=10,
                                  tol::Real, rel::Bool=true,
                                  block::Int=32,
                                  rA::Int=rankdim(ops.au),
                                  rB::Int=rankdim(ops.bv),
                                  compute=nothing) where {T}
    col_ids = collect(Int, cols)
    nmember = _validate_row_run_outputs(U, V, ranks, err_sq, col_ids)
    nmember == 0 &&
        return (; passes=0, active_counts=Int[], member_ids=Int[], side=:left)
    fill!(U, zero(T)); fill!(V, zero(T))
    maxrank = size(U, 2)
    if maxrank == 0
        fill!(ranks, zero(eltype(ranks))); fill!(err_sq, 0.0)
        return (; passes=0, active_counts=Int[], member_ids=collect(1:nmember),
                side=:left)
    end

    backend = get_backend(U)
    ws = ARAWorkspace(T, backend, size(V, 1), maxrank, nmember; block)
    run = RowLeftRunCoupling(
        ops, i, col_ids; alpha, beta, C, block=ws.block, maxrank,
        rA, rB, compute,
    )
    member_ids = collect(1:nmember)
    sampler = function (Y, sketch_width, active_ids)
        apply_left_row_run!(Y, run, sketch_width, length(active_ids); beta, compute)
    end
    basis = ara_build_basis_packed!(
        ws, sampler, member_ids; eps_rel, r_required, compute,
        swap_member! = (p, q) -> _swap_row_left_members!(run, p, q),
    )
    sQ = size(basis.Q, 2)
    sQ == 0 &&
        return (; passes=basis.passes, active_counts=basis.active_counts,
                member_ids=basis.member_ids, side=:left)

    L = allocate(backend, T, size(U, 1), sQ, nmember)
    apply_right_row_run!(L, run, basis.Q, sQ; beta, compute)
    Uh = allocate(backend, T, size(U, 1), sQ, nmember)
    Vh = allocate(backend, T, size(V, 1), sQ, nmember)
    ranks_slot = allocate(backend, eltype(ranks), nmember)
    err_slot = allocate(backend, eltype(err_sq), nmember)
    # Xᵀ ≈ Q_R Lᵀ. Truncating the transpose returns (V, U).
    ara_truncate!(Vh, Uh, ranks_slot, err_slot, basis.Q, L;
                  tol, relative=rel, maxrank=sQ, compute)
    _scatter_range_run!(U, V, ranks, err_sq, Uh, Vh, ranks_slot, err_slot,
                        basis.member_ids, sQ)
    return (; passes=basis.passes, active_counts=basis.active_counts,
            member_ids=basis.member_ids, side=:left)
end
