# Row-basis workspace sizing is intentionally expressed in terms of liveness, rather
# than as one permanent buffer per numerical stage.  The executor will carve
# concrete views from one arena according to this plan.

"""Runtime shape of one regular-grid row-basis schedule candidate."""
struct RowBasisWorkspaceShape
    b::Int
    K::Int
    rA::Int
    rB::Int
    rC::Int
    S::Int
    t::Int
    h::Int
    jblock::Int
    q::Int
    merge_tiles::Int
end

function RowBasisWorkspaceShape(b::Int, K::Int, rA::Int, rB::Int, rC::Int,
                          S::Int, t::Int;
                          h::Int=1, jblock::Int=1, q::Int=K,
                          merge_tiles::Int=h * jblock)
    b >= 0 || throw(ArgumentError("b must be nonnegative"))
    K >= 0 || throw(ArgumentError("K must be nonnegative"))
    rA >= 0 || throw(ArgumentError("rA must be nonnegative"))
    rB >= 0 || throw(ArgumentError("rB must be nonnegative"))
    rC >= 0 || throw(ArgumentError("rC must be nonnegative"))
    0 <= S <= b || throw(ArgumentError("S must satisfy 0 <= S <= b"))
    0 <= t <= S || throw(ArgumentError("t must satisfy 0 <= t <= S"))
    h >= 1 || throw(ArgumentError("h must be positive"))
    jblock >= 1 || throw(ArgumentError("jblock must be positive"))
    0 <= q <= K || throw(ArgumentError("q must satisfy 0 <= q <= K"))
    1 <= merge_tiles <= h * jblock ||
        throw(ArgumentError("merge_tiles must be in 1:h*jblock"))
    return RowBasisWorkspaceShape(b, K, rA, rB, rC, S, t, h, jblock, q, merge_tiles)
end

"""Liveness result for one row-basis workspace candidate, in bytes."""
struct RowBasisWorkspacePlan
    shape::RowBasisWorkspaceShape
    basis_bytes::Int
    accumulation_bytes::Int
    merge_bytes::Int
    bytes::Int
    pipeline_slots::Int
end

@inline _workspace_bytes(nelems::Int, ::Type{T}) where {T} = nelems * sizeof(T)

"""
    row_basis_workspace_plan(shape, T; Tgram=tlr_orthogonalization_type(T),
                             pipeline_slots=1) -> RowBasisWorkspacePlan

Calculate the phase live peaks for one row-basis scheduling candidate. `pipeline_slots=1`
models the default single-stream, liveness-carved arena. `pipeline_slots=2`
models the only initially contemplated pipeline: one basis-build slot overlaps
one accumulate/merge slot, so their arenas cannot alias.

The result deliberately does not claim a GPU-saturation recommendation. That
selection additionally needs backend calibration of candidate `(h, jblock, q)`
shapes. It does provide the exact storage roles the eventual executor must
either carve or keep disjoint.
"""
function row_basis_workspace_plan(shape::RowBasisWorkspaceShape, ::Type{T};
                                  Tgram::Type=tlr_orthogonalization_type(T),
                                  pipeline_slots::Int=1) where {T}
    pipeline_slots in (1, 2) ||
        throw(ArgumentError("only one or two pipeline slots are supported"))

    (; b, K, rA, rB, rC, S, t, h, jblock, q, merge_tiles) = shape
    # The rotated P is compacted into persistent storage after the basis pass.
    persistent = h * (b * t + K * t * rA)

    # Ωγ, Y/Q0, Pfull, and the explicit residual copy. Pfull is reused as the
    # compact P destination; Ebar is released before coefficient accumulation.
    basis_storage = h * (K * rA * S + b * S + S * K * rA + b * K * rA)
    # The weighted covariance is high precision. Its scaled P scratch avoids
    # overwriting the unweighted Pfull needed after eigenvector rotation.
    basis_gram = h * (S * K * rA + S * S + S)

    # The t <= rA branch precompresses A's right factor. The other branch
    # carries S_ikj only until it is overwritten by R.
    contraction = if t <= rA
        h * q * b * t
    else
        h * jblock * q * rA * rB
    end
    rstack = h * jblock * q * rB * t
    coeff = h * jblock * b * t

    # M's first t columns are reused as the first part of Vmerge. The residual
    # factors and merge tail are only live for the selected merge batch.
    rho = min(rC, max(b - t, 0))
    merge_scratch = merge_tiles * b * (2 * rC + rho)

    basis_bytes = _workspace_bytes(basis_storage, T) + _workspace_bytes(basis_gram, Tgram)
    accumulation_bytes = _workspace_bytes(persistent + contraction + rstack + coeff, T)
    merge_bytes = _workspace_bytes(persistent + coeff + merge_scratch, T)
    bytes = pipeline_slots == 1 ?
            max(basis_bytes, accumulation_bytes, merge_bytes) :
            basis_bytes + max(accumulation_bytes, merge_bytes)
    return RowBasisWorkspacePlan(
        shape, basis_bytes, accumulation_bytes, merge_bytes, bytes, pipeline_slots,
    )
end

"""Whether a candidate plan fits a caller-provided workspace budget in bytes."""
@inline workspace_fits(plan::RowBasisWorkspacePlan, budget::Integer) =
    budget >= 0 && plan.bytes <= budget
