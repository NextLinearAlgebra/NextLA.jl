"""Exact numerical workspace requirements and FLOP costs for a CompressedFTLR
FoldRight/FoldLeft row run."""
struct RaggedWorkspaceProfile
    columns::UnitRange{Int}   # output tile columns these costs were computed for
    row_bytes::Vector{Int}
    right_row_bytes::Union{Nothing,Vector{Int}}
    left_row_bytes::Union{Nothing,Vector{Int}}
    right_flops::Union{Nothing,Vector{Int}}
    left_flops::Union{Nothing,Vector{Int}}
    right_byte_prefix::Union{Nothing,Vector{Int}}
    left_byte_prefix::Union{Nothing,Vector{Int}}
    right_flop_prefix::Union{Nothing,Vector{Int}}
    left_flop_prefix::Union{Nothing,Vector{Int}}
    minimum::Int
    maximum::Int
end

"""
    _compressed_ftlr_fold_cost(meta, A, B)

Closed-form per-row byte and FLOP cost for both fold bracketings, given rank
metadata (`metadata.jl`). Mirrors the paper's boxed formulas:

    F_R(i) = 2·n_·Σ_k rA_ik·rB_k· + 2·m_i·n_·ρ_i        (Stage 2 + Stage 3, FoldRight)
    F_L(i) = 2·m_i·Σ_k rA_ik·rB_k· + 2·m_i·n_·γ_·        (Stage 2 + Stage 3, FoldLeft)

Stage 2's cost is identical either way (`pair_ranks[i]`/`Σ_k rA_ik·rB_kj` terms),
so only the Stage-3 term differs. Every quantity here is computed *unconditionally*
per row rather than short-circuited to zero when `pair_ranks[i] == 0`: a row with
no A-rank at all does reduce to zero automatically (every term it appears in is a
product with a factor that is itself zero), but a row can have `pair_ranks[i] == 0`
while still needing real Stage-3 storage/FLOPs — either because `ρ_i > 0` (A has
rank at some `k` where B's row-block `k` happens to be entirely rank-zero, so
Stage 1 contributes nothing there but Stage 3 still processes that row's own
`ρ_i`-wide operand), or because FoldLeft's Stage 3 GEMM spans a run's whole
contiguous physical row range (`run_height`), so a zero-rank row embedded between
nonzero neighbors still occupies its `row_height · b_total_rank` slice of the T
arena even though it contributes nothing to the S arena. A per-row ternary
shortcut on `pair_ranks[i]` would silently drop exactly this contribution and
under-count the true workspace/FLOP need — the unconditional formula, summed via
ordinary prefix sums over any row range, always matches what the executor
(`three_stage.jl`) actually allocates and computes.
"""
function _compressed_ftlr_fold_cost(meta, A, B,
                                    jrange::UnitRange{Int}=1:length(meta.output_col_widths))
    qm = length(meta.pair_ranks)
    qk = length(meta.b_row_ranks)
    Tbytes = sizeof(eltype(A))
    width = _compressed_ftlr_width(meta, jrange)
    total_rank = _compressed_ftlr_total_rank(meta, jrange)

    # `pair_ranks` is stored over the whole grid; restricted to `jrange` it is
    # Σ_k rA_ik·σ_k(jrange). Recomputing costs O(qm·qk), the same order as the
    # rest of this function, and is the only per-column-block work needed.
    pair = if jrange == 1:length(meta.output_col_widths)
        meta.pair_ranks
    else
        p = Base.zeros(Int, qm)
        @inbounds for i in 1:qm, k in 1:qk
            p[i] += _compressed_ftlr_storage_rank(A, i, k) *
                    _compressed_ftlr_row_rank(meta, k, jrange)
        end
        p
    end

    right = _compressed_ftlr_right_valid(A, B) ?
        [(pair[i] + width * meta.a_k_prefix[i, end]) * Tbytes for i in 1:qm] : nothing
    left = _compressed_ftlr_left_valid(A, B) ?
        [(pair[i] + meta.output_row_heights[i] * total_rank) * Tbytes for i in 1:qm] : nothing
    (right === nothing && left === nothing) && throw(ArgumentError(
        "CompressedFTLR needs row-packed B outer factors and either row-packed A outer factors or column-packed B inner factors"))
    row_bytes = [min(right === nothing ? typemax(Int) : right[i],
                     left === nothing ? typemax(Int) : left[i]) for i in 1:qm]

    # Σ_j w_j Σ_k rA_ik rB_kj = Σ_k rA_ik (Σ_j w_j rB_kj) = Σ_k rA_ik ω_k -- hoisting
    # ω_k turns an O(qm·qn·qk) comprehension into O(qk·qn + qm·qk). The FoldLeft
    # Stage-3 term Σ_j w_j γ_j is independent of i, so it is hoisted the same way.
    # Both sums run over `jrange` only.
    omega = Base.zeros(Int, qk)
    @inbounds for k in 1:qk, j in jrange
        omega[k] += meta.output_col_widths[j] * _compressed_ftlr_storage_rank(B, k, j)
    end
    weighted_col_rank = 0
    @inbounds for j in jrange
        weighted_col_rank += meta.output_col_widths[j] * meta.b_col_ranks[j]
    end

    right_flops = right === nothing ? nothing :
        [sum((_compressed_ftlr_storage_rank(A, i, k) * omega[k] for k in 1:qk); init=0) +
         meta.output_row_heights[i] * width * meta.a_k_prefix[i, end]
         for i in 1:qm]
    left_flops = left === nothing ? nothing :
        [meta.output_row_heights[i] * (pair[i] + weighted_col_rank) for i in 1:qm]

    maximum_bytes = min(right === nothing ? typemax(Int) : sum(right),
                        left === nothing ? typemax(Int) : sum(left))
    return RaggedWorkspaceProfile(
        jrange, row_bytes, right, left, right_flops, left_flops,
        right === nothing ? nothing : _compressed_ftlr_prefix(right),
        left === nothing ? nothing : _compressed_ftlr_prefix(left),
        right_flops === nothing ? nothing : _compressed_ftlr_prefix(right_flops),
        left_flops === nothing ? nothing : _compressed_ftlr_prefix(left_flops),
        isempty(row_bytes) ? 0 : maximum(row_bytes), maximum_bytes,
    )
end
