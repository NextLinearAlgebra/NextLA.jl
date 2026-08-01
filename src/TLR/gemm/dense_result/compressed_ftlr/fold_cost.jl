"""Exact numerical workspace requirements and FLOP costs for a CompressedFTLR
FoldRight/FoldLeft row run."""
struct RaggedWorkspaceProfile
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
metadata (`rank_metadata.jl`). Mirrors the paper's boxed formulas:

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
(`execute.jl`) actually allocates and computes.
"""
function _compressed_ftlr_fold_cost(meta, A, B)
    qm = length(meta.pair_ranks)
    qk = length(meta.b_row_ranks)
    qn = length(meta.output_col_widths)
    Tbytes = sizeof(eltype(A))

    right = _compressed_ftlr_right_valid(A, B) ?
        [(meta.pair_ranks[i] + meta.output_col_prefix[end] * meta.a_k_prefix[i, end]) * Tbytes
         for i in 1:qm] : nothing
    left = _compressed_ftlr_left_valid(A, B) ?
        [(meta.pair_ranks[i] + meta.output_row_heights[i] * meta.b_total_rank) * Tbytes
         for i in 1:qm] : nothing
    (right === nothing && left === nothing) && throw(ArgumentError(
        "CompressedFTLR needs row-packed B outer factors and either row-packed A outer factors or column-packed B inner factors"))
    row_bytes = [min(right === nothing ? typemax(Int) : right[i],
                     left === nothing ? typemax(Int) : left[i]) for i in 1:qm]

    # Σ_j w_j Σ_k rA_ik rB_kj = Σ_k rA_ik (Σ_j w_j rB_kj) = Σ_k rA_ik ω_k -- hoisting
    # ω_k turns an O(qm·qn·qk) comprehension into O(qk·qn + qm·qk). The FoldLeft
    # Stage-3 term Σ_j w_j γ_j is independent of i, so it is hoisted the same way.
    omega = Base.zeros(Int, qk)
    @inbounds for k in 1:qk, j in 1:qn
        omega[k] += meta.output_col_widths[j] * _compressed_ftlr_execution_rank(B, k, j)
    end
    weighted_col_rank = 0
    @inbounds for j in 1:qn
        weighted_col_rank += meta.output_col_widths[j] * meta.b_col_ranks[j]
    end

    right_flops = right === nothing ? nothing :
        [sum((_compressed_ftlr_execution_rank(A, i, k) * omega[k] for k in 1:qk); init=0) +
         meta.output_row_heights[i] * meta.output_col_prefix[end] * meta.a_k_prefix[i, end]
         for i in 1:qm]
    left_flops = left === nothing ? nothing :
        [meta.output_row_heights[i] * (meta.pair_ranks[i] + weighted_col_rank) for i in 1:qm]

    maximum_bytes = min(right === nothing ? typemax(Int) : sum(right),
                        left === nothing ? typemax(Int) : sum(left))
    return RaggedWorkspaceProfile(
        row_bytes, right, left, right_flops, left_flops,
        right === nothing ? nothing : _compressed_ftlr_prefix(right),
        left === nothing ? nothing : _compressed_ftlr_prefix(left),
        right_flops === nothing ? nothing : _compressed_ftlr_prefix(right_flops),
        left_flops === nothing ? nothing : _compressed_ftlr_prefix(left_flops),
        isempty(row_bytes) ? 0 : maximum(row_bytes), maximum_bytes,
    )
end
