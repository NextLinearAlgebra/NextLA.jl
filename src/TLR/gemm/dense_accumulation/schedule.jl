"""Whether FoldRight has the required row-packed outer factors."""

@inline compressed_ftlr_right_valid(A, B) =
    compressed_ftlr_outer_order(A) isa TileRowMajor && compressed_ftlr_outer_order(B) isa TileRowMajor
@inline compressed_ftlr_left_valid(A, B) =
    compressed_ftlr_outer_order(B) isa TileRowMajor && compressed_ftlr_inner_order(B) isa TileColMajor

@inline function compressed_ftlr_prefix(values::Vector{Int})
    prefix = Base.zeros(Int, length(values) + 1)
    @inbounds for i in eachindex(values)
        prefix[i + 1] = prefix[i] + values[i]
    end
    return prefix
end

@inline _compressed_ftlr_range_total(prefix::Vector{Int}, rows::UnitRange{Int}) =
    prefix[last(rows) + 1] - prefix[first(rows)]

# column-range views: work is scheduled over tile rows `irange` × tile columns
# `jrange`; a narrower `jrange` than the full `1:qn` is what lets the workspace
# budget be met below one output row, so every `j`-aggregating quantity below
# is queried on the range rather than read whole.

"""`σ_k` restricted to `jrange`: `Σ_{j ∈ jrange} rB_kj`."""
@inline compressed_ftlr_row_rank(meta, k::Int, jrange) =
    meta.b_row_k_prefix[k, last(jrange) + 1] - meta.b_row_k_prefix[k, first(jrange)]

"""Column offset of tile `j` inside a `jrange`-local S block."""
@inline compressed_ftlr_row_rank_offset(meta, k::Int, j::Int, jrange) =
    meta.b_row_k_prefix[k, j] - meta.b_row_k_prefix[k, first(jrange)]

"""Total output width spanned by `jrange`, in matrix columns."""
@inline compressed_ftlr_width(meta, jrange) =
    meta.output_col_prefix[last(jrange) + 1] - meta.output_col_prefix[first(jrange)]

"""Matrix column span of `jrange` in the dense output."""
@inline function compressed_ftlr_output_cols(B, jrange)
    lo = first(tile_axis_range(B, first(jrange), 2))
    hi = last(tile_axis_range(B, last(jrange), 2))
    return lo:hi
end

"""
    _compressed_ftlr_rank_metadata(A, B)

Build host rank prefixes for constant-time range queries. `a_k_prefix` tracks A
over `k`; `b_row_ranks` and `b_col_ranks` are B's row/column rank sums;
`pair_ranks[i] = Σ_k rA_ik·b_row_ranks[k]`. Output extents are rank-independent.
"""
function _compressed_ftlr_rank_metadata(A, B)
    qm, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))

    a_k_prefix = Base.zeros(Int, qm, qk + 1)
    b_row_ranks = Base.zeros(Int, qk)
    b_col_ranks = Base.zeros(Int, qn)
    b_col_k_prefix = Base.zeros(Int, qn, qk + 1)
    b_row_k_prefix = Base.zeros(Int, qk, qn + 1)

    # nonzero B-tile prefixes
    # Stage 2 skips zero-rank tiles, so execution needs O(1) hole detection per range.
    b_row_nonzero_prefix = Base.zeros(Int, qk, qn + 1)
    @inbounds for k in 1:qk, j in 1:qn
        r = compressed_ftlr_storage_rank(B, k, j)
        b_row_ranks[k] += r
        b_col_ranks[j] += r
        b_row_nonzero_prefix[k, j + 1] = b_row_nonzero_prefix[k, j] + (r != 0)
    end

    # B rank prefixes
    @inbounds for j in 1:qn, k in 1:qk
        b_col_k_prefix[j, k + 1] = b_col_k_prefix[j, k] + compressed_ftlr_storage_rank(B, k, j)
    end
    @inbounds for k in 1:qk, j in 1:qn
        b_row_k_prefix[k, j + 1] = b_row_k_prefix[k, j] + compressed_ftlr_storage_rank(B, k, j)
    end

    # A prefixes and shared-stage sizes
    pair_ranks = Base.zeros(Int, qm)
    @inbounds for i in 1:qm, k in 1:qk
        r = compressed_ftlr_storage_rank(A, i, k)
        a_k_prefix[i, k + 1] = a_k_prefix[i, k] + r
        pair_ranks[i] += r * b_row_ranks[k]
    end

    # output geometry
    b_col_prefix = compressed_ftlr_prefix(b_col_ranks)
    output_row_heights = [length(tile_axis_range(A, i, 1)) for i in 1:qm]
    output_col_widths = [length(tile_axis_range(B, j, 2)) for j in 1:qn]
    output_col_prefix = compressed_ftlr_prefix(output_col_widths)

    return (; a_k_prefix, qk, b_col_ranks, b_col_k_prefix, b_row_k_prefix,
              b_col_prefix, b_row_nonzero_prefix, pair_ranks,
              output_row_heights, output_col_widths, output_col_prefix)
end

"""
    _compressed_ftlr_fold_cost(meta, A, B)

Per-row byte and FLOP costs for both fold bracketings:

    F_R(i) = 2·n_·Σ_k rA_ik·rB_k· + 2·m_i·n_·ρ_i        (Stage 2 + Stage 3, FoldRight)
    F_L(i) = 2·m_i·Σ_k rA_ik·rB_k· + 2·m_i·n_·γ_·        (Stage 2 + Stage 3, FoldLeft)

Stage 2 is shared; only Stage 3 differs. Costs must not short-circuit when
`pair_ranks[i] == 0`: A may retain rank where B's contraction row is zero, and a
zero-rank row inside a FoldLeft run still occupies its contiguous output slice.
Unconditional terms therefore match the allocated arenas.
"""
function _compressed_ftlr_fold_cost(meta, A, B,
                                    jrange::UnitRange{Int}=1:length(meta.output_col_widths))
    qm = length(meta.pair_ranks)
    qk = meta.qk
    Tbytes = sizeof(eltype(A))
    width = compressed_ftlr_width(meta, jrange)
    total_rank = meta.b_col_prefix[last(jrange) + 1] -
                 meta.b_col_prefix[first(jrange)]

    # column-restricted shared-stage sizes
    # `pair_ranks` is full-grid, so narrower column ranges are recomputed.
    pair = if jrange == 1:length(meta.output_col_widths)
        meta.pair_ranks
    else
        p = Base.zeros(Int, qm)
        @inbounds for i in 1:qm, k in 1:qk
            p[i] += compressed_ftlr_storage_rank(A, i, k) *
                    compressed_ftlr_row_rank(meta, k, jrange)
        end
        p
    end

    # per-row workspace
    right = compressed_ftlr_right_valid(A, B) ?
        [(pair[i] + width * meta.a_k_prefix[i, end]) * Tbytes for i in 1:qm] : nothing
    left = compressed_ftlr_left_valid(A, B) ?
        [(pair[i] + meta.output_row_heights[i] * total_rank) * Tbytes for i in 1:qm] : nothing
    (right === nothing && left === nothing) && throw(ArgumentError(
        "CompressedFTLR needs row-packed B outer factors and either row-packed A outer factors or column-packed B inner factors"))
    row_bytes = [min(right === nothing ? typemax(Int) : right[i],
                     left === nothing ? typemax(Int) : left[i]) for i in 1:qm]

    # hoisted flop weights over the selected columns
    # `ω_k` reduces the FoldRight sum to O(qk·qn + qm·qk); FoldLeft's weighted
    # column rank is likewise independent of output row.
    omega = Base.zeros(Int, qk)
    @inbounds for k in 1:qk, j in jrange
        omega[k] += meta.output_col_widths[j] * compressed_ftlr_storage_rank(B, k, j)
    end
    weighted_col_rank = 0
    @inbounds for j in jrange
        weighted_col_rank += meta.output_col_widths[j] * meta.b_col_ranks[j]
    end

    # per-row flop costs
    right_flops = right === nothing ? nothing :
        [sum((compressed_ftlr_storage_rank(A, i, k) * omega[k] for k in 1:qk); init=0) +
         meta.output_row_heights[i] * width * meta.a_k_prefix[i, end]
         for i in 1:qm]
    left_flops = left === nothing ? nothing :
        [meta.output_row_heights[i] * (pair[i] + weighted_col_rank) for i in 1:qm]

    # prefix profiles
    maximum_bytes = min(right === nothing ? typemax(Int) : sum(right),
                        left === nothing ? typemax(Int) : sum(left))
    return (
        columns=jrange, nrows=qm,
        right_byte_prefix=right === nothing ? nothing : compressed_ftlr_prefix(right),
        left_byte_prefix=left === nothing ? nothing : compressed_ftlr_prefix(left),
        right_flop_prefix=right_flops === nothing ? nothing : compressed_ftlr_prefix(right_flops),
        left_flop_prefix=left_flops === nothing ? nothing : compressed_ftlr_prefix(left_flops),
        minimum=isempty(row_bytes) ? 0 : maximum(row_bytes), maximum=maximum_bytes,
    )
end

"""One rectangular output-tile run executed with one `fold`."""
function compressed_ftlr_rank_plan(A, B)
    meta = _compressed_ftlr_rank_metadata(A, B)
    profile = _compressed_ftlr_fold_cost(meta, A, B)
    return (; meta..., profile)
end

"""Single-column workspace floor. The maximum workspace gives the fastest schedule."""
function gemm_minimum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    return _compressed_ftlr_column_floor(compressed_ftlr_rank_plan(LA, LB), LA, LB)
end

function gemm_maximum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    return compressed_ftlr_rank_plan(LA, LB).profile.maximum
end

"""
    gemm_workspace_bytes(A, B; runs=1, transA='N', transB='N') -> Int

Smallest workspace yielding at most `runs` rectangular work units. `runs = 1`
equals `gemm_maximum_workspace_bytes`; the result is clamped at
`gemm_minimum_workspace_bytes`.
"""
function gemm_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                              runs::Int=1, transA::Char='N', transB::Char='N')
    runs >= 1 || throw(ArgumentError("runs must be positive; got $runs"))

    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    plan = compressed_ftlr_rank_plan(LA, LB)
    lo = _compressed_ftlr_column_floor(plan, LA, LB)
    hi = plan.profile.maximum
    scheduled(bytes) = length(compressed_ftlr_column_schedule(plan, LA, LB, plan.profile, bytes))

    # monotone budget bisection
    scheduled(lo) <= runs && return lo
    while lo < hi
        mid = lo + (hi - lo) ÷ 2
        if scheduled(mid) <= runs
            hi = mid
        else
            lo = mid + 1
        end
    end
    return hi
end

"""Contiguous output-row runs greedily packed under an exact ragged budget."""
function _compressed_ftlr_row_runs(profile, budget::Int)
    budget >= profile.minimum || throw(ArgumentError(
        "workspace has $budget bytes; at least $(profile.minimum) bytes are required"))

    # greedy contiguous rows
    runs = DenseAccumulationRun[]
    i = 1
    while i <= profile.nrows
        j = i - 1
        while j < profile.nrows && _compressed_ftlr_select_fold(profile, i:(j + 1), budget) !== nothing
            j += 1
        end
        # A zero-work row is allowed even for a zero byte budget.
        j >= i || (j = i)
        fold = _compressed_ftlr_select_fold(profile, i:j, budget)
        fold === nothing && throw(ArgumentError("workspace cannot schedule CompressedFTLR row $i"))
        push!(runs, DenseAccumulationRun(i:j, profile.columns, fold))
        i = j + 1
    end

    return runs
end

function _compressed_ftlr_select_fold(profile, rows::UnitRange{Int}, budget::Int)
    # workspace feasibility
    right_bytes = profile.right_byte_prefix === nothing ? typemax(Int) :
        _compressed_ftlr_range_total(profile.right_byte_prefix, rows)
    left_bytes = profile.left_byte_prefix === nothing ? typemax(Int) :
        _compressed_ftlr_range_total(profile.left_byte_prefix, rows)
    right_ok = right_bytes <= budget
    left_ok = left_bytes <= budget
    !right_ok && !left_ok && return nothing
    right_ok && !left_ok && return :right
    left_ok && !right_ok && return :left

    # flop tiebreak
    right_flops = _compressed_ftlr_range_total(profile.right_flop_prefix, rows)
    left_flops = _compressed_ftlr_range_total(profile.left_flop_prefix, rows)
    return right_flops <= left_flops ? :right : :left
end

# Column-block partitioning lowers the workspace floor when a full output row
# does not fit. The rectangles tile `C` without repeated accumulation; only A's
# outer factors are reread. Full-width scheduling remains the fast path.

"""
Widest column block from `j0` whose row floor fits `budget`. Monotonicity permits
bisection; returns `nothing` if one column cannot fit.
"""
function _compressed_ftlr_widest_column_block(meta, A, B, j0::Int, qn::Int, budget::Int)
    _compressed_ftlr_fold_cost(meta, A, B, j0:j0).minimum <= budget || return nothing
    lo, hi = j0, qn                      # lo always feasible, hi+1 always infeasible
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if _compressed_ftlr_fold_cost(meta, A, B, j0:mid).minimum <= budget
            lo = mid
        else
            hi = mid - 1
        end
    end
    return j0:lo
end

"""
Schedule the output under `budget`, using sequential column blocks only when a
full-width run does not fit. Peak workspace is the maximum over runs.
"""
function compressed_ftlr_column_schedule(meta, A, B, profile,
                                          budget::Int)
    # full-width fast path
    profile.minimum <= budget && return _compressed_ftlr_row_runs(profile, budget)

    # sequential column blocks
    qn = length(meta.output_col_widths)
    runs = DenseAccumulationRun[]
    j = 1
    while j <= qn
        block = _compressed_ftlr_widest_column_block(meta, A, B, j, qn, budget)
        block === nothing && throw(ArgumentError(
            "workspace has $budget bytes; not enough for a single output tile column"))
        append!(runs, _compressed_ftlr_row_runs(
            _compressed_ftlr_fold_cost(meta, A, B, block), budget))
        j = last(block) + 1
    end

    return runs
end

"""Smallest workspace in which any schedule exists: the per-row floor of the
narrowest (single-column) blocks, which is what column subdivision buys over the
full-width floor `profile.minimum`."""
function _compressed_ftlr_column_floor(meta, A, B)
    qn = length(meta.output_col_widths)
    floor_bytes = 0
    @inbounds for j in 1:qn
        floor_bytes = max(floor_bytes, _compressed_ftlr_fold_cost(meta, A, B, j:j).minimum)
    end
    return floor_bytes
end

function prepare_compressed_ftlr_workspace(A, B, plan, workspace)
    profile = plan.profile
    T = eltype(A)

    # workspace ownership and compatibility
    ws = if workspace isa Int
        DenseGemmWorkspace(A, workspace)
    elseif workspace isa DenseGemmWorkspace
        eltype(workspace) === T || throw(ArgumentError(
            "workspace element type $(eltype(workspace)) does not match operand type $T"))
        typeof(get_backend(workspace.storage)) === typeof(get_backend(A)) ||
            throw(ArgumentError("workspace and operands must use the same backend"))
        workspace
    else
        throw(ArgumentError("workspace must be an integer byte count or DenseGemmWorkspace"))
    end

    # column-subdivision floor
    # `profile.minimum` is full-width, so compute the true floor only below it.
    bytes = sizeof(ws)
    if bytes < profile.minimum
        required = _compressed_ftlr_column_floor(plan, A, B)
        bytes >= required || throw(ArgumentError(
            "workspace has $bytes bytes; at least $required bytes are required"))
    end

    # bounded arena
    budget = min(bytes, profile.maximum)
    arena = GemmArena(view(ws.storage, :), 1)
    return ws, arena, budget, profile
end
