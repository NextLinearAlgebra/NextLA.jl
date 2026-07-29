"""Exact numerical workspace requirements for a CompressedFTLR FoldRight row run."""
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

"""Host-only rank reductions and O(1) range-query metadata for one CompressedFTLR GEMM."""
struct CompressedFTLRRankPlan
    a_k_prefix::Matrix{Int}      # (logical i, prefix through logical k)
    b_row_ranks::Vector{Int}     # σ_k = Σ_j rB_kj
    b_col_ranks::Vector{Int}     # γ_j = Σ_k rB_kj
    b_col_k_prefix::Matrix{Int}  # (logical j, prefix through logical k)
    b_col_prefix::Vector{Int}    # prefix of γ_j across logical j
    b_first_active_col::Vector{Int}
    pair_ranks::Vector{Int}      # p_i = Σ_k rA_ik σ_k
    b_total_rank::Int
    output_row_heights::Vector{Int}
    output_col_widths::Vector{Int}
    output_col_prefix::Vector{Int}
    profile::RaggedWorkspaceProfile
end

struct RaggedRowRun
    rows::UnitRange{Int}
    fold::Symbol
end

@inline _compressed_ftlr_axis_range(A::LogicalTLROperand, tile::Int, axis::Int) =
    _tile_axis_range(A, tile, axis)
@inline function _compressed_ftlr_axis_range(A::CompressedFTLRMatrix, tile::Int, axis::Int)
    width = axis == 1 ? tile_size(A, tile, 1)[1] : tile_size(A, 1, tile)[2]
    first = (tile - 1) * nominal_tile_size(A, axis) + 1
    return first:(first + width - 1)
end

@inline _compressed_ftlr_right_valid(A, B) =
    compressed_ftlr_outer_order(A) isa TileRowMajor && compressed_ftlr_outer_order(B) isa TileRowMajor
@inline _compressed_ftlr_left_valid(A, B) =
    compressed_ftlr_outer_order(B) isa TileRowMajor && compressed_ftlr_inner_order(B) isa TileColMajor

@inline function _compressed_ftlr_prefix(values::Vector{Int})
    prefix = Base.zeros(Int, length(values) + 1)
    @inbounds for i in eachindex(values)
        prefix[i + 1] = prefix[i] + values[i]
    end
    return prefix
end

@inline _compressed_ftlr_range_total(prefix::Vector{Int}, rows::UnitRange{Int}) =
    prefix[last(rows) + 1] - prefix[first(rows)]

function _compressed_ftlr_rank_plan(A, B)
    qm, qk = grid_size(A)
    qkB, qn = grid_size(B)
    qk == qkB || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    a_k_prefix = Base.zeros(Int, qm, qk + 1)
    b_row_ranks = Base.zeros(Int, qk)
    b_col_ranks = Base.zeros(Int, qn)
    b_col_k_prefix = Base.zeros(Int, qn, qk + 1)
    b_first_active_col = Base.zeros(Int, qk)
    @inbounds for k in 1:qk, j in 1:qn
        r = _compressed_ftlr_rank(B, k, j)
        b_row_ranks[k] += r
        b_col_ranks[j] += r
        r > 0 && b_first_active_col[k] == 0 && (b_first_active_col[k] = j)
    end
    @inbounds for j in 1:qn, k in 1:qk
        b_col_k_prefix[j, k + 1] = b_col_k_prefix[j, k] + _compressed_ftlr_rank(B, k, j)
    end
    pair_ranks = Base.zeros(Int, qm)
    @inbounds for i in 1:qm, k in 1:qk
        r = _compressed_ftlr_rank(A, i, k)
        a_k_prefix[i, k + 1] = a_k_prefix[i, k] + r
        pair_ranks[i] += r * b_row_ranks[k]
    end
    b_total_rank = sum(b_row_ranks)
    b_col_prefix = _compressed_ftlr_prefix(b_col_ranks)
    row_heights = [length(_compressed_ftlr_axis_range(A, i, 1)) for i in 1:qm]
    col_widths = [length(_compressed_ftlr_axis_range(B, j, 2)) for j in 1:qn]
    col_prefix = _compressed_ftlr_prefix(col_widths)
    Tbytes = sizeof(eltype(A))
    right = _compressed_ftlr_right_valid(A, B) ?
        [pair_ranks[i] == 0 ? 0 :
         (pair_ranks[i] + col_prefix[end] * a_k_prefix[i, end]) * Tbytes for i in 1:qm] : nothing
    left = _compressed_ftlr_left_valid(A, B) ?
        [pair_ranks[i] == 0 ? 0 :
         (pair_ranks[i] + row_heights[i] * b_total_rank) * Tbytes for i in 1:qm] : nothing
    (right === nothing && left === nothing) && throw(ArgumentError(
        "CompressedFTLR needs row-packed B outer factors and either row-packed A outer factors or column-packed B inner factors"))
    row_bytes = [min(right === nothing ? typemax(Int) : right[i],
                     left === nothing ? typemax(Int) : left[i]) for i in 1:qm]
    right_flops = right === nothing ? nothing :
        [sum(col_widths[j] * sum(_compressed_ftlr_rank(A, i, k) *
                                  _compressed_ftlr_rank(B, k, j)
                                  for k in 1:qk)
             for j in 1:qn) +
         row_heights[i] * col_prefix[end] * a_k_prefix[i, end]
         for i in 1:qm]
    left_flops = left === nothing ? nothing :
        [row_heights[i] * pair_ranks[i] +
         sum(row_heights[i] * col_widths[j] * b_col_ranks[j] for j in 1:qn)
         for i in 1:qm]
    maximum_bytes = min(right === nothing ? typemax(Int) : sum(right),
                        left === nothing ? typemax(Int) : sum(left))
    profile = RaggedWorkspaceProfile(
        row_bytes, right, left, right_flops, left_flops,
        right === nothing ? nothing : _compressed_ftlr_prefix(right),
        left === nothing ? nothing : _compressed_ftlr_prefix(left),
        right_flops === nothing ? nothing : _compressed_ftlr_prefix(right_flops),
        left_flops === nothing ? nothing : _compressed_ftlr_prefix(left_flops),
        isempty(row_bytes) ? 0 : maximum(row_bytes), maximum_bytes,
    )
    return CompressedFTLRRankPlan(a_k_prefix, b_row_ranks, b_col_ranks, b_col_k_prefix,
                        b_col_prefix, b_first_active_col, pair_ranks, b_total_rank,
                        row_heights, col_widths, col_prefix,
                        profile)
end

_compressed_ftlr_workspace_profile(A, B) = _compressed_ftlr_rank_plan(A, B).profile

function gemm_minimum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    return _compressed_ftlr_rank_plan(logical_operand(A, transA), logical_operand(B, transB)).profile.minimum
end

function gemm_maximum_workspace_bytes(A::CompressedFTLRMatrix, B::CompressedFTLRMatrix;
                                      transA::Char='N', transB::Char='N')
    return _compressed_ftlr_rank_plan(logical_operand(A, transA), logical_operand(B, transB)).profile.maximum
end

"""Contiguous output-row runs greedily packed under an exact ragged budget."""
function _compressed_ftlr_row_runs(profile::RaggedWorkspaceProfile, budget::Int)
    budget >= profile.minimum || throw(ArgumentError(
        "workspace has $budget bytes; at least $(profile.minimum) bytes are required"))
    runs = RaggedRowRun[]
    i = 1
    while i <= length(profile.row_bytes)
        j = i - 1
        while j < length(profile.row_bytes) && _compressed_ftlr_select_fold(profile, i:(j + 1), budget) !== nothing
            j += 1
        end
        # A zero-work row is allowed even for a zero byte budget.
        j >= i || (j = i)
        fold = _compressed_ftlr_select_fold(profile, i:j, budget)
        fold === nothing && throw(ArgumentError("workspace cannot schedule CompressedFTLR row $i"))
        push!(runs, RaggedRowRun(i:j, fold))
        i = j + 1
    end
    return runs
end

function _compressed_ftlr_select_fold(profile::RaggedWorkspaceProfile, rows::UnitRange{Int}, budget::Int)
    right_bytes = profile.right_byte_prefix === nothing ? typemax(Int) :
        _compressed_ftlr_range_total(profile.right_byte_prefix, rows)
    left_bytes = profile.left_byte_prefix === nothing ? typemax(Int) :
        _compressed_ftlr_range_total(profile.left_byte_prefix, rows)
    right_ok = right_bytes <= budget
    left_ok = left_bytes <= budget
    !right_ok && !left_ok && return nothing
    right_ok && !left_ok && return :right
    left_ok && !right_ok && return :left
    right_flops = _compressed_ftlr_range_total(profile.right_flop_prefix, rows)
    left_flops = _compressed_ftlr_range_total(profile.left_flop_prefix, rows)
    return right_flops <= left_flops ? :right : :left
end

function _prepare_compressed_ftlr_workspace(A, workspace, profile::RaggedWorkspaceProfile)
    T = eltype(A)
    ws = if workspace isa Integer
        bytes = Int(workspace)
        bytes >= profile.minimum || throw(ArgumentError(
            "workspace has $bytes bytes; at least $(profile.minimum) bytes are required"))
        DenseGemmWorkspace(_compressed_ftlr_parent(A), bytes)
    elseif workspace isa DenseGemmWorkspace
        eltype(workspace) === T || throw(ArgumentError(
            "workspace element type $(eltype(workspace)) does not match operand type $T"))
        typeof(get_backend(workspace.storage)) === typeof(get_backend(A)) ||
            throw(ArgumentError("workspace and operands must use the same backend"))
        sizeof(workspace) >= profile.minimum || throw(ArgumentError(
            "workspace has $(sizeof(workspace)) bytes; at least $(profile.minimum) bytes are required"))
        workspace
    else
        throw(ArgumentError("workspace must be an integer byte count or DenseGemmWorkspace"))
    end
    budget = min(sizeof(ws), profile.maximum)
    arena = DenseGemmArena(view(ws.storage, :), 1)
    return ws, arena, budget, profile
end
