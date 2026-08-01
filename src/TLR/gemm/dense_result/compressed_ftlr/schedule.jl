"""Host-only rank metadata plus the fold-cost profile derived from it, for one
CompressedFTLR GEMM. See `rank_metadata.jl` for the pure rank facts and
`fold_cost.jl` for how they turn into FoldRight/FoldLeft costs."""
struct CompressedFTLRRankPlan
    a_k_prefix::Matrix{Int}      # (logical i, prefix through logical k)
    b_row_ranks::Vector{Int}     # σ_k = Σ_j rB_kj
    b_col_ranks::Vector{Int}     # γ_j = Σ_k rB_kj
    b_col_k_prefix::Matrix{Int}  # (logical j, prefix through logical k)
    b_row_k_prefix::Matrix{Int} # (logical k, prefix through logical j): Σ_{j'<j} rB_kj'
    b_col_prefix::Vector{Int}    # prefix of γ_j across logical j
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

function _compressed_ftlr_rank_plan(A, B)
    meta = _compressed_ftlr_rank_metadata(A, B)
    profile = _compressed_ftlr_fold_cost(meta, A, B)
    return CompressedFTLRRankPlan(meta.a_k_prefix, meta.b_row_ranks, meta.b_col_ranks,
                        meta.b_col_k_prefix, meta.b_row_k_prefix, meta.b_col_prefix,
                        meta.pair_ranks, meta.b_total_rank,
                        meta.output_row_heights, meta.output_col_widths, meta.output_col_prefix,
                        profile)
end

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
