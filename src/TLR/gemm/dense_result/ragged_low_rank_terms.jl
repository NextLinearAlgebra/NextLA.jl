function _validate_bclr_layout(A, B)
    nominal_tile_size(A, 2) == nominal_tile_size(B, 1) ||
        throw(DimensionMismatch("BCLR contraction tile dimensions must match"))
    bclr_outer_order(B) isa TileRowMajor ||
        throw(ArgumentError("BCLR Stage-1 fusion requires the logical B outer factors to be tile-row-major"))
    (_bclr_right_valid(A, B) || _bclr_left_valid(A, B)) ||
        throw(ArgumentError("BCLR needs a FoldRight A-U row stack or a FoldLeft B-Z column stack"))
    return nothing
end

function _bclr_gemm!(C::AbstractMatrix, A, B; workspace, alpha, beta, compute)
    T = eltype(A)
    (typeof(get_backend(A)) === typeof(get_backend(B)) &&
     typeof(get_backend(A)) === typeof(get_backend(C))) ||
        throw(ArgumentError("BCLR operands and output must use the same backend"))
    supports_grouped_gemm(get_backend(A)) || throw(ArgumentError(
        "BCLR dense GEMM currently requires CUDA grouped GEMM"))
    size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("C must be size(A,1) × size(B,2)"))
    _validate_bclr_layout(A, B)
    validate_tlr_gemm_precision(get_backend(A), T, eltype(C), compute)
    # cuBLAS grouped GEMMEx accepts the FP16/FP32-compute case, but (unlike
    # ordinary GEMMEx) the grouped entry point rejects an FP16 -> FP32 output
    # signature on current CUDA. Every BCLR stage must stay grouped, so keep
    # storage homogeneous until that API capability becomes available.
    eltype(C) === T || throw(ArgumentError(
        "BCLR grouped GEMMEx currently requires output storage to match operand storage; " *
        "got $T × $T → $(eltype(C))",
    ))
    if maxrank(A) == 0 || maxrank(B) == 0
        return _scale_output!(C, beta)
    end
    plan = _bclr_rank_plan(A, B)
    _, arena, budget, profile = _prepare_bclr_workspace(A, workspace, plan.profile)
    for run in _bclr_row_runs(profile, budget)
        if run.fold === :right
            _execute_bclr_foldright_run!(C, A, B, plan, run.rows, alpha, beta, compute, arena)
        else
            _execute_bclr_foldleft_run!(C, A, B, plan, run.rows, alpha, beta, compute, arena)
        end
    end
    return C
end
