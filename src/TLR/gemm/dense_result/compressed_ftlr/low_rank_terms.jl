function _validate_compressed_ftlr_layout(A, B)
    nominal_tile_size(A, 2) == nominal_tile_size(B, 1) ||
        throw(DimensionMismatch("CompressedFTLR contraction tile dimensions must match"))
    compressed_ftlr_outer_order(B) isa TileRowMajor ||
        throw(ArgumentError("CompressedFTLR Stage-1 fusion requires the logical B outer factors to be tile-row-major"))
    (_compressed_ftlr_right_valid(A, B) || _compressed_ftlr_left_valid(A, B)) ||
        throw(ArgumentError("CompressedFTLR needs a FoldRight A-U row stack or a FoldLeft B-Z column stack"))
    _, qk = grid_size(A)
    qk == grid_size(B)[1] || throw(DimensionMismatch("CompressedFTLR contraction grids do not match"))
    @inbounds for k in 1:qk
        length(_compressed_ftlr_axis_range(A, k, 2)) == length(_compressed_ftlr_axis_range(B, k, 1)) ||
            throw(DimensionMismatch("CompressedFTLR contraction tile $k has incompatible tail extents"))
    end
    return nothing
end

function _compressed_ftlr_gemm!(C::AbstractMatrix, A, B; workspace, alpha, beta, compute)
    T = eltype(A)
    (typeof(get_backend(A)) === typeof(get_backend(B)) &&
     typeof(get_backend(A)) === typeof(get_backend(C))) ||
        throw(ArgumentError("CompressedFTLR operands and output must use the same backend"))
    supports_grouped_gemm(get_backend(A)) || throw(ArgumentError(
        "CompressedFTLR dense GEMM currently requires CUDA grouped GEMM"))
    T === Core.BFloat16 && !supports_bfloat16_grouped_gemm(get_backend(A)) &&
        throw(ArgumentError("CompressedFTLR BF16 grouped GEMMEx requires an NVIDIA SM80 or newer device"))
    size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("C must be size(A,1) × size(B,2)"))
    _validate_compressed_ftlr_layout(A, B)
    validate_tlr_gemm_precision(get_backend(A), T, eltype(C), compute)
    # cuBLAS grouped GEMMEx accepts the FP16/FP32-compute case, but (unlike
    # ordinary GEMMEx) the grouped entry point rejects an FP16 -> FP32 output
    # signature on current CUDA. Every CompressedFTLR stage must stay grouped, so keep
    # storage homogeneous until that API capability becomes available.
    eltype(C) === T || throw(ArgumentError(
        "CompressedFTLR grouped GEMMEx currently requires output storage to match operand storage; " *
        "got $T × $T → $(eltype(C))",
    ))
    if maxrank(A) == 0 || maxrank(B) == 0
        return _scale_output!(C, beta)
    end
    plan = _compressed_ftlr_rank_plan(A, B)
    _, arena, budget, profile = _prepare_compressed_ftlr_workspace(A, workspace, plan.profile)
    for run in _compressed_ftlr_row_runs(profile, budget)
        if run.fold === :right
            _execute_compressed_ftlr_foldright_run!(C, A, B, plan, run.rows, alpha, beta, compute, arena)
        else
            _execute_compressed_ftlr_foldleft_run!(C, A, B, plan, run.rows, alpha, beta, compute, arena)
        end
    end
    return C
end
