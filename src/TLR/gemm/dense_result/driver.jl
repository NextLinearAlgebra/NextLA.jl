# gemm! entry points for the dense-output TLR GEMM: materializes a dense
# C from compressed full-grid factors and optional dense diagonal storage.

@inline function _validate_logical_gemm(
    C, LA::AbstractTLRMatrix, LB::AbstractTLRMatrix)
    size(LA, 2) == size(LB, 1) ||
        throw(DimensionMismatch("inner dimensions must match: size(op(A),2) == size(op(B),1)"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("op(A)'s column tile size must equal op(B)'s row tile size (contraction tiling)"))
    return nothing
end

"""
    gemm!(C, A, B; workspace,
          alpha=true, beta=false,
          transA='N', transB='N', compute=nothing) -> C

Compute `C := alpha·(op(A)·op(B)) + beta·C` for dense-diagonal TLR matrices.
The compressed off-diagonal product uses the common three-stage grouped
lowering; off-diagonal/diagonal cross terms and the diagonal product are added
as grouped second-pass updates.
`compute` selects the accumulation mode; when omitted it defaults to `Float32` for
`Float16` operands and otherwise to the operand type. `alpha` and `beta` are
converted to that compute type.
"""
function gemm!(C::AbstractMatrix, A::TLRMatrix{BackendT,T}, B::TLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char=('N'), transB::Char=('N'), compute=nothing) where {BackendT,T}
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    _validate_logical_gemm(C, LA, LB)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    backend = get_backend(A)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    validate_tlr_gemm_storage(LA, mode; name="compressed left operand")
    nominal_tile_size(LA, 2) == nominal_tile_size(LB, 1) ||
        throw(DimensionMismatch("TLR contraction tile dimensions must match"))

    ScalarT = gemm_compute_type(mode)
    α = ScalarT(alpha)
    β = ScalarT(beta)
    ws = workspace isa DenseGemmWorkspace ? workspace :
         DenseGemmWorkspace(A, Int(workspace))
    required = gemm_minimum_workspace_bytes(A, B; transA, transB)
    sizeof(ws) >= required || throw(ArgumentError(
        "workspace has $(sizeof(ws)) bytes; at least $required bytes are required"))
    gemm!(C, offdiagonal(A), offdiagonal(B);
          workspace=ws, alpha=α, beta=β, transA, transB, compute=mode)
    return _tlr_add_diagonal_terms!(C, A, B, ws, α, transA, transB, mode)
end

"""
    gemm!(C, A::CompressedFTLRMatrix, B::CompressedFTLRMatrix; workspace, alpha=true, beta=false)

Exact-rank CompressedFTLR dense accumulation on CPU and CUDA. Supports nominal
grids with trailing boundary tiles and logical `N/T` operands, using the common
grouped interface for all three stages and selecting FoldRight/FoldLeft from
packed layouts.
"""
function gemm!(C::AbstractMatrix, A::CompressedFTLRMatrix{BackendT,T}, B::CompressedFTLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char='N', transB::Char='N', compute=nothing, analysis=nothing) where {BackendT,T}
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    ScalarT = gemm_compute_type(mode)
    α = ScalarT(alpha)
    β = ScalarT(beta)
    if analysis === nothing
        # One-shot path: build the symbolic analysis, execute once, discard it.
        # This shares the prepared-descriptor lowering instead of maintaining a
        # second implementation that rebuilt every cuBLAS descriptor per row run
        # (measured ~38x slower on H100 for repeated calls, and identical cost
        # for a single call since analysis does the same work once).
        _validate_compressed_ftlr_gemm(C, LA, LB, mode)
        # Rank vectors are intentionally mutable so a reserved-capacity TLR
        # container can still be populated through the legacy in-place API.
        # Do not trust the construction-time cached maximum for this fast path.
        (all(iszero, ranks(A)) || all(iszero, ranks(B))) &&
            return _scale_output!(C, β)
        ws = workspace isa DenseGemmWorkspace ? workspace :
             DenseGemmWorkspace(A, Int(workspace))
        one_shot = analyze_compressed_gemm(
            C, A, B; workspace=ws, transA, transB, compute=mode)
        try
            return _execute_compressed_gemm_analysis!(
                one_shot, C, A, B, ws, α, β, transA, transB, mode)
        finally
            close(one_shot)
        end
    elseif analysis isa CompressedGemmAnalysis
        return _execute_compressed_gemm_analysis!(
            analysis, C, A, B, workspace, α, β, transA, transB, mode)
    end
    throw(ArgumentError("analysis must be nothing or CompressedGemmAnalysis"))
end

@inline function _validate_dense_backend(C, compressed, dense)
    backend = get_backend(compressed)
    typeof(get_backend(dense)) === typeof(backend) &&
        typeof(get_backend(C)) === typeof(backend) ||
        throw(ArgumentError(
            "compressed operand, dense operand, and output must use the same backend"))
    return backend
end

"""
    gemm!(C, A::TLRMatrix, B::AbstractMatrix; ...)

Compute `C := alpha*op(A)*op(B) + beta*C` for a dense-diagonal TLR left
operand. The compressed off-diagonal part uses the fixed FoldRight two-stage
lowering; the dense diagonal is then added as independent block GEMMs.
"""
function gemm!(C::AbstractMatrix, A::TLRMatrix{BackendT,T},
    B::AbstractMatrix{T};
    workspace, alpha=true, beta=false,
    transA::Char='N', transB::Char='N', compute=nothing) where {BackendT,T}
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    ScalarT = gemm_compute_type(mode)
    alpha_value = ScalarT(alpha)
    gemm!(C, offdiagonal(A), B;
          workspace, alpha=alpha_value, beta=ScalarT(beta), transA, transB,
          compute=mode)
    return _tlr_diag_times_dense!(
        C, transA == 'T' ? transpose(A) : A, logical_dense_operand(B, transB),
        alpha_value, mode)
end

"""
    gemm!(C, A::AbstractMatrix, B::TLRMatrix; ...)

Compute `C := alpha*op(A)*op(B) + beta*C` for a dense-diagonal TLR right
operand. The compressed off-diagonal part uses the fixed FoldLeft two-stage
lowering; the dense diagonal is then added as independent block GEMMs.
"""
function gemm!(C::AbstractMatrix, A::AbstractMatrix{T},
    B::TLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false,
    transA::Char='N', transB::Char='N', compute=nothing) where {BackendT,T}
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    ScalarT = gemm_compute_type(mode)
    alpha_value = ScalarT(alpha)
    gemm!(C, A, offdiagonal(B);
          workspace, alpha=alpha_value, beta=ScalarT(beta), transA, transB,
          compute=mode)
    return _dense_times_tlr_diag!(
        C, logical_dense_operand(A, transA), transB == 'T' ? transpose(B) : B,
        alpha_value, mode)
end

function _execute_one_shot_two_stage!(
    C, A, B, workspace, alpha, beta, transA, transB, mode)
    analysis = analyze_compressed_gemm(
        C, A, B; workspace, transA, transB, compute=mode)
    try
        return _execute_compressed_mixed_analysis!(
            analysis, C, A, B, workspace, alpha, beta,
            transA, transB, mode)
    finally
        close(analysis)
    end
end

"""
    gemm!(C, A::CompressedFTLRMatrix, B::AbstractMatrix; ...)

Compute `C := alpha·op(A)·op(B) + beta·C`. The first stage fuses
`Vᵢₖ′Bₖ` contractions across an output-column run; the second stage applies
the corresponding row-packed `Uᵢₖ` stack and accumulates into dense `C`.
"""
function gemm!(C::AbstractMatrix, A::CompressedFTLRMatrix{BackendT,T},
    B::AbstractMatrix{T};
    workspace, alpha=true, beta=false, analysis=nothing,
    transA::Char='N', transB::Char='N', compute=nothing) where {BackendT,T}
    LA = transA == 'T' ? transpose(A) : A
    LB = logical_dense_operand(B, transB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    backend = _validate_dense_backend(C, A, B)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    validate_tlr_gemm_storage(LA, mode; name="compressed left operand")
    ScalarT = gemm_compute_type(mode)
    if analysis !== nothing
        analysis isa CompressedMixedGemmAnalysis || throw(ArgumentError(
            "analysis must be nothing or CompressedMixedGemmAnalysis"))
        return _execute_compressed_mixed_analysis!(
            analysis, C, A, B, workspace, ScalarT(alpha), ScalarT(beta),
            transA, transB, mode)
    end
    ws, arena, budget = _prepare_dense_result_workspace(A, workspace)
    plan = _two_stage_rank_plan(LA, :right)
    if budget < _two_stage_workspace_floor(plan, T)
        return _compressed_dense_gemm_sequential!(
            C, LA, LB, ScalarT(alpha), ScalarT(beta), budget, mode, arena)
    end
    return _execute_one_shot_two_stage!(
        C, A, B, ws, ScalarT(alpha), ScalarT(beta), transA, transB, mode)
end

"""
    gemm!(C, A::AbstractMatrix, B::CompressedFTLRMatrix; ...)

Compute `C := alpha·op(A)·op(B) + beta·C`. The mirrored two-stage lowering
fuses dense-panel-times-`Uₖⱼ` contractions over an output-row run before applying
the column-packed `Vₖⱼ` stack to dense `C`.
"""
function gemm!(C::AbstractMatrix, A::AbstractMatrix{T},
    B::CompressedFTLRMatrix{BackendT,T};
    workspace, alpha=true, beta=false, analysis=nothing,
    transA::Char='N', transB::Char='N', compute=nothing) where {BackendT,T}
    LA = logical_dense_operand(A, transA)
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    backend = _validate_dense_backend(C, B, A)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    validate_tlr_gemm_precision(backend, T, eltype(C), mode)
    validate_tlr_gemm_storage(LB, mode; name="compressed right operand")
    ScalarT = gemm_compute_type(mode)
    if analysis !== nothing
        analysis isa CompressedMixedGemmAnalysis || throw(ArgumentError(
            "analysis must be nothing or CompressedMixedGemmAnalysis"))
        return _execute_compressed_mixed_analysis!(
            analysis, C, A, B, workspace, ScalarT(alpha), ScalarT(beta),
            transA, transB, mode)
    end
    ws, arena, budget = _prepare_dense_result_workspace(B, workspace)
    plan = _two_stage_rank_plan(LB, :left)
    if budget < _two_stage_workspace_floor(plan, T)
        return _dense_compressed_gemm_sequential!(
            C, LA, LB, ScalarT(alpha), ScalarT(beta), budget, mode, arena)
    end
    return _execute_one_shot_two_stage!(
        C, A, B, ws, ScalarT(alpha), ScalarT(beta), transA, transB, mode)
end
