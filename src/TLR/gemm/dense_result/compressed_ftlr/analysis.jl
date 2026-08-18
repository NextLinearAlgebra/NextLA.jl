"""
    CompressedGemmAnalysis

Explicit symbolic metadata for `CompressedFTLRMatrix × CompressedFTLRMatrix → dense`.
The object owns device pointer tables and is bound to the output, operands,
workspace, logical operations, compute policy, and rank metadata used to create it.
Factor values and numerical scalars may be changed between numerical calls.
"""
mutable struct CompressedGemmAnalysis{CT,AT,BT,WT,LAT,LBT,ModeT,PlanT,RAT,RBT}
    C::CT
    A::AT
    B::BT
    workspace::WT
    logical_A::LAT
    logical_B::LBT
    transA::Char
    transB::Char
    compute::ModeT
    plan::PlanT
    runs::Vector{PreparedDenseResultRun}
    # Snapshots in the operands' own rank type: the guard runs on every
    # numerical call, and converting to `Int` there would allocate four
    # vectors per call purely to compare them. These must be COPIES -- holding
    # the operands' live vectors would make the comparison vacuous.
    A_ranks::RAT
    B_ranks::RBT
    workspace_bytes::Int
    has_fallback::Bool
    closed::Bool
end

function _destroy_compressed_gemm_analysis!(analysis::CompressedGemmAnalysis)
    return _close_dense_result_analysis!(analysis)
end

Base.close(analysis::CompressedGemmAnalysis) = _destroy_compressed_gemm_analysis!(analysis)

"""
    analyze_compressed_gemm(C, A, B; workspace, transA='N', transB='N', compute=nothing)

Perform the explicit symbolic phase for a compressed dense-output GEMM. `workspace`
must be a reusable `DenseGemmWorkspace`; allocation of numerical storage is kept
separate from symbolic analysis so benchmark timing boundaries remain explicit.
"""
function analyze_compressed_gemm(
    C::AbstractMatrix,
    A::CompressedFTLRMatrix{BackendT,T},
    B::CompressedFTLRMatrix{BackendT,T};
    workspace,
    transA::Char='N',
    transB::Char='N',
    compute=nothing,
) where {BackendT,T}
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "symbolic compressed GEMM analysis requires a reusable DenseGemmWorkspace"))
    LA = transA == 'T' ? transpose(A) : A
    LB = transB == 'T' ? transpose(B) : B
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    _validate_compressed_ftlr_gemm(C, LA, LB, mode)

    plan = _compressed_ftlr_rank_plan(LA, LB)
    ws, arena, budget, profile =
        _prepare_compressed_ftlr_workspace(LA, LB, plan, workspace)
    ws === workspace || error("internal error: symbolic analysis replaced its workspace")
    scalar_type = gemm_compute_type(mode)
    placeholder_alpha = one(scalar_type)
    placeholder_beta = zero(scalar_type)
    # Subdivides into column blocks only when a full-width schedule does not fit;
    # otherwise this is exactly the whole-width row-run schedule.
    schedule = _compressed_ftlr_column_schedule(plan, LA, LB, profile, budget)
    prepared_runs = _prepare_dense_result_runs(schedule, mode) do run
        _build_compressed_ftlr_run(
                C, LA, LB, plan, run.rows, run.cols, run.fold,
                placeholder_alpha, placeholder_beta, arena)
    end

    analysis = CompressedGemmAnalysis(
        C, A, B, workspace, LA, LB, transA, transB, mode, plan, prepared_runs,
        copy(ranks(A)), copy(ranks(B)),
        sizeof(workspace),
        _dense_result_runs_have_fallback(prepared_runs),
        false)
    finalizer(analysis) do object
        try
            _destroy_compressed_gemm_analysis!(object)
        catch
            # Device teardown may precede Julia object finalization.
        end
    end
    return analysis
end

function _execute_compressed_gemm_analysis!(
    analysis::CompressedGemmAnalysis, C, A, B, workspace,
    alpha, beta, transA, transB, mode)
    _validate_dense_result_analysis_binding(
        analysis, C, A, B, workspace, transA, transB, mode)
    # Same-eltype `==` compares element-wise without materialising a temporary.
    ranks(A) == analysis.A_ranks ||
        throw(ArgumentError("left operand exact ranks changed after symbolic analysis"))
    ranks(B) == analysis.B_ranks ||
        throw(ArgumentError("right operand exact ranks changed after symbolic analysis"))
    backend = get_backend(analysis.logical_A)
    return _execute_prepared_dense_result_runs!(
        analysis.runs, C, backend, eltype(A), alpha, beta,
        analysis.has_fallback)
end
