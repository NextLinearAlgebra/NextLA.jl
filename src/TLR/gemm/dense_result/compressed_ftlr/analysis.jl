"""One row-run whose grouped cuBLAS descriptors have already been uploaded."""
struct PreparedCompressedFTLRRun
    rows::UnitRange{Int}
    fold::Symbol
    stage1::Union{Nothing,AbstractPreparedGroupedGemm}
    stage2::Union{Nothing,AbstractPreparedGroupedGemm}
    stage3::Union{Nothing,AbstractPreparedGroupedGemm}
    tdata::Any
    needs_tdata_zero::Bool
    scale_targets::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
end

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
    runs::Vector{PreparedCompressedFTLRRun}
    # Snapshots in the operands' own rank type: the guard runs on every
    # numerical call, and converting to `Int` there would allocate four
    # vectors per call purely to compare them. These must be COPIES -- holding
    # the operands' live vectors would make the comparison vacuous.
    A_ranks::RAT
    B_ranks::RBT
    A_execution_ranks::RAT
    B_execution_ranks::RBT
    workspace_bytes::Int
    has_fallback::Bool
    closed::Bool
end

@inline function _prepare_compressed_stage(tasks, mode)
    tasks === nothing && return nothing
    return prepare_precision_gemm_grouped(tasks, mode)
end

"""Destroy whichever of a run's three prepared stages were actually built —
the one destruction pattern shared by a partially-failed `_prepare_compressed_run`,
a fully closed `CompressedGemmAnalysis`, and a partially-failed `analyze_compressed_gemm`."""
@inline function _destroy_compressed_stages!(stage1, stage2, stage3)
    stage1 === nothing || destroy_prepared_grouped_gemm!(stage1)
    stage2 === nothing || destroy_prepared_grouped_gemm!(stage2)
    stage3 === nothing || destroy_prepared_grouped_gemm!(stage3)
    return nothing
end
@inline _destroy_compressed_stages!(run::PreparedCompressedFTLRRun) =
    _destroy_compressed_stages!(run.stage1, run.stage2, run.stage3)

function _prepare_compressed_run(tasks::CompressedFTLRRunTasks, rows, fold, mode)
    stage1 = stage2 = stage3 = nothing
    try
        stage1 = _prepare_compressed_stage(tasks.stage1, mode)
        stage2 = _prepare_compressed_stage(tasks.stage2, mode)
        stage3 = _prepare_compressed_stage(tasks.stage3, mode)
    catch
        _destroy_compressed_stages!(stage1, stage2, stage3)
        rethrow()
    end
    return PreparedCompressedFTLRRun(
        rows, fold, stage1, stage2, stage3, tasks.tdata,
        tasks.needs_tdata_zero, tasks.scale_targets)
end

function _destroy_compressed_gemm_analysis!(analysis::CompressedGemmAnalysis)
    analysis.closed && return analysis
    analysis.closed = true
    for run in analysis.runs
        _destroy_compressed_stages!(run)
    end
    return analysis
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
    opA = _normalize_tlr_op(transA)
    opB = _normalize_tlr_op(transB)
    LA = logical_operand(A, opA)
    LB = logical_operand(B, opB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    _validate_compressed_ftlr_gemm(C, LA, LB, mode)

    plan = _compressed_ftlr_rank_plan(LA, LB)
    ws, arena, budget, profile =
        _prepare_compressed_ftlr_workspace(LA, workspace, plan.profile)
    ws === workspace || error("internal error: symbolic analysis replaced its workspace")
    scalar_type = gemm_compute_type(mode)
    placeholder_alpha = one(scalar_type)
    placeholder_beta = zero(scalar_type)
    prepared_runs = PreparedCompressedFTLRRun[]
    # TEMPORARY indirection: selects greedy (default) or the DP scheduler --
    # see compressed_ftlr/schedule_dp.jl. Revert to `_compressed_ftlr_row_runs`
    # when the benchmark picks a winner.
    runs = _compressed_ftlr_schedule(profile, budget)
    sizehint!(prepared_runs, length(runs))
    try
        for run in runs
            tasks = if run.fold === :right
                _build_compressed_ftlr_foldright_run(
                    C, LA, LB, plan, run.rows, placeholder_alpha, placeholder_beta, arena)
            else
                _build_compressed_ftlr_foldleft_run(
                    C, LA, LB, plan, run.rows, placeholder_alpha, placeholder_beta, arena)
            end
            push!(prepared_runs, _prepare_compressed_run(tasks, run.rows, run.fold, mode))
        end
    catch
        for run in prepared_runs
            run.stage1 === nothing || destroy_prepared_grouped_gemm!(run.stage1)
            run.stage2 === nothing || destroy_prepared_grouped_gemm!(run.stage2)
            run.stage3 === nothing || destroy_prepared_grouped_gemm!(run.stage3)
        end
        rethrow()
    end

    analysis = CompressedGemmAnalysis(
        C, A, B, workspace, LA, LB, opA, opB, mode, plan, prepared_runs,
        copy(ranks(A)), copy(ranks(B)),
        copy(execution_ranks(A)), copy(execution_ranks(B)),
        sizeof(workspace),
        any(run -> run.stage1 isa PreparedGroupedGemmBundle ||
                   run.stage2 isa PreparedGroupedGemmBundle ||
                   run.stage3 isa PreparedGroupedGemmBundle, prepared_runs),
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

function _validate_compressed_gemm_analysis(
    analysis::CompressedGemmAnalysis, C, A, B, workspace, transA, transB, mode)
    analysis.closed && throw(ArgumentError("CompressedGemmAnalysis has been closed"))
    C === analysis.C || throw(ArgumentError("analysis is bound to a different output matrix"))
    A === analysis.A || throw(ArgumentError("analysis is bound to a different left operand"))
    B === analysis.B || throw(ArgumentError("analysis is bound to a different right operand"))
    workspace === analysis.workspace ||
        throw(ArgumentError("analysis is bound to a different numerical workspace"))
    sizeof(workspace) == analysis.workspace_bytes ||
        throw(ArgumentError("analysis numerical-workspace capacity has changed"))
    _normalize_tlr_op(transA) == analysis.transA ||
        throw(ArgumentError("analysis transA does not match the numerical call"))
    _normalize_tlr_op(transB) == analysis.transB ||
        throw(ArgumentError("analysis transB does not match the numerical call"))
    typeof(mode) === typeof(analysis.compute) ||
        throw(ArgumentError("analysis compute policy does not match the numerical call"))
    # Same-eltype `==` compares element-wise without materialising a temporary.
    ranks(A) == analysis.A_ranks ||
        throw(ArgumentError("left operand exact ranks changed after symbolic analysis"))
    ranks(B) == analysis.B_ranks ||
        throw(ArgumentError("right operand exact ranks changed after symbolic analysis"))
    execution_ranks(A) == analysis.A_execution_ranks ||
        throw(ArgumentError("left operand execution ranks changed after symbolic analysis"))
    execution_ranks(B) == analysis.B_execution_ranks ||
        throw(ArgumentError("right operand execution ranks changed after symbolic analysis"))
    return nothing
end

@inline function _submit_analysis_stage(stage, backend, manage_pointer_mode, overrides...)
    stage === nothing && return nothing
    if manage_pointer_mode && !(stage isa PreparedGroupedGemmBundle)
        return _with_grouped_host_pointer_mode(backend) do
            precision_gemm_grouped_prepared!(stage, overrides...)
        end
    end
    return precision_gemm_grouped_prepared!(stage, overrides...)
end

function _execute_analysis_runs!(analysis, C, A, alpha, beta, manage_pointer_mode)
    backend = get_backend(analysis.logical_A)
    for run in analysis.runs
        run.needs_tdata_zero && fill!(run.tdata, zero(eltype(A)))
        @inbounds for (rows, cols) in run.scale_targets
            _scale_output!(view(C, rows, cols), beta)
        end
        _submit_analysis_stage(run.stage1, backend, manage_pointer_mode)
        _submit_analysis_stage(run.stage2, backend, manage_pointer_mode)
        _submit_analysis_stage(run.stage3, backend, manage_pointer_mode, alpha, beta)
    end
    return C
end

function _execute_compressed_gemm_analysis!(
    analysis::CompressedGemmAnalysis, C, A, B, workspace,
    alpha, beta, transA, transB, mode)
    _validate_compressed_gemm_analysis(
        analysis, C, A, B, workspace, transA, transB, mode)
    backend = get_backend(analysis.logical_A)
    if analysis.has_fallback
        return _execute_analysis_runs!(analysis, C, A, alpha, beta, true)
    end
    return _with_grouped_host_pointer_mode(backend) do
        _execute_analysis_runs!(analysis, C, A, alpha, beta, false)
    end
end
