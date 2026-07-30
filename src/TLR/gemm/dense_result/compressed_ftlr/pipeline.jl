"""One of two fixed-capacity row-run metadata buffers."""
mutable struct CompressedGemmPipelineSlot
    stage1::AbstractReusableGroupedGemmSlot
    stage2::AbstractReusableGroupedGemmSlot
    stage3::AbstractReusableGroupedGemmSlot
    upload_event::Any
    completion_event::Any
    busy::Bool
end

"""
Internal workspace for `_gemm_compressed_pipelined!`. Construction allocates only
capacity: no row schedule, fold choice, group, or operand pointer is prepared
until the GEMM call itself.
"""
mutable struct CompressedGemmPipelineWorkspace{W,ModeT,BackendT}
    numerical::W
    slots::NTuple{2,CompressedGemmPipelineSlot}
    compute::ModeT
    backend::BackendT
    transA::Char
    transB::Char
    grid::NTuple{3,Int}
    output_size::NTuple{2,Int}
    max_rows_per_run::Int
    closed::Bool
end

function _destroy_compressed_pipeline_slot!(slot::CompressedGemmPipelineSlot)
    destroy_reusable_grouped_gemm!(slot.stage3)
    destroy_reusable_grouped_gemm!(slot.stage2)
    destroy_reusable_grouped_gemm!(slot.stage1)
    return slot
end

function _destroy_compressed_pipeline_workspace!(pipeline::CompressedGemmPipelineWorkspace)
    pipeline.closed && return pipeline
    pipeline.closed = true
    for slot in pipeline.slots
        _destroy_compressed_pipeline_slot!(slot)
    end
    return pipeline
end

Base.close(pipeline::CompressedGemmPipelineWorkspace) =
    _destroy_compressed_pipeline_workspace!(pipeline)

function _new_compressed_pipeline_slot(backend, T, mode, stage_caps)
    built = AbstractReusableGroupedGemmSlot[]
    try
        stage1 = create_reusable_grouped_gemm_slot(
            backend, stage_caps[1], stage_caps[1], T, T, mode)
        push!(built, stage1)
        stage2 = create_reusable_grouped_gemm_slot(
            backend, stage_caps[2], stage_caps[2], T, T, mode)
        push!(built, stage2)
        stage3 = create_reusable_grouped_gemm_slot(
            backend, stage_caps[3], stage_caps[3], T, T, mode)
        push!(built, stage3)
        return CompressedGemmPipelineSlot(
            stage1, stage2, stage3, create_event(backend), create_event(backend), false)
    catch
        for descriptor in reverse(built)
            destroy_reusable_grouped_gemm!(descriptor)
        end
        rethrow()
    end
end

"""
    _compressed_gemm_pipeline_workspace(C, A, B; workspace,
                                        max_rows_per_run=4, transA='N', transB='N',
                                        compute=nothing)

Allocate two empty run-metadata slots. This is intentionally internal: poster
benchmarks may call it explicitly, but it is not part of NextLA's supported API.
"""
function _compressed_gemm_pipeline_workspace(
    C::AbstractMatrix,
    A::CompressedFTLRMatrix{BackendT,T},
    B::CompressedFTLRMatrix{BackendT,T};
    workspace,
    max_rows_per_run::Int=4,
    transA::Char='N',
    transB::Char='N',
    compute=nothing,
) where {BackendT,T}
    workspace isa DenseGemmWorkspace || throw(ArgumentError(
        "compressed pipeline requires a reusable DenseGemmWorkspace"))
    max_rows_per_run > 0 || throw(ArgumentError("max_rows_per_run must be positive"))
    opA = _normalize_tlr_op(transA); opB = _normalize_tlr_op(transB)
    LA = logical_operand(A, opA); LB = logical_operand(B, opB)
    size(LA, 2) == size(LB, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(C) == (size(LA, 1), size(LB, 2)) ||
        throw(DimensionMismatch("C must be size(op(A),1) × size(op(B),2)"))
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    _validate_compressed_ftlr_gemm(C, LA, LB, mode)
    eltype(workspace) === T || throw(ArgumentError("pipeline numerical workspace type mismatch"))
    get_backend(workspace.storage) == get_backend(A) ||
        throw(ArgumentError("pipeline numerical workspace backend mismatch"))
    qm, qk = grid_size(LA); qkB, qn = grid_size(LB)
    qk == qkB || throw(DimensionMismatch("contraction tile grids do not match"))
    rows = min(max_rows_per_run, qm)
    stage_caps = (rows * qk, rows * qk * qn, rows * qn)
    backend = get_backend(A)
    first_slot = _new_compressed_pipeline_slot(backend, T, mode, stage_caps)
    second_slot = try
        _new_compressed_pipeline_slot(backend, T, mode, stage_caps)
    catch
        _destroy_compressed_pipeline_slot!(first_slot)
        rethrow()
    end
    pipeline = CompressedGemmPipelineWorkspace(
        workspace, (first_slot, second_slot), mode, backend, opA, opB,
        (qm, qk, qn), size(C), rows, false)
    finalizer(pipeline) do object
        try
            _destroy_compressed_pipeline_workspace!(object)
        catch
        end
    end
    return pipeline
end

function _refresh_compressed_pipeline_slot!(slot::CompressedGemmPipelineSlot,
                                             tasks::CompressedFTLRRunTasks,
                                             mode, backend, prep_stream)
    with_stream(backend, prep_stream) do
        refresh_reusable_grouped_gemm!(slot.stage1, tasks.stage1, mode)
        refresh_reusable_grouped_gemm!(slot.stage2, tasks.stage2, mode)
        refresh_reusable_grouped_gemm!(slot.stage3, tasks.stage3, mode)
        record_event!(backend, slot.upload_event, prep_stream)
    end
    return slot
end

function _submit_compressed_pipeline_slot!(
    slot::CompressedGemmPipelineSlot, tasks::CompressedFTLRRunTasks,
    C, A, alpha, beta, backend, execution_stream)
    with_stream(backend, execution_stream) do
        wait_event!(backend, slot.upload_event, execution_stream)
        tasks.tdata === nothing || fill!(tasks.tdata, zero(eltype(A)))
        @inbounds for (rows, cols) in tasks.scale_targets
            _scale_output!(view(C, rows, cols), beta)
        end
        submit_reusable_grouped_gemm!(slot.stage1)
        submit_reusable_grouped_gemm!(slot.stage2)
        submit_reusable_grouped_gemm!(slot.stage3, alpha, beta)
        record_event!(backend, slot.completion_event, execution_stream)
    end
    slot.busy = true
    return slot
end

function _validate_compressed_pipeline_workspace(
    pipeline::CompressedGemmPipelineWorkspace, C, LA, LB, workspace,
    transA, transB, mode)
    pipeline.closed && throw(ArgumentError("CompressedGemmPipelineWorkspace has been closed"))
    workspace === pipeline.numerical ||
        throw(ArgumentError("pipeline is bound to a different numerical workspace"))
    _normalize_tlr_op(transA) == pipeline.transA ||
        throw(ArgumentError("pipeline transA does not match its construction"))
    _normalize_tlr_op(transB) == pipeline.transB ||
        throw(ArgumentError("pipeline transB does not match its construction"))
    typeof(mode) === typeof(pipeline.compute) ||
        throw(ArgumentError("pipeline compute policy does not match its construction"))
    qm, qk = grid_size(LA); qkB, qn = grid_size(LB)
    (qm, qk, qn) == pipeline.grid && qk == qkB ||
        throw(ArgumentError("pipeline tile grid does not match the operands"))
    size(C) == pipeline.output_size ||
        throw(ArgumentError("pipeline output shape does not match its construction"))
    return nothing
end

"""
    _gemm_compressed_pipelined!(C, A, B; pipeline, workspace=pipeline.numerical, ...)

Experimental honest single-call execution: every schedule/group is built during
this call. While the GPU executes row-run `r`, the host prepares row-run `r+1`
in the other fixed-capacity metadata slot.
"""
function _gemm_compressed_pipelined!(
    C::AbstractMatrix,
    A::CompressedFTLRMatrix{BackendT,T},
    B::CompressedFTLRMatrix{BackendT,T};
    pipeline::CompressedGemmPipelineWorkspace,
    workspace=pipeline.numerical,
    alpha=true,
    beta=false,
    transA::Char='N',
    transB::Char='N',
    compute=nothing,
) where {BackendT,T}
    LA = logical_operand(A, transA); LB = logical_operand(B, transB)
    mode = compute === nothing ? default_gemm_compute_mode(T) : gemm_compute_mode(compute)
    _validate_compressed_ftlr_gemm(C, LA, LB, mode)
    _validate_compressed_pipeline_workspace(
        pipeline, C, LA, LB, workspace, transA, transB, mode)
    scalar_type = gemm_compute_type(mode)
    α = scalar_type(alpha); β = scalar_type(beta)
    plan = _compressed_ftlr_rank_plan(LA, LB)
    _, arena, budget, profile =
        _prepare_compressed_ftlr_workspace(LA, workspace, plan.profile)
    runs = _compressed_ftlr_row_runs_limited(
        profile, budget, pipeline.max_rows_per_run)
    backend = pipeline.backend
    execution_stream = workspace.streams[1]
    prep_stream = workspace.streams[2]
    sync_streams_with_default(backend, workspace.streams)

    _with_grouped_host_pointer_mode(backend) do
        for (run_index, run) in enumerate(runs)
            slot = pipeline.slots[isodd(run_index) ? 1 : 2]
            slot.busy && sync_event(backend, slot.completion_event)
            tasks = if run.fold === :right
                _build_compressed_ftlr_foldright_run(
                    C, LA, LB, plan, run.rows, α, β, arena)
            else
                _build_compressed_ftlr_foldleft_run(
                    C, LA, LB, plan, run.rows, α, β, arena)
            end
            _refresh_compressed_pipeline_slot!(slot, tasks, mode, backend, prep_stream)
            _submit_compressed_pipeline_slot!(
                slot, tasks, C, LA, α, β, backend, execution_stream)
        end
    end
    sync_stream(backend, execution_stream)
    return C
end
