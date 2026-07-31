"""
    GroupedGemmTask(transA, transB, alpha, A, B, beta, C)

One heterogeneous GEMM member for a CUDA grouped-GEMM submission.  Unlike
`gemm_batched!`, members are allowed to have different `m`, `n`, and `k`.
The task owns no storage; its matrices must remain live through the call.
"""
struct GroupedGemmTask{AT<:AbstractMatrix,BT<:AbstractMatrix,CT<:AbstractMatrix,S}
    transA::Char
    transB::Char
    alpha::S
    A::AT
    B::BT
    beta::S
    C::CT
end

@inline GroupedGemmTask(transA::Char, transB::Char, alpha, A::AbstractMatrix,
                         B::AbstractMatrix, beta, C::AbstractMatrix) =
    GroupedGemmTask{typeof(A),typeof(B),typeof(C),promote_type(typeof(alpha), typeof(beta))}(
        transA, transB, alpha, A, B, beta, C)

"""Whether a backend has the heterogeneous grouped-GEMM primitive used by compressed FTLR."""
@inline supports_grouped_gemm(::Type) = false
@inline supports_grouped_gemm(backend) = supports_grouped_gemm(typeof(backend))
"""Whether this backend/device has native BF16 grouped GEMMEx support."""
@inline supports_bfloat16_grouped_gemm(::Type) = false
@inline supports_bfloat16_grouped_gemm(backend) = supports_bfloat16_grouped_gemm(typeof(backend))

"""
    precision_gemm_grouped!(tasks, mode)

Execute heterogeneous GEMMs. CUDA supplies the only implementation: callers
must group by backend capability rather than silently falling back to a loop.
"""
function precision_gemm_grouped!(tasks::AbstractVector{<:GroupedGemmTask},
                                 mode::AbstractGEMMComputeMode)
    isempty(tasks) && return tasks
    backend = get_backend(first(tasks).C)
    supports_grouped_gemm(backend) || throw(ArgumentError(
        "heterogeneous grouped GEMM is supported only on CUDA; got $(typeof(backend))"))
    needs_fallback = false
    for task in tasks
        get_backend(task.A) == backend && get_backend(task.B) == backend &&
            get_backend(task.C) == backend ||
            throw(ArgumentError("all grouped GEMM operands must use the same backend"))
        validate_gemm_signature(backend, eltype(task.A), eltype(task.B), eltype(task.C), mode)
        needs_fallback |= !_grouped_gemm_task_alignment_safe(task, mode)
    end
    needs_fallback || return _precision_gemm_grouped!(tasks, mode)

    # Some cuBLAS grouped kernels fault rather than returning an error for an
    # unaligned member. Keep aligned members grouped and route only unsafe
    # members through ordinary GEMMEx.
    TaskT = eltype(tasks)
    grouped = TaskT[]; fallback = TaskT[]
    sizehint!(grouped, length(tasks)); sizehint!(fallback, length(tasks))
    for task in tasks
        push!(_grouped_gemm_task_alignment_safe(task, mode) ? grouped : fallback, task)
    end
    isempty(grouped) || _precision_gemm_grouped!(grouped, mode)
    for task in fallback
        precision_gemm!(task.transA, task.transB, task.alpha, task.A, task.B,
                        task.beta, task.C, mode)
    end
    return tasks
end

_grouped_gemm_task_alignment_safe(task, mode) = true

function _precision_gemm_grouped!(tasks, mode)
    throw(ArgumentError("grouped GEMM is not implemented for $(typeof(get_backend(first(tasks).C)))"))
end

"""Opaque backend-owned descriptor for a prepared grouped-GEMM submission."""
abstract type AbstractPreparedGroupedGemm end

"""Prepared aligned groups plus ordinary-GEMM fallback members."""
struct PreparedGroupedGemmBundle{P,V,M,B} <: AbstractPreparedGroupedGemm
    grouped::P
    fallback::V
    mode::M
    backend::B
end

"""Typed, allocation-free-to-compare key used while constructing GEMM groups."""
struct GroupedGemmKey{S}
    transA::Char
    transB::Char
    m::Int32
    n::Int32
    k::Int32
    lda::Int32
    ldb::Int32
    ldc::Int32
    alpha::S
    beta::S
end

"""
    prepare_precision_gemm_grouped(tasks, mode)

Build and upload the immutable pointer/group metadata for `tasks`. The returned
descriptor retains no numerical workspace; callers must keep every referenced
matrix alive until all submissions using the descriptor have completed.
"""
function prepare_precision_gemm_grouped(tasks::AbstractVector{<:GroupedGemmTask},
                                        mode::AbstractGEMMComputeMode)
    isempty(tasks) && return nothing
    backend = get_backend(first(tasks).C)
    supports_grouped_gemm(backend) || throw(ArgumentError(
        "heterogeneous grouped GEMM is supported only on CUDA; got $(typeof(backend))"))
    needs_fallback = false
    @inbounds for task in tasks
        get_backend(task.A) == backend && get_backend(task.B) == backend &&
            get_backend(task.C) == backend ||
            throw(ArgumentError("all grouped GEMM operands must use the same backend"))
        validate_gemm_signature(backend, eltype(task.A), eltype(task.B), eltype(task.C), mode)
        needs_fallback |= !_grouped_gemm_task_alignment_safe(task, mode)
    end
    needs_fallback || return _prepare_precision_gemm_grouped(tasks, mode)

    grouped_tasks = GroupedGemmTask[]
    fallback = GroupedGemmTask[]
    sizehint!(grouped_tasks, length(tasks)); sizehint!(fallback, length(tasks))
    @inbounds for task in tasks
        push!(_grouped_gemm_task_alignment_safe(task, mode) ? grouped_tasks : fallback, task)
    end
    grouped = isempty(grouped_tasks) ? nothing :
        _prepare_precision_gemm_grouped(grouped_tasks, mode)
    return PreparedGroupedGemmBundle(grouped, fallback, mode, backend)
end

function _prepare_precision_gemm_grouped(tasks, mode)
    throw(ArgumentError(
        "prepared grouped GEMM is not implemented for $(typeof(get_backend(first(tasks).C)))"))
end

"""Submit a prepared group using the scalars captured during preparation."""
function precision_gemm_grouped_prepared!(prepared::AbstractPreparedGroupedGemm)
    return _precision_gemm_grouped_prepared!(prepared)
end

"""Submit a prepared group, replacing every group's scalar pair in place."""
function precision_gemm_grouped_prepared!(prepared::AbstractPreparedGroupedGemm, alpha, beta)
    return _precision_gemm_grouped_prepared!(prepared, alpha, beta)
end

function _precision_gemm_grouped_prepared!(prepared, args...)
    throw(ArgumentError("prepared grouped GEMM is not implemented for $(typeof(prepared))"))
end

function _precision_gemm_grouped_prepared!(bundle::PreparedGroupedGemmBundle,
                                                   overrides...)
    if bundle.grouped !== nothing
        _with_grouped_host_pointer_mode(bundle.backend) do
            _precision_gemm_grouped_prepared!(bundle.grouped, overrides...)
        end
    end
    _with_grouped_device_pointer_mode(bundle.backend) do
        for task in bundle.fallback
            alpha = isempty(overrides) ? task.alpha : overrides[1]
            beta = isempty(overrides) ? task.beta : overrides[2]
            precision_gemm!(task.transA, task.transB, alpha, task.A, task.B,
                            beta, task.C, bundle.mode)
        end
    end
    return bundle
end

function _destroy_prepared_grouped_gemm!(bundle::PreparedGroupedGemmBundle)
    bundle.grouped === nothing || destroy_prepared_grouped_gemm!(bundle.grouped)
    return bundle
end

"""Release device metadata owned by a prepared descriptor."""
function destroy_prepared_grouped_gemm!(prepared::AbstractPreparedGroupedGemm)
    return _destroy_prepared_grouped_gemm!(prepared)
end
_destroy_prepared_grouped_gemm!(prepared) = prepared

"""Run `f` while grouped-GEMM scalar arrays are interpreted as host memory."""
_with_grouped_host_pointer_mode(f, backend) = f()
_with_grouped_device_pointer_mode(f, backend) = f()


"""Opaque fixed-capacity grouped-GEMM metadata slot used by streaming drivers."""
abstract type AbstractReusableGroupedGemmSlot end

function create_reusable_grouped_gemm_slot(backend, max_tasks::Int, max_groups::Int,
                                           ::Type{Tin}, ::Type{Tout}, mode) where {Tin,Tout}
    max_tasks >= 0 || throw(ArgumentError("max_tasks must be nonnegative"))
    max_groups >= 0 || throw(ArgumentError("max_groups must be nonnegative"))
    return _create_reusable_grouped_gemm_slot(
        backend, max_tasks, max_groups, Tin, Tout, mode)
end

function _create_reusable_grouped_gemm_slot(backend, args...)
    throw(ArgumentError("reusable grouped GEMM slots are not implemented for $(typeof(backend))"))
end

function refresh_reusable_grouped_gemm!(slot::AbstractReusableGroupedGemmSlot,
                                        tasks, mode)
    return _refresh_reusable_grouped_gemm!(slot, tasks, mode)
end

function _refresh_reusable_grouped_gemm!(slot, tasks, mode)
    throw(ArgumentError("cannot refresh reusable grouped GEMM slot $(typeof(slot))"))
end

function submit_reusable_grouped_gemm!(slot::AbstractReusableGroupedGemmSlot)
    return _submit_reusable_grouped_gemm!(slot)
end

function submit_reusable_grouped_gemm!(slot::AbstractReusableGroupedGemmSlot, alpha, beta)
    return _submit_reusable_grouped_gemm!(slot, alpha, beta)
end

function _submit_reusable_grouped_gemm!(slot, args...)
    throw(ArgumentError("cannot submit reusable grouped GEMM slot $(typeof(slot))"))
end

function destroy_reusable_grouped_gemm!(slot::AbstractReusableGroupedGemmSlot)
    return _destroy_reusable_grouped_gemm!(slot)
end
_destroy_reusable_grouped_gemm!(slot) = slot
