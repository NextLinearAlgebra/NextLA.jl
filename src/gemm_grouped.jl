"""
    GroupedGemmTask(transA, transB, alpha, A, B, beta, C)

One heterogeneous GEMM member. CUDA submits members through grouped GEMM;
CPU executes identical semantics with a loop. Unlike `gemm_batched!`, members
are allowed to have different `m`, `n`, and `k`.
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

"""Whether a backend has the heterogeneous grouped-GEMM interface used by compressed FTLR."""
@inline supports_grouped_gemm(::Type) = false
@inline supports_grouped_gemm(::Type{<:KernelAbstractions.CPU}) = true
@inline supports_grouped_gemm(backend) = supports_grouped_gemm(typeof(backend))
"""Whether this backend/device has native BF16 grouped GEMMEx support."""
@inline supports_bfloat16_grouped_gemm(::Type) = false
@inline supports_bfloat16_grouped_gemm(backend) = supports_bfloat16_grouped_gemm(typeof(backend))

"""
    precision_gemm_grouped!(tasks, mode)

Execute heterogeneous GEMMs. CUDA uses its native grouped primitive; CPU
implements the same semantics with an ordinary GEMM loop.
"""
function precision_gemm_grouped!(tasks::AbstractVector{<:GroupedGemmTask},
                                 mode::AbstractGEMMComputeMode)
    isempty(tasks) && return tasks
    backend = get_backend(first(tasks).C)
    supports_grouped_gemm(backend) || throw(ArgumentError(
        "heterogeneous grouped GEMM is unsupported on $(typeof(backend))"))
    needs_fallback = false
    for task in tasks
        get_backend(task.A) == backend && get_backend(task.B) == backend &&
            get_backend(task.C) == backend ||
            throw(ArgumentError("all grouped GEMM operands must use the same backend"))
        validate_gemm_signature(backend, eltype(task.A), eltype(task.B), eltype(task.C), mode)
        needs_fallback |= !_grouped_gemm_task_alignment_safe(backend, task, mode)
    end
    needs_fallback || return _precision_gemm_grouped!(backend, tasks, mode)

    # Some cuBLAS grouped kernels fault rather than returning an error for an
    # unaligned member. Keep aligned members grouped and route only unsafe
    # members through ordinary GEMMEx.
    TaskT = eltype(tasks)
    grouped = TaskT[]; fallback = TaskT[]
    sizehint!(grouped, length(tasks)); sizehint!(fallback, length(tasks))
    for task in tasks
        push!(_grouped_gemm_task_alignment_safe(backend, task, mode) ? grouped : fallback, task)
    end
    isempty(grouped) || _precision_gemm_grouped!(backend, grouped, mode)
    for task in fallback
        precision_gemm!(task.transA, task.transB, task.alpha, task.A, task.B,
                        task.beta, task.C, mode)
    end
    return tasks
end

"""
    _grouped_gemm_task_alignment_safe(backend, task, mode) -> Bool

Whether `task` is safe to submit through `backend`'s native grouped-GEMM
primitive at `mode`. Dispatches on `backend` (not `mode`) so a backend with
alignment-sensitive tensor-core kernels (see CUDA's override) cannot shadow
another backend's dispatch merely by specializing `mode` more tightly; the
default here covers every backend without that sensitivity.
"""
_grouped_gemm_task_alignment_safe(backend, task, mode) = true

function _precision_gemm_grouped!(backend, tasks, mode)
    throw(ArgumentError("grouped GEMM is not implemented for $(typeof(backend))"))
end

function _precision_gemm_grouped!(
    ::KernelAbstractions.CPU, tasks::AbstractVector{<:GroupedGemmTask}, mode::AbstractGEMMComputeMode)
    @inbounds for task in tasks
        precision_gemm!(task.transA, task.transB, task.alpha, task.A, task.B,
                        task.beta, task.C, mode)
    end
    return tasks
end

"""Opaque backend-owned descriptor for a prepared grouped-GEMM submission."""
abstract type AbstractPreparedGroupedGemm end

"""CPU prepared group: retain task views and replay them with ordinary GEMMs."""
struct CPUPreparedGroupedGemm{V,M} <: AbstractPreparedGroupedGemm
    tasks::V
    mode::M
end

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
        "heterogeneous grouped GEMM is unsupported on $(typeof(backend))"))
    needs_fallback = false
    @inbounds for task in tasks
        get_backend(task.A) == backend && get_backend(task.B) == backend &&
            get_backend(task.C) == backend ||
            throw(ArgumentError("all grouped GEMM operands must use the same backend"))
        validate_gemm_signature(backend, eltype(task.A), eltype(task.B), eltype(task.C), mode)
        needs_fallback |= !_grouped_gemm_task_alignment_safe(backend, task, mode)
    end
    needs_fallback || return _prepare_precision_gemm_grouped(backend, tasks, mode)

    grouped_tasks = GroupedGemmTask[]
    fallback = GroupedGemmTask[]
    sizehint!(grouped_tasks, length(tasks)); sizehint!(fallback, length(tasks))
    @inbounds for task in tasks
        push!(_grouped_gemm_task_alignment_safe(backend, task, mode) ? grouped_tasks : fallback, task)
    end
    grouped = isempty(grouped_tasks) ? nothing :
        _prepare_precision_gemm_grouped(backend, grouped_tasks, mode)
    return PreparedGroupedGemmBundle(grouped, fallback, mode, backend)
end

function _prepare_precision_gemm_grouped(backend, tasks, mode)
    throw(ArgumentError(
        "prepared grouped GEMM is not implemented for $(typeof(backend))"))
end

function _prepare_precision_gemm_grouped(
    ::KernelAbstractions.CPU, tasks::AbstractVector{<:GroupedGemmTask}, mode::AbstractGEMMComputeMode)
    return CPUPreparedGroupedGemm(tasks, mode)
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

function _precision_gemm_grouped_prepared!(prepared::CPUPreparedGroupedGemm,
                                            overrides...)
    @inbounds for task in prepared.tasks
        alpha = isempty(overrides) ? task.alpha : overrides[1]
        beta = isempty(overrides) ? task.beta : overrides[2]
        precision_gemm!(task.transA, task.transB, alpha, task.A, task.B,
                        beta, task.C, prepared.mode)
    end
    return prepared
end

function _precision_gemm_grouped_prepared!(bundle::PreparedGroupedGemmBundle,
                                                   overrides...)
    if bundle.grouped !== nothing
        with_grouped_host_pointer_mode(bundle.backend) do
            _precision_gemm_grouped_prepared!(bundle.grouped, overrides...)
        end
    end
    with_grouped_device_pointer_mode(bundle.backend) do
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
with_grouped_host_pointer_mode(f, backend) = f()
with_grouped_device_pointer_mode(f, backend) = f()
