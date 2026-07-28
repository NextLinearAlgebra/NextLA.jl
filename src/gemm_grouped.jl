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
    for task in tasks
        get_backend(task.A) == backend && get_backend(task.B) == backend &&
            get_backend(task.C) == backend ||
            throw(ArgumentError("all grouped GEMM operands must use the same backend"))
        validate_gemm_signature(backend, eltype(task.A), eltype(task.B), eltype(task.C), mode)
    end
    return _precision_gemm_grouped!(tasks, mode)
end

function _precision_gemm_grouped!(tasks, mode)
    throw(ArgumentError("grouped GEMM is not implemented for $(typeof(get_backend(first(tasks).C)))"))
end
