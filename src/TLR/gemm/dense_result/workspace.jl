"""Reusable numerical storage for GEMMs whose destination is dense."""
struct DenseGemmWorkspace{T,A<:AbstractVector{T}}
    storage::A
end

Base.eltype(::DenseGemmWorkspace{T}) where {T} = T
Base.sizeof(ws::DenseGemmWorkspace) = sizeof(ws.storage)

function DenseGemmWorkspace(A::AbstractTLRMatrix{T}, bytes::Integer) where {T}
    bytes >= 0 || throw(ArgumentError("workspace bytes must be nonnegative"))
    storage = allocate(get_backend(A), T, fld(Int(bytes), sizeof(T)))
    return DenseGemmWorkspace(storage)
end

function DenseGemmWorkspace(A::AbstractTLRMatrix{T},
                            B::AbstractTLRMatrix{T};
                            bytes::Integer,
                            transA::Char='N', transB::Char='N') where {T}
    required = gemm_minimum_workspace_bytes(A, B; transA, transB)
    bytes >= required || throw(ArgumentError(
        "workspace has $bytes bytes; at least $required bytes are required"))
    workspace = DenseGemmWorkspace(A, bytes)
    sizeof(workspace) >= required || throw(ArgumentError(
        "workspace byte count must contain at least $required bytes of complete $T elements"))
    return workspace
end

function _prepare_dense_result_workspace(
    A::AbstractTLRMatrix{T}, workspace) where {T}
    ws = if workspace isa Integer
        DenseGemmWorkspace(A, Int(workspace))
    elseif workspace isa DenseGemmWorkspace
        eltype(workspace) === T || throw(ArgumentError(
            "workspace element type $(eltype(workspace)) does not match operand type $T"))
        typeof(get_backend(workspace.storage)) === typeof(get_backend(A)) ||
            throw(ArgumentError("workspace and operands must use the same backend"))
        workspace
    else
        throw(ArgumentError(
            "workspace must be an integer byte count or DenseGemmWorkspace"))
    end
    return ws, GemmArena(view(ws.storage, :), 1), sizeof(ws)
end
