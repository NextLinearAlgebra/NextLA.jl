"""
    DenseGemmWorkspace

Reusable numerical arena for dense-output TLR GEMM. Backend-library internal
storage is outside this workspace.
"""
struct DenseGemmWorkspace{T,A<:AbstractVector{T}}
    storage::A
end

Base.eltype(::DenseGemmWorkspace{T}) where {T} = T
Base.length(ws::DenseGemmWorkspace) = length(ws.storage)
Base.sizeof(ws::DenseGemmWorkspace) = length(ws) * sizeof(eltype(ws))

function DenseGemmWorkspace(A::AbstractTLRMatrix{<:Any,T}, bytes::Integer) where {T}
    bytes >= 0 || throw(ArgumentError("workspace bytes must be nonnegative"))
    backend = get_backend(A)
    storage = allocate(backend, T, fld(Int(bytes), sizeof(T)))
    return DenseGemmWorkspace(storage)
end

function DenseGemmWorkspace(A::AbstractTLRMatrix{<:Any,T},
                            B::AbstractTLRMatrix{<:Any,T};
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

mutable struct DenseGemmArena{A}
    storage::A
    cursor::Int
end

@inline _arena_reset!(arena::DenseGemmArena) = (arena.cursor = firstindex(arena.storage); arena)
@inline _arena_reset!(::Nothing) = nothing

function _arena_array!(arena::DenseGemmArena, ::Type{T}, dims::Int...) where {T}
    T === eltype(arena.storage) ||
        throw(ArgumentError("workspace element type $(eltype(arena.storage)) does not match $T"))
    count = prod(dims)
    first = arena.cursor
    last = first + count - 1
    last <= lastindex(arena.storage) ||
        throw(ArgumentError("workspace slice exhausted while requesting $(count * sizeof(T)) bytes"))
    arena.cursor = last + 1
    return reshape(view(arena.storage, first:last), dims...)
end

@inline _workspace_array!(::Nothing, backend, ::Type{T}, dims::Int...) where {T} =
    allocate(backend, T, dims...)
@inline _workspace_array!(arena::DenseGemmArena, _, ::Type{T}, dims::Int...) where {T} =
    _arena_array!(arena, T, dims...)

function _prepare_single_gemm_workspace(A::AbstractTLRMatrix{<:Any,T},
                                        workspace) where {T}
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
    return ws, DenseGemmArena(view(ws.storage, :), 1), sizeof(ws)
end
