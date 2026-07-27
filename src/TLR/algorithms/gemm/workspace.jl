"""
    InteriorFirstWorkspace

Static two-stream workspace policy. Reserve the auxiliary stream's minimum
workspace, then give all remaining capacity to the interior until its runs are
full width. Any remaining capacity enlarges the serialized boundary runs.
"""
struct InteriorFirstWorkspace end

"""
    DenseGemmWorkspace

Reusable numerical arena and execution streams for dense-output TLR GEMM.
Backend-library internal storage is outside this workspace.
"""
struct DenseGemmWorkspace{T,A<:AbstractVector{T},S}
    storage::A
    streams::S
end

Base.eltype(::DenseGemmWorkspace{T}) where {T} = T
Base.length(ws::DenseGemmWorkspace) = length(ws.storage)
Base.sizeof(ws::DenseGemmWorkspace) = length(ws) * sizeof(eltype(ws))

function DenseGemmWorkspace(A::AbstractTLRMatrix{<:Any,T}, bytes::Integer) where {T}
    bytes >= 0 || throw(ArgumentError("workspace bytes must be nonnegative"))
    backend = get_backend(A)
    storage = allocate(backend, T, fld(Int(bytes), sizeof(T)))
    return DenseGemmWorkspace(storage, create_streams(backend, 2))
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

@inline function _split_workspace(ws::DenseGemmWorkspace,
                                  interior_bytes::Int, auxiliary_bytes::Int)
    T = eltype(ws)
    ni = cld(interior_bytes, sizeof(T))
    na = cld(auxiliary_bytes, sizeof(T))
    ni + na <= length(ws) || throw(ArgumentError(
        "workspace has $(sizeof(ws)) bytes but the selected split requires " *
        "$((ni + na) * sizeof(T)) bytes"))
    interior = DenseGemmArena(view(ws.storage, 1:ni), 1)
    auxiliary = DenseGemmArena(view(ws.storage, (ni + 1):(ni + na)), 1)
    return interior, auxiliary
end

function _prepare_dense_gemm_workspace(A::AbstractTLRMatrix{<:Any,T},
                                       B::AbstractTLRMatrix{<:Any,T},
                                       workspace,
                                       policy::InteriorFirstWorkspace;
                                       transA::Char='N',
                                       transB::Char='N') where {T}
    ws, requested = if workspace isa Integer
        bytes = Int(workspace)
        bytes >= 0 || throw(ArgumentError("workspace bytes must be nonnegative"))
        temporary = DenseGemmWorkspace(A, bytes)
        (temporary, sizeof(temporary))
    elseif workspace isa DenseGemmWorkspace
        eltype(workspace) === T || throw(ArgumentError(
            "workspace element type $(eltype(workspace)) does not match operand type $T"))
        typeof(get_backend(workspace.storage)) === typeof(get_backend(A)) ||
            throw(ArgumentError("workspace and operands must use the same backend"))
        (workspace, sizeof(workspace))
    else
        throw(ArgumentError(
            "workspace must be an integer byte count or DenseGemmWorkspace"))
    end
    split = _gemm_workspace_split(
        A, B, requested, policy; transA, transB)
    interior, auxiliary = _split_workspace(
        ws, split.interior, split.auxiliary)
    return ws, interior, auxiliary, split
end

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

"""
    ARARunArena(backend, ::Type{T}, ::Type{Thi},
                persistent_t_bytes, phase_t_bytes, phase_thi_bytes)

Reusable numerical scratch for one canonical TLR-output GEMM run (a
`ColumnRunCoupling`/`RowRightRunCoupling`/`RowLeftRunCoupling` plus its
`ARAWorkspace`), threaded through as the `arena` keyword everywhere a run's
constructors would otherwise call `allocate` directly. Reset with
[`_arena_reset!`](@ref) once per run in the driver's traversal loop, so the
whole run's scratch is reused across iterations instead of being allocated
and freed on every one.

`persistent_t` holds `Q`, `S`, and packed factor stacks for the whole run.
`phase_t` is rewound after basis construction and reused for co-range and
truncation, so its size is the maximum of the sampling and finalization
phases rather than their sum. `phase_thi` holds the promoted Cholesky-QR
scratch, which is live only during sampling.

`t_arena`/`thi_arena` are accessed only through [`_run_t_arena`](@ref)/
[`_run_thi_arena`](@ref), which also accept `arena=nothing` and return
`nothing`, so every call site can be written once and work whether or not an
arena was supplied.
"""
struct ARARunArena{PA<:DenseGemmArena,TA<:DenseGemmArena,HA<:DenseGemmArena}
    persistent_t::PA
    phase_t::TA
    phase_thi::HA
end

function ARARunArena(backend, ::Type{T}, ::Type{Thi},
                     persistent_t_bytes::Integer, phase_t_bytes::Integer,
                     phase_thi_bytes::Integer) where {T,Thi}
    persistent_t_bytes >= 0 && phase_t_bytes >= 0 && phase_thi_bytes >= 0 ||
        throw(ArgumentError("arena byte counts must be nonnegative"))
    persistent = allocate(backend, T, cld(Int(persistent_t_bytes), sizeof(T)))
    phase = allocate(backend, T, cld(Int(phase_t_bytes), sizeof(T)))
    phase_hi = allocate(backend, Thi, cld(Int(phase_thi_bytes), sizeof(Thi)))
    return ARARunArena(
        DenseGemmArena(persistent, 1),
        DenseGemmArena(phase, 1),
        DenseGemmArena(phase_hi, 1),
    )
end

@inline _arena_reset!(arena::ARARunArena) =
    (_arena_reset!(arena.persistent_t); _arena_reset!(arena.phase_t);
     _arena_reset!(arena.phase_thi); arena)
@inline _arena_reset_phase!(arena::ARARunArena) =
    (_arena_reset!(arena.phase_t); _arena_reset!(arena.phase_thi); arena)
@inline _arena_reset_phase!(::Nothing) = nothing

@inline _run_t_arena(::Nothing) = nothing
@inline _run_t_arena(arena::ARARunArena) = arena.phase_t
@inline _run_persistent_t_arena(::Nothing) = nothing
@inline _run_persistent_t_arena(arena::ARARunArena) = arena.persistent_t
@inline _run_thi_arena(::Nothing) = nothing
@inline _run_thi_arena(arena::ARARunArena) = arena.phase_thi

@inline _numerical_bytes(A::AbstractArray) = length(A) * sizeof(eltype(A))

Base.sizeof(arena::ARARunArena) =
    _numerical_bytes(arena.persistent_t.storage) +
    _numerical_bytes(arena.phase_t.storage) +
    _numerical_bytes(arena.phase_thi.storage)

"""
    TLRGemmWorkspace

Reusable numerical storage for canonical TLR-output GEMM. Unlike the
dense-output workspace, execution is currently single-stream; the object owns
one phase-reusing `ARARunArena` plus the traversal output, diagnostic, and
scatter buffers that would otherwise be allocated once per `gemm!` call.
"""
struct TLRGemmWorkspace{A,U,V,R,E,RS,ES,I,RD,ED,K}
    arena::A
    U::U
    V::V
    ranks::R
    errors::E
    ranks_slot::RS
    errors_slot::ES
    indices::I
    ranks_global::RD
    errors_global::ED
    key::K
end

function Base.sizeof(ws::TLRGemmWorkspace)
    return sizeof(ws.arena) + _numerical_bytes(ws.U) + _numerical_bytes(ws.V) +
           _numerical_bytes(ws.ranks) + _numerical_bytes(ws.errors) +
           _numerical_bytes(ws.ranks_slot) + _numerical_bytes(ws.errors_slot) +
           _numerical_bytes(ws.indices) + _numerical_bytes(ws.ranks_global) +
           _numerical_bytes(ws.errors_global)
end
