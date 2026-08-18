# Result-independent bump arena shared by dense-output and compressed-output ARA GEMM.

mutable struct GemmArena{A}
    storage::A
    cursor::Int
end

@inline _arena_reset!(arena::GemmArena) =
    (arena.cursor = firstindex(arena.storage); arena)
@inline _arena_reset!(::Nothing) = nothing

function _workspace_array!(arena::GemmArena, _, ::Type{T}, dims::Int...) where {T}
    T === eltype(arena.storage) || throw(ArgumentError(
        "workspace element type $(eltype(arena.storage)) does not match $T"))
    count = prod(dims)
    first = arena.cursor
    last = first + count - 1
    last <= lastindex(arena.storage) || throw(ArgumentError(
        "workspace slice exhausted while requesting $(count * sizeof(T)) bytes"))
    arena.cursor = last + 1
    return reshape(view(arena.storage, first:last), dims...)
end

@inline _workspace_array!(::Nothing, backend, ::Type{T}, dims::Int...) where {T} =
    allocate(backend, T, dims...)
