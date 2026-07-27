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

function TLRGemmWorkspace(C::TLRMatrix{BackendT,T},
                          A::TLRMatrix{BackendT,T},
                          B::TLRMatrix{BackendT,T};
                          transA::Char='N', transB::Char='N',
                          block::Int=32) where {BackendT,T}
    spec = _tlr_gemm_workspace_spec(C, A, B; transA, transB, block)
    backend = get_backend(C)
    ab = spec.arena_bytes
    arena = ARARunArena(
        backend, T, spec.Thi, ab.persistent_t_bytes,
        ab.phase_t_bytes, ab.phase_thi_bytes)
    n = spec.nmember
    key = spec.key
    return TLRGemmWorkspace(
        arena,
        allocate(backend, T, key.bm, key.maxrank, n),
        allocate(backend, T, key.bn, key.maxrank, n),
        allocate(backend, key.rankT, n),
        allocate(backend, Float64, n),
        allocate(backend, key.rankT, n),
        allocate(backend, Float64, n),
        allocate(backend, Int32, n),
        allocate(backend, key.rankT, key.qm * key.qn),
        allocate(backend, Float64, key.qm * key.qn),
        key,
    )
end

"""
    tlr_gemm_workspace_bytes(C, A, B; transA='N', transB='N', block=32)

Exact numerical storage owned by a reusable `TLRGemmWorkspace` for the
canonical TLR-output operation and sampling choice.
"""
function tlr_gemm_workspace_bytes(C::TLRMatrix, A::TLRMatrix, B::TLRMatrix;
                                  transA::Char='N', transB::Char='N',
                                  block::Int=32)
    spec = _tlr_gemm_workspace_spec(C, A, B; transA, transB, block)
    k = spec.key
    ab = spec.arena_bytes
    arena = ab.persistent_t_bytes + ab.phase_t_bytes + ab.phase_thi_bytes
    traversal_t = (k.bm + k.bn) * k.maxrank * k.nmember * sizeof(k.T)
    diagnostics = 2 * k.nmember * sizeof(k.rankT) +
                  2 * k.nmember * sizeof(Float64) +
                  k.nmember * sizeof(Int32) +
                  k.qm * k.qn * (sizeof(k.rankT) + sizeof(Float64))
    return arena + traversal_t + diagnostics
end
