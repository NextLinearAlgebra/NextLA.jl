"""
    ARARunArena(backend, ::Type{T}, ::Type{Thi},
                persistent_t_bytes, phase_t_bytes, phase_thi_bytes)

Reusable scratch for a `RunCoupling` and its `ARAWorkspace`, reset by
[`arena_reset!`](@ref) between runs. `persistent_t` holds `Q`, `S`, and packed
factor stacks; `phase_t` is rewound between sampling and finalization;
`phase_thi` holds promoted Cholesky-QR scratch. [`run_t_arena`](@ref) and
[`run_thi_arena`](@ref) also accept `nothing` for allocation-backed callers.
"""
struct ARARunArena{PA<:GemmArena,TA<:GemmArena,HA<:GemmArena}
    persistent_t::PA
    phase_t::TA
    phase_thi::HA
end

function ARARunArena(backend, ::Type{T}, ::Type{Thi},
                     persistent_t_bytes::Int, phase_t_bytes::Int,
                     phase_thi_bytes::Int) where {T,Thi}
    persistent_t_bytes >= 0 && phase_t_bytes >= 0 && phase_thi_bytes >= 0 ||
        throw(ArgumentError("arena byte counts must be nonnegative"))

    persistent = allocate(backend, T, cld(persistent_t_bytes, sizeof(T)))
    phase = allocate(backend, T, cld(phase_t_bytes, sizeof(T)))
    phase_hi = allocate(backend, Thi, cld(phase_thi_bytes, sizeof(Thi)))

    return ARARunArena(
        GemmArena(persistent, 1),
        GemmArena(phase, 1),
        GemmArena(phase_hi, 1),
    )
end

@inline arena_reset!(arena::ARARunArena) =
    (arena_reset!(arena.persistent_t); arena_reset!(arena.phase_t);
     arena_reset!(arena.phase_thi); arena)
@inline arena_reset_phase!(arena::ARARunArena) =
    (arena_reset!(arena.phase_t); arena_reset!(arena.phase_thi); arena)
@inline arena_reset_phase!(::Nothing) = nothing

@inline run_t_arena(::Nothing) = nothing
@inline run_t_arena(arena::ARARunArena) = arena.phase_t
@inline run_persistent_t_arena(::Nothing) = nothing
@inline run_persistent_t_arena(arena::ARARunArena) = arena.persistent_t
@inline run_thi_arena(::Nothing) = nothing
@inline run_thi_arena(arena::ARARunArena) = arena.phase_thi

Base.sizeof(arena::ARARunArena) =
    sizeof(arena.persistent_t.storage) +
    sizeof(arena.phase_t.storage) +
    sizeof(arena.phase_thi.storage)

"""
    CompressedGemmWorkspace

Reusable single-stream storage for canonical TLR-output GEMM: one
phase-reusing `ARARunArena` plus traversal, diagnostic, and scatter buffers.
"""
struct CompressedGemmWorkspace{A,U,V,RS,ES,I,RD,ED,AS,M,P,O,IH,K}
    arena::A
    U::U
    V::V
    ranks_slot::RS
    errors_slot::ES
    indices::I
    ranks_global::RD
    errors_global::ED
    ara_state::AS
    member_ids::M
    progress::P
    output_slots::O
    output_slots_inner::O
    indices_host::IH
    operation::K
    capacity::Int
end

function Base.sizeof(ws::CompressedGemmWorkspace)
    return sizeof(ws.arena) + sizeof(ws.U) + sizeof(ws.V) +
           sizeof(ws.ranks_slot) + sizeof(ws.errors_slot) +
           sizeof(ws.indices) + sizeof(ws.ranks_global) +
           sizeof(ws.errors_global) + sizeof(ws.ara_state.dR) +
           sum(sizeof, (
               ws.ara_state.status, ws.ara_state.kcut,
               ws.ara_state.samples, ws.ara_state.ranks,
               ws.ara_state.svec, ws.ara_state.jcount,
               ws.ara_state.rmax,
           ))
end

function CompressedGemmWorkspace(C::CompressedFTLRMatrix{BackendT,T}, spec;
                          bytes=nothing) where {BackendT,T}
    # requested byte budget and slot capacity
    requested = bytes === nothing ?
        tlr_gemm_workspace_bytes(spec, spec.nmember) : bytes
    requested >= tlr_gemm_workspace_bytes(spec, 1) || throw(ArgumentError(
        "workspace has $requested bytes; at least " *
        "$(tlr_gemm_workspace_bytes(spec, 1)) bytes are required"))
    capacity = _tlr_workspace_capacity(spec, requested)
    backend = get_backend(C)

    # run arena for the selected capacity
    ab = ara_run_workspace_bytes(
        spec.family, spec.rA, spec.rB, spec.qk, capacity,
        spec.block, spec.maxrank, spec.bm, spec.bn,
        T, spec.Thi)
    arena = ARARunArena(
        backend, T, spec.Thi, ab.persistent_t_bytes,
        ab.phase_t_bytes, ab.phase_thi_bytes)

    # convergence-state scratch, one entry per slot
    n = capacity
    opkey = spec
    ara_state = (
        dR=allocate(backend, Float64, opkey.block, n),
        status=allocate(backend, Int32, n),
        status_host=Vector{Int32}(undef, n),
        kcut=allocate(backend, Int32, n),
        kcut_host=Vector{Int32}(undef, n),
        samples=allocate(backend, Int32, n),
        ranks=allocate(backend, Int32, n),
        svec=allocate(backend, Int32, n),
        jcount=allocate(backend, Int32, n),
        rmax=allocate(backend, Float64, n),
        samples_host=Vector{Int32}(undef, n),
    )

    # traversal output (U, V), diagnostic, and scatter buffers
    return CompressedGemmWorkspace(
        arena,
        allocate(backend, T, opkey.bm, opkey.maxrank, n),
        allocate(backend, T, opkey.bn, opkey.maxrank, n),
        allocate(backend, opkey.rankT, n),
        allocate(backend, Float64, n),
        allocate(backend, Int32, n),
        allocate(backend, opkey.rankT, opkey.qm * opkey.qn),
        allocate(backend, Float64, opkey.qm * opkey.qn),
        ara_state,
        Vector{Int}(undef, n),
        Base.zeros(Int, n),
        Vector{Int}(undef, n),
        Vector{Int}(undef, n),
        Vector{Int32}(undef, n),
        spec,
        capacity,
    )
end

function tlr_gemm_workspace_bytes(spec, capacity::Int)
    1 <= capacity <= spec.nmember ||
        throw(ArgumentError("slot capacity must be in 1:$(spec.nmember)"))

    # arena and traversal storage
    k = spec
    ab = ara_run_workspace_bytes(
        k.family, k.rA, k.rB, k.qk, capacity, k.block,
        k.maxrank, k.bm, k.bn, k.T, k.Thi)
    arena = ab.persistent_t_bytes + ab.phase_t_bytes + ab.phase_thi_bytes
    traversal_t = (k.bm + k.bn) * k.maxrank * capacity * sizeof(k.T)

    # diagnostics and convergence state
    diagnostics = capacity * sizeof(k.rankT) +
                  capacity * sizeof(Float64) +
                  capacity * sizeof(Int32) +
                  k.qm * k.qn * (sizeof(k.rankT) + sizeof(Float64))
    ara_state = k.block * capacity * sizeof(Float64) +
                6 * capacity * sizeof(Int32) +
                capacity * sizeof(Float64)

    return arena + traversal_t + diagnostics + ara_state
end

function _tlr_workspace_capacity(spec, bytes::Int)
    lo, hi = 1, spec.nmember
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        if tlr_gemm_workspace_bytes(spec, mid) <= bytes
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end
