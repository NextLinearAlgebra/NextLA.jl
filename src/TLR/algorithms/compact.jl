# ─── Kernel: padded [b, maxrank, noff] → compact flat buffer ──────────────────
#
# One workgroup per tile; BLOCKDIM threads stride over the tile's elements.
# Each element is stored in column-major order inside the flat buffer:
#   element (row, col) → flat index  (col-1)*nrows + row  (1-based within tile)

@kernel function _compact_to_flat_kernel!(
    out::AbstractVector{T},
    src::AbstractArray{T,3},
    offsets::AbstractVector{Int},    # 1-based, length = noffdiag
    nrows_v::AbstractVector{Int},
    ranks::AbstractVector{<:Integer}) where {T}

    ob = @index(Group, Linear)           # tile index
    tid = @index(Local, Linear)           # thread within tile
    gs = @groupsize()[1]

    r_ob = Int(@inbounds ranks[ob])
    n_rows = Int(@inbounds nrows_v[ob])
    off = Int(@inbounds offsets[ob]) - 1    # 0-based for arithmetic
    n_elem = n_rows * r_ob

    @inbounds for k in tid:gs:n_elem
        col = (k - 1) ÷ n_rows + 1
        row = (k - 1) % n_rows + 1
        out[off+k] = src[row, col, ob]   # column-major layout in flat buffer
    end
end

# ─── compact! ────────────────────────────────────────────────────────────────

"""
    compact!(A_tlr::TLRMatrix)

Convert `A_tlr` from padded `UniformTileStorage` (shape `b × maxrank` per tile) to
`CompactTileStorage` (flat buffers sized to the actual per-tile rank)
**in place**.

`compress!` must have been called first so that `ranks(A_tlr)` is populated.

A single KernelAbstractions kernel performs the gather on both CPU and GPU
backends. The factor payload stays on the source backend; only the metadata
arrays are mirrored to the execution backend for the gather.

The metadata arrays (`U_offsets`, `V_offsets`, `ranks`, …) are always stored
in host memory so that per-tile access from Julia code remains allocation-free.
"""
function compact!(A_tlr::TLRMatrix{<:Any,T}) where {T}
    s = A_tlr.storage
    if s isa CompactTileStorage
        return A_tlr      # already compact — idempotent
    end
    s isa UniformTileStorage ||
        throw(ArgumentError("compact! requires UniformTileStorage; " *
                            "use TLRMatrix(...) to allocate a fresh container"))
    A_tlr.storage = _to_compact(A_tlr.backend, s, A_tlr.layout)
    return A_tlr
end

function _to_compact(backend, s::UniformTileStorage{T}, layout::TileMap) where {T}
    noff = noffdiag_tiles(layout)
    rk_cpu = s.ranks isa Vector ? s.ranks : Array(s.ranks)   # GPU → CPU if needed
    noff == 0 && return CompactTileStorage(
        similar(s.U, T, 0),
        similar(s.V, T, 0),
        Int[],
        Int[],
        Int[],
        Int[],
        Array(s.D),
        eltype(rk_cpu)[],
        s.compress_diag,
    )

    # Per-tile metadata (always CPU)
    U_nrows = Vector{Int}(undef, noff)
    V_nrows = Vector{Int}(undef, noff)
    for ob in 1:noff
        lin = offdiag_linear_index(layout, ob)
        tile_i, tile_j = inverse_tile_index(layout, lin)
        tm, tn = tile_sizes(layout, tile_i, tile_j)
        U_nrows[ob] = tm
        V_nrows[ob] = tn
    end

    # Flat-buffer offsets (1-based, always CPU)
    U_sizes = [U_nrows[ob] * Int(rk_cpu[ob]) for ob in 1:noff]
    V_sizes = [V_nrows[ob] * Int(rk_cpu[ob]) for ob in 1:noff]
    U_offsets = cumsum([1; U_sizes])[1:noff]    # U_offsets[ob] = start in U_data
    V_offsets = cumsum([1; V_sizes])[1:noff]
    U_total = sum(U_sizes)
    V_total = sum(V_sizes)

    # Allocate flat buffers on the same device as the padded factors
    U_data = similar(s.U, T, U_total)
    V_data = similar(s.V, T, V_total)

    # Mirror metadata to the execution backend for the gather kernel.
    U_off_d = similar(s.U, Int, noff)
    copyto!(U_off_d, U_offsets)
    V_off_d = similar(s.V, Int, noff)
    copyto!(V_off_d, V_offsets)
    U_nr_d = similar(s.U, Int, noff)
    copyto!(U_nr_d, U_nrows)
    V_nr_d = similar(s.V, Int, noff)
    copyto!(V_nr_d, V_nrows)
    rk_d = backend isa KernelAbstractions.CPU ? rk_cpu : s.ranks

    BLOCKDIM = 128        # threads per tile; stride-loops handle larger tiles
    kernel! = _compact_to_flat_kernel!(backend, BLOCKDIM)
    kernel!(U_data, s.U, U_off_d, U_nr_d, rk_d;
        ndrange=(BLOCKDIM * noff,), workgroupsize=BLOCKDIM)
    kernel!(V_data, s.V, V_off_d, V_nr_d, rk_d;
        ndrange=(BLOCKDIM * noff,), workgroupsize=BLOCKDIM)
    KernelAbstractions.synchronize(backend)

    return CompactTileStorage(
        U_data, V_data,
        U_offsets, V_offsets,
        U_nrows, V_nrows,
        Array(s.D),
        rk_cpu,
        s.compress_diag,
    )
end
