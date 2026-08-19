"""
Reusable dense-to-FTLR compression storage.

There is one category per distinct logical tile shape. `diagonal=:compressed`
includes every tile; `diagonal=:dense` excludes diagonal tiles so the same
workspace builds the compressed off-diagonal part of a `TLRMatrix`.
"""
struct FTLRCompressionWorkspace{CatsT,StreamV,KeyT}
    cats::CatsT
    streams::StreamV
    key::KeyT
end

function _make_category_workspace(tile_shape::NTuple{2,Int}, tile_ids,
                                  U::AbstractArray{T,3}, V;
                                  block::Int=max(min(32, size(U, 2), tile_shape...), 1),
                                  p0s=nothing, q0s=nothing) where {T}
    tm, tn = tile_shape
    kout = size(U, 2)
    ntiles = size(U, 3)

    # shape checks
    size(U) == (tm, kout, ntiles) || throw(DimensionMismatch("invalid U batch shape"))
    size(V) == (tn, kout, ntiles) || throw(DimensionMismatch("invalid V batch shape"))
    length(tile_ids) == ntiles || throw(DimensionMismatch("tile_ids must match batch size"))

    # sketch width and ARA basis scratch
    backend = get_backend(U)
    S = min(kout, tm, tn)
    blk = S == 0 ? 0 : max(min(block, S), 1)
    Z = view(V, :, 1:S, :)
    Q = zeros(backend, T, tm, S, ntiles)
    ara = S == 0 ? nothing : ARAWorkspace(Q; block=blk)
    omega = zeros(backend, T, tn, blk, ntiles)
    Y = S == 0 ? zeros(backend, T, tm, 0, ntiles) : ara.Yblk

    # tile-origin device buffers, empty when the caller has none
    empty_i32 = allocate(backend, Int32, 0)
    pdev = p0s === nothing ? empty_i32 : p0s
    qdev = q0s === nothing ? empty_i32 : q0s

    return (
        tile_shape, tile_ids, U, V,
        ranks=zeros(backend, Int, ntiles),
        errors_sq=zeros(backend, Float64, ntiles),
        R_keep=min(kout, S), Z, V_tiles=_batch_views(Z, S),
        p0s=pdev, q0s=qdev, ara, Q_tiles=_batch_views(Q, S), omega,
        omega_tiles=_batch_views(omega, blk), Y_tiles=_batch_views(Y, blk),
    )
end

function _compression_batch_specs(m::Int, n::Int,
                                  tile_size::NTuple{2,Int}, diagonal::Symbol)
    diagonal in (:compressed, :dense) || throw(ArgumentError(
        "diagonal must be :compressed or :dense"))
    bm, bn = tile_size
    qm, qn = cld(m, bm), cld(n, bn)
    shapes = NTuple{2,Int}[]
    ids_by_shape = Dict{NTuple{2,Int},Vector{NTuple{2,Int}}}()
    @inbounds for j in 1:qn, i in 1:qm
        diagonal === :dense && i == j && continue
        tm = min(bm, m - (i - 1) * bm)
        tn = min(bn, n - (j - 1) * bn)
        shape = (tm, tn)
        if !haskey(ids_by_shape, shape)
            ids_by_shape[shape] = NTuple{2,Int}[]
            push!(shapes, shape)
        end
        push!(ids_by_shape[shape], (i, j))
    end
    return [(shape=shape, tile_ids=ids_by_shape[shape]) for shape in shapes]
end

function FTLRCompressionWorkspace(A::AbstractMatrix{T},
                                  tile_size::NTuple{2,Int};
                                  maxrank::Int,
                                  diagonal::Symbol=:compressed,
                                  block::Int=max(min(32, maxrank), 1)) where {T}
    m, n = size(A)
    bm, bn = tile_size
    m > 0 && n > 0 && bm > 0 && bn > 0 && maxrank >= 0 || throw(ArgumentError(
        "matrix and tile dimensions must be positive and maxrank nonnegative"))

    # one category workspace per distinct tile shape
    backend = get_backend(A)
    specs = _compression_batch_specs(m, n, tile_size, diagonal)
    cats = map(specs) do spec
        ids = spec.tile_ids
        count = length(ids)
        tm, tn = spec.shape
        p0_host = Int32[(i - 1) * bm + 1 for (i, _) in ids]
        q0_host = Int32[(j - 1) * bn + 1 for (_, j) in ids]
        p0s = copyto!(allocate(backend, Int32, count), p0_host)
        q0s = copyto!(allocate(backend, Int32, count), q0_host)
        U = zeros(backend, T, tm, maxrank, count)
        V = zeros(backend, T, tn, maxrank, count)
        _make_category_workspace(
            spec.shape, ids, U, V; block, p0s, q0s)
    end
    key = (; device=backend, T, m, n, tile_size, maxrank, diagonal)
    return FTLRCompressionWorkspace(cats, create_streams(backend, length(cats)), key)
end

FTLRCompressionWorkspace(A::AbstractMatrix, b::Int; kwargs...) =
    FTLRCompressionWorkspace(A, (b, b); kwargs...)

function _validate_compression_workspace(ws::FTLRCompressionWorkspace,
                                         A::AbstractMatrix,
                                         tile_size::NTuple{2,Int},
                                         maxrank::Int,
                                         diagonal::Symbol)
    key = ws.key
    expected = (typeof(get_backend(A)), eltype(A), size(A, 1), size(A, 2),
                tile_size, maxrank, diagonal)
    actual = (typeof(key.device), key.T, key.m, key.n, key.tile_size, key.maxrank,
              key.diagonal)
    actual == expected || throw(ArgumentError(
        "FTLRCompressionWorkspace does not match source, geometry, maxrank, " *
        "or diagonal policy"))
    return ws
end
