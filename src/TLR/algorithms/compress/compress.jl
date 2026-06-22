"""
    compress!(A_tlr, A; block_size=min(maxrank(A_tlr), 32), eps, ...)

Compress dense matrix `A` into the destination TLR container `A_tlr`.

This path materializes an internal packed dense-tile workspace before running
the shared adaptive randomized algorithm.

### Notes

Let `b = blocksize(A_tlr)`, `R = maxrank(A_tlr)`, `s = min(block_size, R)`,
`Ntiles = prod(size(A_tlr.layout))`, and let `B` be the number of tiles
compressed by ARA (`noffdiag_tiles(layout)` when `compress_diag=false`,
otherwise `Ntiles`).

Beyond the input dense matrix `A` and the destination TLR storage, the dominant
additional memory is:

- packed dense tiles: `sizeof(T) * b^2 * Ntiles`
- ARA floating-point workspace:
  `sizeof(T) * B * (b^2 + bR + 3bs + Rs + 2s^2)`
- smaller integer / pointer bookkeeping of order `O(B)`

So the extra workspace is linear in the number of tiles, with one packed
`b × b × Ntiles` tile buffer plus an adaptive randomized compression workspace
of size `O(B * (b^2 + bR + bs + Rs + s^2))`.
"""
function compress!(
    A_tlr::TLRMatrix{<:Any,T},
    A::AbstractMatrix{T};
    block_size::Int=min(maxrank(A_tlr), 32),
    eps,
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
) where {T}
    require_compressible_tlr(A_tlr, A)
    storage = A_tlr.storage

    if compress_diag(storage)
        tiles = allocate(A_tlr.backend, T, blocksize(A_tlr), blocksize(A_tlr), prod(size(A_tlr.layout)))
        to_tiles!(
            tiles,
            A,
            A_tlr.layout;
            ROWS_PER_WORKGROUP,
            NWORKGROUPS,
            backend=A_tlr.backend,
        )

        if size(tiles, 3) == 0
            fill!(ranks(storage), zero(eltype(ranks(storage))))
            return A_tlr
        end

        ara_batched!(
            left_factors(storage),
            right_factors(storage),
            ranks(storage),
            tiles,
            maxrank(storage),
            block_size,
            eps,
        )
        return A_tlr
    end

    packed = allocate_packed_dense_tiles(A_tlr.backend, T, A_tlr.layout)
    to_tiles!(
        packed,
        A;
        ROWS_PER_WORKGROUP,
        NWORKGROUPS,
        backend=A_tlr.backend,
    )
    copyto!(dense_diag(storage), packed.diag)

    if size(packed.offdiag, 3) == 0
        fill!(ranks(storage), zero(eltype(ranks(storage))))
        return A_tlr
    end

    ara_batched!(
        left_factors(storage),
        right_factors(storage),
        ranks(storage),
        packed.offdiag,
        maxrank(storage),
        block_size,
        eps,
    )
    return A_tlr
end

"""
    compress!(A_tlr, op; block_size=min(maxrank(A_tlr), 32), eps)

Compress a matrix-free operator `op` into the destination TLR container
`A_tlr`.
"""
function compress!(
    A_tlr::TLRMatrix{<:Any,T},
    op;
    block_size::Int=min(maxrank(A_tlr), 32),
    eps,
) where {T}
    require_compressible_tlr(A_tlr, op)
    storage = A_tlr.storage

    if !compress_diag(storage)
        materialize_diag_tiles!(dense_diag(storage), op, A_tlr.layout; backend=A_tlr.backend)
    end

    if length(ranks(storage)) == 0
        fill!(ranks(storage), zero(eltype(ranks(storage))))
        return A_tlr
    end

    ara_batched_operator!(
        left_factors(storage),
        right_factors(storage),
        ranks(storage),
        op,
        A_tlr.layout,
        maxrank(storage),
        block_size,
        eps;
        compress_diag=compress_diag(storage),
        backend=A_tlr.backend,
    )

    return A_tlr
end
