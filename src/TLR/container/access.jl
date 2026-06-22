"""Return the nominal tile size of `A`."""
@inline blocksize(A::TLRMatrix) = blocksize(A.layout)
"""Return the maximum tile rank representable in `A`."""
@inline maxrank(A::TLRMatrix) = maxrank(A.storage)
"""Return whether diagonal tiles are compressed in `A`."""
@inline compress_diag(A::TLRMatrix) = compress_diag(A.storage)
"""Return the rank buffer owned by `A.storage`."""
@inline ranks(A::TLRMatrix) = ranks(A.storage)
"""Return the dense diagonal tile storage owned by `A.storage`."""
@inline dense_diag(A::TLRMatrix) = dense_diag(A.storage)
"""Return the left low-rank factors owned by `A.storage`."""
@inline left_factors(A::TLRMatrix) = left_factors(A.storage)
"""Return the right low-rank factors owned by `A.storage`."""
@inline right_factors(A::TLRMatrix) = right_factors(A.storage)
"""Return the number of low-rank tile slots owned by `A.storage`."""
@inline nstored_tiles(A::TLRMatrix) = stored_tile_count(A.storage, A.layout)
