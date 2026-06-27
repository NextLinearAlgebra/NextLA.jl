@inline blocksize(A::TLRMatrix) = A.tile_m
@inline maxrank(A::TLRMatrix) = A.maxrank
@inline ranks(A::TLRMatrix) = A.ranks
@inline dense_diag(A::TLRMatrix) = A.D

"""
Return a named tuple `(interior, right, bottom)` of the left low-rank factor
arrays for each tile category.
"""
@inline left_factors(A::TLRMatrix) =
    (interior=A.int_U, right=A.right_U, bottom=A.bottom_U)

"""
Return a named tuple `(interior, right, bottom)` of the right low-rank factor
arrays for each tile category.
"""
@inline right_factors(A::TLRMatrix) =
    (interior=A.int_V, right=A.right_V, bottom=A.bottom_V)

# ─── Per-tile factor accessors ────────────────────────────────────────────────

"""
    tile_u(A, ob) → AbstractMatrix

Left low-rank factor for off-diagonal tile `ob`, trimmed to its effective rank.
The returned view aliases the underlying storage — no allocation.
"""
@inline function tile_u(A::TLRMatrix, ob::Integer)
    k = A.local_index[ob]
    r = Int(A.ranks[ob])
    cat = A.category[ob]
    if cat == _TILE_INT
        return view(A.int_U, :, 1:r, k)
    elseif cat == _TILE_RIGHT
        return view(A.right_U, :, 1:r, k)
    else
        return view(A.bottom_U, :, 1:r, k)
    end
end

"""
    tile_v(A, ob) → AbstractMatrix

Right low-rank factor for off-diagonal tile `ob`, trimmed to its effective rank.
The returned view aliases the underlying storage — no allocation.
"""
@inline function tile_v(A::TLRMatrix, ob::Integer)
    k = A.local_index[ob]
    r = Int(A.ranks[ob])
    cat = A.category[ob]
    if cat == _TILE_INT
        return view(A.int_V, :, 1:r, k)
    elseif cat == _TILE_RIGHT
        return view(A.right_V, :, 1:r, k)
    else
        return view(A.bottom_V, :, 1:r, k)
    end
end
