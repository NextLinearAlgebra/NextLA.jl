# Recursive Cholesky over a SymmMixedPrec container. Source: vicki-development
# @ 694d019, by Vicki <vickicar@mit.edu>, where it sat in src/cholesky_tree_subdiv.jl.

"""
    potrf_recursive!(A::SymmMixedPrec, block_size::Integer=POTRF_BLOCK_SIZE) -> A

Factors the symmetric positive definite `A` in place: `A = L * transpose(L)` with `L` stored in
the lower triangle when `A.uplo == 'L'`, or `A = transpose(U) * U` with `U` in the upper
triangle when it is `'U'`. It walks the block tree of `A`: each leaf is factored by the dense
`potrf_recursive!` with `block_size` and `A.uplo`, each off-diagonal block is solved against
the factored diagonal block before it, and the trailing block is updated with `recsyrk!`. A
matrix that is not positive definite throws `PosDefException` from a leaf; its `info` counts
within that leaf.

An off-diagonal block is solved against the `T_Base` diagonal block on its stored values,
since the solve is linear and its scale carries through, in a `T_Base` copy rounded back
once when it is narrower; a result past `floatmax` of the block's type is stored with the
block's scale raised.
A leaf stored with a scale factor throws `ArgumentError`, since its factor does not scale with it.
`reconstruct_matrix(TriMixedPrec(A))` recovers the dense factor with zeros in the other
triangle; `reconstruct_matrix(A)` would mirror it into both.
"""
function potrf_recursive!(A::SymmMixedPrec{T_Base}, block_size::Integer=POTRF_BLOCK_SIZE) where {T_Base}
    if A.Base !== nothing
        isone(A.Base_scale) ||
            throw(ArgumentError("potrf_recursive!: a leaf stored with a scale factor cannot hold its Cholesky factor; " *
                                "build A with a wider precision for the leaves (precisions[end])"))
        potrf_recursive!(A.Base, block_size; uplo=A.uplo)
        return A
    end

    potrf_recursive!(A.A11, block_size)
    side, tri = A.uplo == 'L' ? ('R', 'L') : ('L', 'U')
    A.OffDiag_scale = _solve_block!(A.OffDiag, A.OffDiag_scale, T_Base) do W
        unified_rectrxm!(side, tri, 'T', 'N', one(T_Base), 'S', TriMixedPrec(A.A11), W)
    end
    X = A.uplo == 'L' ? A.OffDiag : copy(transpose(A.OffDiag))
    recsyrk!(-(Float64(A.OffDiag_scale)^2), X, one(T_Base), A.A22)
    potrf_recursive!(A.A22, block_size)
    return A
end
