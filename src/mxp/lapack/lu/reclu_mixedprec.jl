# The mixed-precision lu_recursive! spawns a level's two off-diagonal solves on
# separate threads while the level has more than this many rows, by default.
# recgemm!'s value for the same decision; not tuned.
const RECLU_PARALLEL_THRESHOLD = 512

"""
    lu_recursive!(A::FullMixedPrec, block_size::Integer=LU_BLOCK_SIZE;
                  parallel=size(A, 1) > RECLU_PARALLEL_THRESHOLD) -> A

Factors `A = L * U` in place without pivoting, walking the block tree of `A`: each leaf is
factored by the dense `lu_recursive!` with `block_size`, each off-diagonal block is solved
against the factored diagonal block before it, and the trailing block is updated with
`recgemm!`. `L` has a unit diagonal and is stored below it, `U` on and above it, as LAPACK's
`getrf` stores them without pivots, so `A` must not need pivoting -- a diagonally dominant
matrix, for example.

An off-diagonal block is solved against the `T_Base` diagonal block on its stored values,
since the solve is linear and its scale carries through, in a `T_Base` copy rounded back once
when it is narrower; a result past `floatmax` of the block's type is stored with the block's
scale raised. A leaf stored with a scale factor throws `ArgumentError`, since one scale cannot
hold both `L` and `U`.

With `parallel`, the two off-diagonal solves of a level, which are independent, run on
separate threads at the top level and, below it, at each level with more than
`RECLU_PARALLEL_THRESHOLD` rows; the trailing `recgemm!` uses its own threshold. `parallel =
false` runs the whole factorization on the calling thread.

# Examples
```jldoctest
julia> using NextLA.MixedPrecision, LinearAlgebra

julia> A = rand(1024, 1024) + 1024I;   # diagonally dominant: no pivoting needed

julia> M = FullMixedPrec(A; precisions = DataType[Float32, Float64]);

julia> lu_recursive!(M);

julia> F = reconstruct_matrix(M);   # L below the diagonal, U on and above it

julia> isapprox(UnitLowerTriangular(F) * UpperTriangular(F), A; rtol = 1e-5)
true
```
"""
function lu_recursive!(A::FullMixedPrec{T_Base}, block_size::Integer=LU_BLOCK_SIZE;
                       parallel::Bool=(size(A, 1) > RECLU_PARALLEL_THRESHOLD)) where {T_Base}
    if A.Base !== nothing
        isone(A.Base_scale) ||
            throw(ArgumentError("lu_recursive!: a leaf stored with a scale factor cannot hold both L and U; " *
                                "build A with a wider precision for the leaves (precisions[end])"))
        lu_recursive!(A.Base, block_size)
        return A
    end

    lu_recursive!(A.A11, block_size; parallel=parallel && size(A.A11, 1) > RECLU_PARALLEL_THRESHOLD)
    function solve12()
        A.A12_scale = _solve_mixedprec_offdiag_block!(A.A12, A.A12_scale, T_Base) do W
            unified_rectrxm!('L', 'L', 'N', 'U', one(T_Base), 'S', A.A11, W)
        end
    end
    function solve21()
        A.A21_scale = _solve_mixedprec_offdiag_block!(A.A21, A.A21_scale, T_Base) do W
            unified_rectrxm!('R', 'U', 'N', 'N', one(T_Base), 'S', A.A11, W)
        end
    end
    if parallel
        @sync begin
            Threads.@spawn solve12()
            Threads.@spawn solve21()
        end
    else
        solve12()
        solve21()
    end
    recgemm!(-(Float64(A.A21_scale) * Float64(A.A12_scale)), A.A21, A.A12, one(T_Base), A.A22;
             parallel=parallel && size(A.A22, 1) > RECGEMM_PARALLEL_THRESHOLD)
    lu_recursive!(A.A22, block_size; parallel=parallel && size(A.A22, 1) > RECLU_PARALLEL_THRESHOLD)
    return A
end
