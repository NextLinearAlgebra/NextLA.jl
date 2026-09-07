"""
    adaptive_precisions(A, U=DataType[Float32, Float64], n_min=4, epsilon=1e-8; uplo='L')

Chooses a working precision for each level of the recursive splitting of `A`, and returns them
as a `Vector{DataType}` that can be passed as `precisions` to `FullMixedPrec`, `SymmMixedPrec`
or `TriMixedPrec`. `uplo` is 'L' or 'U' for a triangular or symmetric `A`, of which only that
triangle is read, and 'F' for a full one. `adaptive_precision_LT` is the same function under
its original name.

`A` is split as those containers split it. At level `k`, counted from 0 at the top, `E_k` is
the largest Frobenius norm of the level's off-diagonal blocks -- the `uplo` one, or both for
'F' -- relative to the norm of the part of `A` that is read, and the target unit roundoff is
`epsilon / (2^((k+1)/2) * E_k)`. The level gets
the coarsest type in `U` whose unit roundoff `eps(T)/2` does not exceed the target, or the
finest if none does. Splitting stops before any block would have fewer than `n_min` rows on
either side, and the result ends with the finest type in `U` for the leaves, so a matrix too
small to split returns that type alone.

`U` may list `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`, in any
order. The part of `A` that is read must be finite and not zero, `A` square and non-empty,
`uplo` 'L', 'U' or 'F', `n_min` at least 1 and `epsilon` positive.
"""
function adaptive_precisions(
    A::AbstractMatrix,
    U::AbstractVector{DataType}=DataType[Float32, Float64],
    n_min::Integer=4,
    epsilon::Real=1e-8;
    uplo::Char='L'
)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    isempty(A) && throw(ArgumentError("A must not be empty"))
    uplo in ('L', 'U', 'F') || throw(ArgumentError("uplo must be 'L', 'U' or 'F'"))
    isempty(U) && throw(ArgumentError("U must not be empty"))
    all(_supported, U) ||
        throw(ArgumentError("U must be Float8_E5M2, BFloat16, Float16, Float32 or Float64"))
    n_min >= 1 || throw(ArgumentError("n_min must be at least 1"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
    norm_A = norm(uplo == 'L' ? LowerTriangular(A) : uplo == 'U' ? UpperTriangular(A) : A)
    isfinite(norm_A) || throw(ArgumentError("A must be finite"))
    iszero(norm_A) && throw(ArgumentError("A must not be zero"))

    types = sort(unique(U); by=T -> Float64(eps(T)), rev=true)
    roundoffs = [Float64(eps(T)) / 2 for T in types]
    levels = DataType[]
    blocks = [1:n]
    k = 0
    while all(r -> length(r) >= 2 && length(r) - _rec_split(length(r)) >= n_min, blocks)
        next = UnitRange{Int}[]
        level_norm = 0.0
        for r in blocks
            mid = _rec_split(length(r))
            top, bottom = r[1:mid], r[mid+1:end]
            uplo == 'U' || (level_norm = max(level_norm, norm(view(A, bottom, top))))
            uplo == 'L' || (level_norm = max(level_norm, norm(view(A, top, bottom))))
            push!(next, top, bottom)
        end
        target = epsilon / (2^((k + 1) / 2) * (level_norm / norm_A))
        i = findfirst(<=(target), roundoffs)
        push!(levels, i === nothing ? types[end] : types[i])
        blocks = next
        k += 1
    end
    push!(levels, types[end])
    return levels
end

const adaptive_precision_LT = adaptive_precisions
