_norm_mixedprec_block(X) = sizeof(eltype(X)) < 4 ? sqrt(sum(x -> Float64(x)^2, X)) : norm(X)

"""
    adaptive_precisions(A, U=DataType[Float32, Float64], n_min=4, epsilon=1e-8;
                        uplo='L') -> Vector{DataType}

Chooses a working precision for each level of the recursive splitting of `A`, and returns them
as a `Vector{DataType}` that can be passed as `precisions` to `FullMixedPrec`, `SymmMixedPrec`
or `TriMixedPrec`. `uplo` is 'L' or 'U' for a triangular or symmetric `A`, of which only that
triangle is read, and 'F' for a full one.

`A` is split as those containers split it. At level `k`, counted from 0 at the top, `E_k` is
the largest Frobenius norm of the level's off-diagonal blocks -- the `uplo` one, or both for
'F' -- relative to the norm of the part of `A` that is read, both taken in `Float64` for a
16- or 8-bit `A` so they neither overflow nor underflow, and the target unit roundoff is
`epsilon / (2^((k+1)/2) * E_k)`. The level gets the coarsest type, no finer than the leaves',
whose unit roundoff `eps(T)/2` does not exceed the target, or the leaves' type if none does.
Splitting stops before any block would have fewer than `n_min` rows on either side, and the
result ends with the leaves' type, so a matrix too small to split returns that type alone.

The leaves get the coarsest type in `U` that is at least as fine as `eltype(A)` and has the
range of `Float32`, so a `Float32` matrix gets `Float32` leaves with the default `U`; no level
uses a type finer than the leaves', since `A` has no digits for it. When `eltype(A)` is not a
floating-point type or no type in `U` qualifies, the leaves get the finest type in `U`.

`U` may list `Float8_E5M2` (q52), `BFloat16`, `Float16`, `Float32` and `Float64`, in any
order. The part of `A` that is read must be finite and not zero, `A` square and non-empty,
`uplo` 'L', 'U' or 'F', `U` non-empty, `n_min` at least 1 and `epsilon` positive; a non-square
`A` throws `DimensionMismatch` and any other violation an `ArgumentError`. The result builds a
container only when its last type meets the containers' leaf rule, `Float32` or `Float64`,
which it does whenever `U` holds one of them.
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
    all(_is_block_precision, U) ||
        throw(ArgumentError("U must be Float8_E5M2, BFloat16, Float16, Float32 or Float64"))
    n_min >= 1 || throw(ArgumentError("n_min must be at least 1"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))

    types = sort(unique(U); by=T -> Float64(eps(T)), rev=true)
    data_eps = eltype(A) <: AbstractFloat ? Float64(eps(eltype(A))) : 0.0
    i_leaf = findfirst(T -> Float64(eps(T)) <= data_eps && floatmax(T) >= floatmax(Float32), types)
    types = types[1:something(i_leaf, end)]

    level_norms = Float64[]
    norm_A = 0.0
    blocks = [1:n]
    while all(r -> length(r) >= 2 && length(r) - _rec_split(length(r)) >= n_min, blocks)
        next = UnitRange{Int}[]
        level_norm = 0.0
        for r in blocks
            mid = _rec_split(length(r))
            top, bottom = r[1:mid], r[mid+1:end]
            if uplo != 'U'
                b = _norm_mixedprec_block(view(A, bottom, top))
                level_norm, norm_A = max(level_norm, b), hypot(norm_A, b)
            end
            if uplo != 'L'
                b = _norm_mixedprec_block(view(A, top, bottom))
                level_norm, norm_A = max(level_norm, b), hypot(norm_A, b)
            end
            push!(next, top, bottom)
        end
        push!(level_norms, level_norm)
        blocks = next
    end
    for r in blocks
        leaf = view(A, r, r)
        norm_A = hypot(norm_A, _norm_mixedprec_block(uplo == 'L' ? tril(leaf) : uplo == 'U' ? triu(leaf) : leaf))
    end
    isfinite(norm_A) || throw(ArgumentError("A must be finite"))
    iszero(norm_A) && throw(ArgumentError("A must not be zero"))

    roundoffs = [Float64(eps(T)) / 2 for T in types]
    levels = DataType[]
    for (k, level_norm) in enumerate(level_norms)
        target = epsilon / (2^(k / 2) * (level_norm / norm_A))
        i = findfirst(<=(target), roundoffs)
        push!(levels, i === nothing ? types[end] : types[i])
    end
    push!(levels, types[end])
    return levels
end
