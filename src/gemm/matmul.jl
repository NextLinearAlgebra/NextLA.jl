export GEMM_ADD!, GEMM_SUB!

const TILE_DIM = 32

# The scale factor takes the destination's element type rather than Float64.
# It was pinned to Float64 in the kernel signature and converted at the launch,
# so every call multiplied in double precision inside the kernel whatever the
# matrices were -- on every backend, and on a device with no Float64 at all it
# cannot run. The accumulator was already @private eltype(output), so only this
# one multiply changes. `true` is the default because it is the multiplicative
# identity that promotes to whatever it meets rather than widening.
# TAcc is the type the tiles and the running sum are kept in: eltype(output), or
# Float32 for a Float16 product into a Float16 output, so half precision
# accumulates wider than it is stored without a Float32 copy of the output. The
# sum is rounded once, on store.
@kernel function matmul!(
    output, input1, input2, N::Int, R::Int, M::Int, ::Type{TAcc}, alpha = true, transA::Char = 'N', transB::Char = 'N',
    ::Val{BANK} = Val(1)
) where {TAcc, BANK}

    gi, gj = @index(Group,   NTuple)
    i,  j  = @index(Local,   NTuple)

    TILE_DIM = @uniform @groupsize()[1]

    tile1 = @localmem TAcc (TILE_DIM + BANK, TILE_DIM)
    tile2 = @localmem TAcc (TILE_DIM + BANK, TILE_DIM)

    outval = @private TAcc 1
    @inbounds outval[1] = zero(TAcc)

    @uniform NUM_TILES = ceil(Int, R / TILE_DIM)

    # I and J are recomputed in every block delimited by @synchronize. On the
    # CPU backend a @synchronize splits the kernel body into separate ndrange
    # loops, and only @localmem/@private survive the split -- a plain local
    # assigned before it is simply out of scope afterwards, at which point `I`
    # resolves to LinearAlgebra.I and the kernel dies with
    # `isless(::UniformScaling{Bool}, ::Int64)`. That made GEMM_ADD! and
    # GEMM_SUB!, and so the recursive TRMM path, unusable on CPU.
    for t in 0:(NUM_TILES - 1)
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        K = t * TILE_DIM + j

        if I <= N && K <= R
            @inbounds tile1[i, j] =
                (transA == 'N' || transA == 'n') ? input1[I, K] :
                (transA == 'T' || transA == 't') ? input1[K, I] :
                                                   conj(input1[K, I])  # 'C', checked at launch
        else
            @inbounds tile1[i, j] = zero(TAcc)
        end

        K = t * TILE_DIM + i
        if K <= R && J <= M
            @inbounds tile2[i, j] =
                (transB == 'N' || transB == 'n') ? input2[K, J] :
                (transB == 'T' || transB == 't') ? input2[J, K] :
                                                   conj(input2[J, K])  # 'C', checked at launch
        else
            @inbounds tile2[i, j] = zero(TAcc)
        end

        @synchronize

        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        if I <= N && J <= M
            tmp = zero(TAcc)
            @simd for k in 1:TILE_DIM
                @inbounds tmp += tile1[i, k] * tile2[k, j]
            end
            outval[1] += tmp
        end

        @synchronize
    end

    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j
    if I <= N && J <= M
        @inbounds output[I, J] += alpha * outval[1]
    end
end

# A Transpose or Adjoint operand is unwrapped into the kernel's transA/transB
# flags rather than materialised. matmul! has always taken those flags and an
# alpha; the wrappers below simply never passed them.
@inline _untranspose(X) = X isa Transpose ? (parent(X), 'T') :
                          X isa Adjoint   ? (parent(X), 'C') : (X, 'N')

# The kernel reads its operands with @inbounds, so shapes that do not fit would
# read or write past the arrays rather than throw.
function _check_matmul_dims(A, B, C)
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("A is $(size(A)) and B is $(size(B)), so the result is " *
                                "$((size(A, 1), size(B, 2))), but C is $(size(C))"))
    return nothing
end

# Accumulating a Float16 product in Float16 loses the running sum and can
# overflow part way, so with all three Float16 the kernel keeps its tiles and sum
# in Float32 and rounds once, on store; a result beyond floatmax(Float16) is Inf.
@inline _needs_wide_accum(A, B, C) =
    eltype(A) == Float16 && eltype(B) == Float16 && eltype(C) == Float16

function _matmul_launch!(output, in1, in2, N, R, M, alpha)
    backend = get_backend(output)
    A_mat, transA = _untranspose(in1)
    B_mat, transB = _untranspose(in2)
    transA in ('N', 'T', 'C') && transB in ('N', 'T', 'C') ||
        throw(ArgumentError("transA and transB must be 'N', 'T' or 'C'"))
    TAcc = _needs_wide_accum(in1, in2, output) ? Float32 : eltype(output)
    matmul!(backend, (TILE_DIM, TILE_DIM))(
        output, A_mat, B_mat, N, R, M, TAcc, TAcc(alpha), transA, transB,
        ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    return output
end

"""
    GEMM_ADD!(A, B, C, scale=1) -> C

`C .+= scale * A * B`. `A` or `B` may be a `Transpose` or an `Adjoint`. A `Float16`
`C` accumulates in `Float32`; a result beyond `floatmax(Float16)` is stored as `Inf`.
Shapes that do not fit throw `DimensionMismatch`.
"""
function GEMM_ADD!(A, B, C, scale::Number = 1)
    _check_matmul_dims(A, B, C)
    N, R, M = size(A, 1), size(B, 1), size(B, 2)
    return _matmul_launch!(C, A, B, N, R, M, scale)
end

"""
    GEMM_SUB!(A, B, C, scale=1) -> A

`A .-= scale * B * C`. Note the output is the *first* argument here, unlike
`GEMM_ADD!` — both orders match the callers in `src/trsm/rectrxm.jl`. Operands,
`Float16` accumulation and shape checks are as for `GEMM_ADD!`.
"""
function GEMM_SUB!(A, B, C, scale::Number = 1)
    _check_matmul_dims(B, C, A)
    N, R, M = size(A, 1), size(C, 1), size(A, 2)
    return _matmul_launch!(A, B, C, N, R, M, -scale)
end