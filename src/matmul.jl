using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays

const TILE_DIM = 32

@kernel function coalesced_matmul_kernel!(
    output, @Const(input1), @Const(input2), N, R, M, sum,
    ::Val{BANK} = Val(1)
) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]

    tile1 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)
    tile2 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)

    outval = @private eltype(output) 1
    @inbounds outval[1] = zero(eltype(output))

    @uniform NUM_TILES = ceil(Int, R / TILE_DIM)

    for t in 0:(NUM_TILES - 1)
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        K = t * TILE_DIM + j

        if I <= N && K <= R
            @inbounds tile1[i, j] = input1[I, K]
        else
            @inbounds tile1[i, j] = zero(eltype(output))
        end

        K = t * TILE_DIM + i
        if K <= R && J <= M
            @inbounds tile2[i, j] = input2[K, J]
        else
            @inbounds tile2[i, j] = zero(eltype(output))
        end

        @synchronize

        if I <= N && J <= M
            out = zero(eltype(output))
            @simd for k in 1:TILE_DIM
                @inbounds out += tile1[i, k] * tile2[k, j]
            end
            outval[1] += out
        end

        @synchronize
    end

    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    if I <= N && J <= M
        if sum
            @inbounds output[I, J] += outval[1]  # Add the result
        else
            @inbounds output[I, J] -= outval[1]  # Subtract the result
        end
    end
end
