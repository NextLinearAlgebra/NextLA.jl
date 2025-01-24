using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays

const TILE_DIM = 32

@kernel function matmul!(
    output, input1, input2, N::Int, R::Int, M::Int, sum::Bool = false,
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

export GEMM_ADD!


@kernel function GEMM_ADD_kernel!(output, @Const(input1), @Const(input2),
                                    ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    #allocating shared memory for the sub matrix product calculation
    #BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(output) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(output) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    temp = @private eltype(input2) 1
    @inbounds temp[1] = -zero(eltype(input2))

    @uniform N = size(input1, 1)
    @uniform R = size(input1, 2)
    @uniform M = size(input2, 2)


    #the number of tiles required will be dependent on the inner dimensions
    @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM)
    #loop over all tiles needed for the calculation
    for t in 0:(NUM_TILES-1)
        # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t*TILE_DIM + j <= R
            @inbounds tile1[i, j] = input1[I, t*TILE_DIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t*TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = input2[t*TILE_DIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again (because of synchronize?)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(output))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        temp[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds output[I, J] += temp[1]
    end
    @synchronize

end

# wrapper function for the GEMM_ADD kernel
function GEMM_ADD!(A, B, C; nthreads = (16, 16))
    # Bupper = A*B_lower + B_upper
    backend = get_backend(A)
    kernel = GEMM_ADD_kernel!(backend, nthreads)
    kernel(C, A, B; ndrange = max(size(A), size(C)))
end