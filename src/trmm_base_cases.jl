using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays


# Kernel function for solving lower triangular system Ax = b
@kernel function lower_left_trmm_kernel(A, B, C, ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    #allocating shared memory for the sub matrix product calculation
    #BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(C) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(C) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    C_sub = @private eltype(C) 1
    @inbounds C_sub[1] = -zero(eltype(C))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)


    #the number of tiles required will be dependent on the inner dimensions
    @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM)

    #loop over all tiles needed for the calculation
    for t in 0:(NUM_TILES-1)
        # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t*TILE_DIM + j <= R
            @inbounds tile1[i, j] = A[I, t*TILE_DIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t*TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = B[t*TILE_DIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again (because of synchronize?)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(C))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        C_sub[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds C[I, J] = C_sub[1]
    end
    @synchronize
end

# Kernel function for solving upper triangular system Ax = b
@kernel function upper_left_trmm_kernel(A, B, n)
    #implementation goes here
end

# Kernel function for solving lower triangular system xA = b
@kernel function right_lower_trmm_kernel(A, B, n)
    #implementation goes here
end

# Kernel function for solving upper triangular system xA = b
@kernel function right_upper_trmm_kernel(A, B, n)
    #implementation goes here
end
