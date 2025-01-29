using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays
export LeftLowerTRMM!
export LeftUpperTRMM!
export RightLowerTRMM!


# Performs in place TRMM B = A * B 
# where A is an NxN lower triangular matrix and B is an NxM matrix
# A is limited to matrix size 16x16 due to shared memory constraints

@kernel function LeftLowerTRMM_kernel!(A,B,
                            ::Val{BANK} = Val(1)) where BANK
    
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if i <= N && j <= N
        @inbounds tile1[i, j] = A[i, j]
    else
        @inbounds tile1[i, j] = 0.0
        
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= R && J <= M
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize

    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in 1:i
        @inbounds out += tile1[i, k] * tile2[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize

end





# A is an NxN upper triangular matrix and B is an NxM matrix
@kernel function LeftUpperTRMM_kernel!(A,B,
                            ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if I <= N && j <= R
        @inbounds tile1[i, j] = A[I, j]
    else
        @inbounds tile1[i, j] = 0.0
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= R && J <= M
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize

    # get global values again (because of synchronize?)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in i:N
        @inbounds out += tile1[i, k] * tile2[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize
end








@kernel function RightLowerTRMM_kernel!(A,B,
                ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform M = size(B, 1)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if i <= N && j <= N
        @inbounds tile1[i, j] = A[i, j]
    else
        @inbounds tile1[i, j] = 0.0
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= M && J <= N
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize


    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in j:N
        @inbounds out += tile2[i, k] * tile1[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= M && J <= N
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize

end

@kernel function RightUpperTRMM_kernel!(A,B,
    ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform M = size(B, 1)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if i <= N && j <= N
        @inbounds tile1[i, j] = A[i, j]
    else
        @inbounds tile1[i, j] = 0.0
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= M && J <= N
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize


    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in 1:j
        @inbounds out += tile2[i, k] * tile1[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= M && J <= N
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize
end




# wrapper function for the LLTRMM kernel
function LeftLowerTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    # need to specify ndrange as the larger of the 2 ARGUMENTS
    LeftLowerTRMM_kernel!(backend, n_threads)(A, B, ndrange = max(size(A), size(B)))
end

# wrapper function for the LUTRMM kernel
function LeftUpperTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    # could not use overloading with only 2 args
    LeftUpperTRMM_kernel!(backend, n_threads)(A, B, ndrange = max(size(A), size(B)))
end

# wrapper function for the RLTRMM kernel
function RightLowerTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    # could not use overloading with only 2 args
    RightLowerTRMM_kernel!(backend, n_threads)(A, B, ndrange = max(size(A), size(B)))
end

function RightUpperTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    # could not use overloading with only 2 args
    RightUpperTRMM_kernel!(backend, n_threads)(A, B, ndrange = max(size(A), size(B)))
end