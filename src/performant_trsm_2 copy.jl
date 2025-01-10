using LinearAlgebra
using KernelAbstractions
using CUDA

# # Function documentation:
# """
# This function implements a GPU-accelerated algorithm to solve a system of linear equations of the form:
#     A * x = b, where A is a lower triangular matrix.

# The solution involves two steps:
# 1. **Normalization**: For each row `i`, divide both `b[i]` and the lower triangular elements of `A` by the diagonal element `A[i, i]`.
# 2. **Substitution**: Update each element of `b` by subtracting contributions from previously solved variables using the elements of `A`.

# Key assumptions:
# - The matrix `A` is lower triangular.
# - The algorithm is optimized for GPU acceleration using CUDA and KernelAbstractions.
# - The input system is not transposed, and `A` is not upper triangular.

# The result is computed in-place, modifying `B` to contain the solution `x`.
# """


@kernel function upper_left_kernel(A, B, n)
    col = @index(Group)
    row = @index(Local)

    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024

    if row <= n
        @inbounds diag[row] = A[row, row]
        @inbounds B_c[row] = B[row, col] / diag[row]
    end

    for i in n:-1:1
        @synchronize
        if row < i
            @inbounds B_c[row] -= (A[i, row] / diag[row]) * B_c[i]
        end
    end

    if row <= n
        @inbounds B[row, col] = B_c[row]
    end
end


@kernel function both_steps_parallel(A, B, n)
    col = @index(Group)
    row = @index(Local)

    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024
    shared_col = @localmem eltype(A) 1024

    if row <= n
        @inbounds diag[row] = A[row, row]
        @inbounds B_c[row] = B[row, col] / diag[row]
    end

    for i in 1:n
        @synchronize
        if row > i
            @inbounds shared_col[i] = A[i, row]
            @inbounds B_c[row] -= (shared_col[i] / diag[row]) * B_c[i]
        end
    end

    if row <= n
        @inbounds B[row, col] = B_c[row]
    end
end


@kernel function right_lower_kernel(A, B, n)
    # xA = b, A is lower triangular
    row = @index(Group)
    col = @index(Local)

    diag = @localmem eltype(A) 1024
    B_r = @localmem eltype(B) 1024
    A_row = @localmem eltype(A) 1024

    if col <= n
        @inbounds diag[col] = A[col, col]
        @inbounds B_r[col] = B[row, col] / diag[col]
    end

    for i in n:-1:1
        @synchronize
        if col < i
            @inbounds A_row[i] = A[col, i] / diag[col]
            @inbounds B_r[col] -= B_r[i] * A_row[i] 
        end
    end

    if col <= n
        @inbounds B[row, col] = B_r[col]
    end
end



@kernel function right_upper_kernel(A, B, n)
    # xA = b, A is upper triangular
    row = @index(Group)
    col = @index(Local)

    diag = @localmem eltype(A) 1024
    B_r = @localmem eltype(B) 1024
    A_row = @localmem eltype(A) 1024

    if col <= n
        @inbounds diag[col] = A[col, col]
        @inbounds B_r[col] = B[row, col] / diag[col]
    end
    

    for i in 1:n
        @synchronize
        if col > i
            @inbounds A_row[i] = A[col, i] / diag[col]
            @inbounds B_r[col] -= B_r[i] * A_row[i]
        end
    end

    if col <= n
        @inbounds B[row, col] = B_r[col]
    end
end



function performant_trsm_2_2!(
    side::Char, uplo::Char, transposed::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where T
    n, m = size(B)
    # @assert size(A, 1) == size(A, 2) == n "Matrix A must be square and match the number of rows in B"

    backend = get_backend(A)

    if side == 'L' && uplo == 'L' && transposed == 'N'
        # A is lower triangular, not transposed
        both_steps_parallel(backend, (n,))(transpose(A), B, n, ndrange=(n, m))

    elseif side == 'L' && uplo == 'U' && transposed == 'N'
        # A is upper triangular, not transposed
        upper_left_kernel(backend, (n,))(transpose(A), B, n, ndrange=(n, m))

    elseif side == 'L' && uplo == 'L' && transposed == 'T'
        # A is lower triangular, transposed
        right_lower_kernel(backend, (n,))(transpose(A), B, n, ndrange=(n, m))

    elseif side == 'L' && uplo == 'U' && transposed == 'T'
        # A is upper triangular, transposed
        right_upper_kernel(backend, (n,))(transpose(A), B, n, ndrange=(n, m))

    elseif side == 'R' && uplo == 'L' && transposed == 'N'
        # B * A, A is lower triangular, not transposed
        right_lower_kernel(backend, (m,))(transpose(A), B, m, ndrange=(m, n))

    elseif side == 'R' && uplo == 'U' && transposed == 'N'
        # B * A, A is upper triangular, not transposed
        right_upper_kernel(backend, (m,))(transpose(A), B, n, ndrange=(m, n))

    elseif side == 'R' && uplo == 'L' && transposed == 'T'
        # B * A, A is lower triangular, transposed
        both_steps_parallel(backend, (m,))(A, B, n, ndrange=(m, n))

    elseif side == 'R' && uplo == 'U' && transposed == 'T'
        # B * A, A is upper triangular, transposed
        upper_left_kernel(backend, (m,))(A, B, n, ndrange=(m, n))

    else
        error("Unsupported combination of side, uplo, and transposed parameters.")
    end

    return B
end



# TILE_DIM = 32

# @kernel function tiled_TRSM!(A, B)
#     TILE_DIM = 32
#     gi, gj = @index(Group, NTuple)
#     i, j = @index(Local, NTuple)
#     N, M = size(B)

#     # Shared memory for tiles of A and B
#     tile_A = @localmem eltype(A) (TILE_DIM, TILE_DIM)
#     tile_B = @localmem eltype(B) (TILE_DIM, TILE_DIM)

#     # Global indices
#     I = gi * TILE_DIM + i
#     J = gj * TILE_DIM + j

#     # Load A and B into shared memory
#     if I <= N && J <= N
#         @inbounds tile_A[i, j] = A[I, J]
#     else
#         @inbounds tile_A[i, j] = 0.0
#     end

#     if I <= N && J <= M
#         @inbounds tile_B[i, j] = B[I, J]
#     else
#         @inbounds tile_B[i, j] = 0.0
#     end

#     # Synchronize threads after loading data
#     @synchronize

#     # Process the tiles
#     for t in 0:(div(N, TILE_DIM) - 1)
#         # Perform triangular solve: B = A^-1 * B
#         # Iterate over the rows of the tile
#         for k in 1:TILE_DIM
#             # Update tile B using elements of A
#             tile_B[i, j] -= tile_A[i, k] * tile_B[k, j]
#         end
#     end

#     # Synchronize again
#     @synchronize

#     # Write the result back to global memory
#     if I <= N && J <= M
#         @inbounds B[I, J] = tile_B[i, j]
#     end
# end

# # function performant_trsm_2_2!(
# #     side::Char, uplo::Char, transposed::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}
# # ) where T
# #     n, m = size(B)
# #     @assert size(A, 1) == size(A, 2) == n "Matrix A must be square and match the number of rows in B"

# #     backend = get_backend(A)
# #     TILE_SIZE = 32

# #     if side == 'L' && uplo == 'L' && transposed == 'N'
# #         tiled_TRSM!(backend, (TILE_SIZE, TILE_SIZE))(transpose(A), B, ndrange=(n, m))
# #     elseif side == 'L' && uplo == 'U' && transposed == 'N'
# #         upper_left_kernel(backend, (n,))(transpose(A), B, n, ndrange=(n, m))
# #     else
# #         error("Unsupported combination of side, uplo, and transposed parameters.")
# #     end

# #     return B
# # end

# @kernel function optimized_trsm!(A, B, alpha)
#     # tx <- threadIdx.x
#     tx = @index(Local, Linear) - 1  # Equivalent to threadIdx.x, zero-based
#     bx = @index(Group, Linear) - 1  # Equivalent to blockIdx.x, zero-based

#     # Corresponds to shared T shared mem 32 * 32
#     TILE_DIM = 32
#     M = size(A, 1)
#     # NT <- M/32
#     NT = div(M, TILE_DIM)

#     # shared T shared mem 32 * 32 //shared memory to hold a tile of A
#     shmem_A = @localmem eltype(A) (TILE_DIM, TILE_DIM)
#     # Additional shared memory for B, not in pseudocode
#     shmem_B = @localmem eltype(B) TILE_DIM
#     # b <- _ldg(B[r]) //load tile B[r] into register
#     b = @private eltype(B) 1
#     # sum <- 0 //register holding the dot product
#     sum = @private eltype(B) 1
    

#     # for c <- o to NT -1 do
#     for c in 0:NT-1
#         @inbounds sum[1] = zero(eltype(B))
#         # for r <- 0 to c-1 do 
#         for r in 0:c-1
#             # shmem <- _ldg(A[r, c]) //load tile A[r, c] into shared memory
#             if tx < TILE_DIM && r*TILE_DIM + tx < M && c*TILE_DIM + tx < M
#                 @inbounds shmem_A[tx+1, tx+1] = A[r*TILE_DIM + tx + 1, c*TILE_DIM + tx + 1]
#             end
            
#             # b <- _ldg(B[r]) //load tile B[r] into register
#             if tx < TILE_DIM && r*TILE_DIM + tx < M
#                 @inbounds shmem_B[tx+1] = B[r*TILE_DIM + tx + 1, bx + 1]
#             end

#             # syncthreads()
#             @synchronize

#             # sum <- sum + sum k=0 to 31(shmem_tx,k * Shuffle(b,k)) // GEMM: B[c] += A[r, c] * B[r]
#             if tx < TILE_DIM && c*TILE_DIM + tx < M
#                 for k in 1:TILE_DIM
#                     @inbounds sum[1] += shmem_A[tx+1, k] * shmem_B[k]
#                 end
#             end

#             # synchthreads()
#             @synchronize
#         end

#         # scmem <- _ldg(A[c,c]) //load tile A[c, c] from global memory to shared memory
#         if tx < TILE_DIM && c*TILE_DIM + tx < M
#             @inbounds shmem_A[tx+1, tx+1] = A[c*TILE_DIM + tx + 1, c*TILE_DIM + tx + 1]
#         end

#         # b <- _ldg(B[c]) //load tile B[c] into registers 
#         if tx < TILE_DIM && c*TILE_DIM + tx < M
#             @inbounds b[1] = B[c*TILE_DIM + tx + 1, bx + 1]
#         end

#         # syncthreads()
#         @synchronize

#         # for j <- 0 to 31 do //perform TRSM on shared memory and registers
#         for j in 1:TILE_DIM
#             # if j=tx then
#             #    b<-(alpha*b-sum)/shmem[j,j]
#             # endif
#             if j-1 == tx && tx < TILE_DIM && c*TILE_DIM + tx < M
#                 @inbounds b[1] = (alpha * b[1] - sum[1]) / shmem_A[j, j]
#             end
#             @synchronize
#             # sum <- sum + shmem[tx + j*32] * Shuffle(b, j)
#             if tx < TILE_DIM && c*TILE_DIM + tx < M
#                 @inbounds sum[1] += shmem_A[tx+1, j] * b[1]
#             end
#             @synchronize
#         end

#         # B[c] = b //store back tile B[c] to global memory
#         if tx < TILE_DIM && c*TILE_DIM + tx < M
#             @inbounds B[c*TILE_DIM + tx + 1, bx + 1] = b[1]
#         end
#     end
# end



# function performant_trsm_2_2!(side::Char, uplo::Char, transposed::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
#     @assert side == 'L' && uplo == 'L' && transposed == 'N' "Only left, lower, non-transposed case is implemented"
#     M, N = size(B)
#     @assert size(A) == (M, M) "Matrix A must be square and match the number of rows in B"

#     TILE_DIM = 32
#     threads = (TILE_DIM, 1)
#     blocks = (N, 1)

#     backend = get_backend(A)
#     optimized_trsm!(backend, threads)(A, B, 1, ndrange=blocks)

#     return B
# end




# @kernel function trsm_kernel!(A,B,
#     ::Val{BANK} = Val(1)) where BANK

#     gi,gj = @index(Group, NTuple)
#     i,j = @index(Local, NTuple) #i is tx

#     TILE_DIM = @uniform @groupsize()[1]
#     BLOCK_ROWS = @uniform @groupsize()[2]

#     #allocating shared memory for the sub matrix product calculation
#     #BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
#     tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM) #this is the shmem
#     tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM) #instead of shuffle i think but we shall see

#     #declaring a private variable to accumulate the result of submatrix multiplication
#     sum = @private eltype(B) 1
#     @inbounds sum[1] = -zero(eltype(B))

#     @uniform N = size(A, 1)
#     @uniform R = size(A, 2)
#     @uniform M = size(B, 2)


#     #the number of tiles required will be dependent on the inner dimensions
#     @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM) #this is NT

#     #loop over all tiles needed for the calculation
#     for c in 0:(NUM_TILES-1)
#         # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
#         I = (gi-1) * TILE_DIM + i
#         J = (gj-1) * TILE_DIM + j
#         #for r from to c-1:
#             # load inputs into tiles, with bounds checking for non-square matrices
#             if I <= N && t*TILE_DIM + j <= R
#                 @inbounds tile1[i, j] = A[I, t*TILE_DIM + j] #this corresponds to shmem <- _ldg(A[r,c]??? i think. it should do that
#             else
#                 @inbounds tile1[i, j] = 0.0 #this probs not necessary
#             end
#             if t*TILE_DIM + i <= R && J <= M
#                 @inbounds tile2[i, j] = B[t*TILE_DIM + i, J] #loading b into the shared memory. should load B[r] into shared mem
#             else
#                 @inbounds tile2[i, j] = 0.0
#             end
    
#             # wait for all tiles to be loaded
#             @synchronize

#             #then we need to do 
#             # sum = sum + (from k=1 to 32) tileA of [i, k] * 'tileB of [k] 'really it is supposed to be shuffle(b, k) but i dont think kernel abstractions can do that
#         #end

        

#         #now we load A[c, c] into tile1 for a
#         # and we load B[c] into tile2 for B
#         #synch 
#         # get global values again (because of synchronize?? lol)
#         I = (gi-1) * TILE_DIM + i
#         J = (gj-1) * TILE_DIM + j # do we even need this now that i think about it? i am kinda confused what the I and J are for

#         #now we do for j is 1 to 32 i think 
#         # if j == i 
#         # tile2 = (tile2 - sum)/tile1[j, j ] <- diag
#         #end if
#         sum = sum + tile1[i+j*32] * tile2[j] #again supposed to be shuffle but i think this is prob fine
#         #end for

#         # calculate value of spot in output, use temporary value to allow for vectorization
#         out = zero(eltype(B))
#         @simd for k in 1:TILE_DIM
#             @inbounds out += tile1[i, k] * tile2[k, j] #sum += sum 1 to tile dim of tile1[]
#         end
#         sum[1] += out

#         @synchronize
#     end

#     # get global indices again
#     I = (gi-1) * TILE_DIM + i
#     J = (gj-1) * TILE_DIM + j

#     @synchronize

#     # save if inbounds
#     if I <= N && J <= M
#         @inbounds B[I, J] = tile2[#idk what index]
#     end
    
# end
