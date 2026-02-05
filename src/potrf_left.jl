using KernelAbstractions
using CUDA
using LinearAlgebra


const REG_THREADS = 768 
const N_MATRIX = 64
const STRIP_WIDTH = 4

# left looking cholesky kernel
# @kernel function chol_kernel_left!(A, N)
#     tx = @index(Global, Linear)
    
#     # shared memory alloc
#     tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE)

#     # load into shmem
#     total_elements = N * N
#     idx = tx
#     while idx <= total_elements
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1
#         s_idx = (c - 1) * STRIDE + r
        
#         @inbounds tile[s_idx] = A[r, c]
#         idx += MAX_THREADS
#     end

#     @synchronize

#     # main factorization
#     for k in 1:N
        
#         # sqrt diag
#         if tx == 1
#             idx_kk = (k - 1) * STRIDE + k
#             @inbounds tile[idx_kk] = sqrt(tile[idx_kk])
#         end
#         @synchronize

#         # take col k
#         diag_val = tile[(k - 1) * STRIDE + k]
        
#         # scale rows i > k in column k
#         i = tx
#         while i <= N
#             if i > k
#                 idx_ik = (k - 1) * STRIDE + i
#                 @inbounds tile[idx_ik] /= diag_val
#             end
#             i += MAX_THREADS
#         end
#         @synchronize

#         # update trailing submatrix A[i, j] = A[i, j] - L[i, k] * L[j, k]' for all i,j > k
        
#         # assign threads to handle rows 'i' in the trailing submatrix
#         i = tx
#         while i <= N
#             if i > k
                
#                 # load L[i, k] (the factor from the pivot column) into a register
#                 idx_ik = (k - 1) * STRIDE + i
#                 @private L_ik = @inbounds tile[idx_ik] 

#                 for j in (k + 1):i
#                     idx_jk = (k - 1) * STRIDE + j  
#                     idx_ij = (j - 1) * STRIDE + i  

#                     @inbounds tile[idx_ij] = muladd(-L_ik, tile[idx_jk], tile[idx_ij])
#                 end
#             end
#             i += MAX_THREADS
#         end
#         @synchronize
#     end

#     # write back
#     idx = tx
#     while idx <= total_elements
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1
#         s_idx = (c - 1) * STRIDE + r
#         @inbounds A[r, c] = tile[s_idx]
#         idx += MAX_THREADS
#     end
# end

#put everything into register with every warp owning 4 consecutive columns - except first half of matrix which is owned by two warps per column (bc we are working on lower triangular)
#do diag and update column step and then put that into shared memory 
#do elimination for 4 consecutive elements in a thread (owned by the thread bc eafch thread will own 4 elmts) and pull from shmem only once per that step. pull tiel[r,k] from register and top from shmem do directlyu from shmem bc will broadcast so not a pronlem
#768 threads

#right looking cholesky kernel
# @kernel cpu=false inbounds=true unsafe_indices=false function chol_kernel_lower!(A, ::Val{N}) where N
#     tx = @index(Global, Linear) #change global to local? 

#     # put block into shared memory 
#     tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE) #stride prevents bank conflicts [64][65]

#     total_elements = N * N
#     idx = tx

#     #load into shared memory 
#     while idx <= total_elements
#         # julia is column-major
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1

#         s_idx = (c - 1) * STRIDE + r
        
#         @inbounds tile[s_idx] = A[r, c]
#         idx += MAX_THREADS
#     end

#     @synchronize

#     #iterate thru diagonal k
#     for k in 1:N
#         # one thread does sqrt
#         diag_idx = (k - 1) * STRIDE + k
#         if tx == 1
#             @inbounds tile[diag_idx] = sqrt(tile[diag_idx])
#         end

#         @synchronize

#         # division is now parallelized 
#         # divide col by diag
#         diag = @inbounds tile[diag_idx]
#         idx = k + tx 
#         while idx <= N
#             s_idx = (k - 1) * STRIDE + idx
#             @inbounds tile[s_idx] /= diag
#             idx += MAX_THREADS
#         end

#         @synchronize

#         # Elimination step
#         # updates submatrix to right/bottom
#         # treat the 2D submatrix as a 1D flat array to balance work evenly (quicker than doing it by row)
        
#         len = Int32(N - k) #size of submatrix
#         tx_32 = Int32(tx)
#         if len > 0
#             limit = len * len #total items to process

#             # map thread ID to the starting index in the flattened submatrix
#             t_idx = tx_32 - Int32(1) 
#             stride = Int32(MAX_THREADS)

#             # initial r, c
#             col_offset = div(t_idx, len)
#             row_offset = rem(t_idx, len)

#             # precalculate how much R/C change per stride to avoid division inside loop
#             stride_c = div(stride, len)
#             stride_r = rem(stride, len)

#             #register for tile[idx_ck] (because it is repeated sometimes)
#             last_c = Int32(-1)
#             current_L_ck = zero(eltype(A))
            
#             # loop until this thread has finished its share of the submatrix
#             while t_idx < limit
                
#                 #actual r, c
#                 c = col_offset + Int32(k + 1)
#                 r = row_offset + Int32(k + 1)
                
#                 # the top multiplier (tile[k, c]) stays the same for a whole column
#                 # if 'c' hasn't changed, reuse the value from the register.
#                 if c != last_c
#                     idx_ck = (k - 1) * STRIDE + c
#                     current_L_ck = @inbounds tile[idx_ck]
#                     last_c = c
#                 end
#                 # indices for the Target (rc) and the Left Multiplier (rk)
#                 idx_rc = (c - 1) * STRIDE + r
#                 idx_rk = (k - 1) * STRIDE + r
                
#                 # perform the update: A[r,c] = A[r,c] - L[r,k] * L[c,k]
#                 # idx_ck = (k - 1) * STRIDE + c
#                 # use muladd instead of * and - for speed
#                 # @inbounds tile[idx_rc] = muladd(-tile[idx_rk], current_L_ck, tile[idx_rc])
#                 @inbounds tile[idx_rc] -= tile[idx_rk] * current_L_ck

#                 # manual index updates to avoid modulo operations; update by stride
#                 t_idx += stride
#                 col_offset += stride_c
#                 row_offset += stride_r

#                 #wrap around logic
#                 if row_offset >= len
#                     row_offset -= len
#                     col_offset += Int32(1)
#                 end
                
#             end
#         end

#         @synchronize
#     end

#     # write results back to global memory - have a warp dedicated to write back (as write warps)??? but cant do it after sync?
#     idx = tx
#     while idx <= total_elements
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1
        
#         s_idx = (c - 1) * STRIDE + r
        
#         # @inbounds A[r, c] = tile[s_idx]

#         if r >= c
#             @inbounds A[r, c] = tile[s_idx]
#         end
#         idx += MAX_THREADS
#     end
# end

#version w row in register
# @kernel function chol_kernel_lower!(A, N)
#     tx = @index(Global, Linear)
    
#     # shmem
#     tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE)

#     # load into shmem
#     total_elements = N * N
#     idx = tx
#     while idx <= total_elements
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1
#         s_idx = (c - 1) * STRIDE + r
#         @inbounds tile[s_idx] = A[r, c]
#         idx += MAX_THREADS
#     end

#     @synchronize

#     # main loop
#     for k in 1:N
        
#         # sqrt step
#         diag_idx = (k - 1) * STRIDE + k
#         if tx == 1
#             @inbounds tile[diag_idx] = sqrt(tile[diag_idx])
#         end
#         @synchronize

#         # read diagonal once into reg
#         diag_val = @inbounds tile[diag_idx] 
        
#         r_idx = k + tx
#         while r_idx <= N
#             s_idx = (k - 1) * STRIDE + r_idx
#             @inbounds tile[s_idx] /= diag_val
#             r_idx += MAX_THREADS
#         end
#         @synchronize

#         # elimination
#         # assign threads to rows
#         r = (k + 1) + (tx - 1)
        
#         while r <= N
#             # load L[r, k] into a register variable `L_rk`.
#             L_rk_idx = (k - 1) * STRIDE + r
#             L_rk = @inbounds tile[L_rk_idx]

#             # iterate over columns c 
#             for c in (k + 1):r
#                 L_ck_idx = (k - 1) * STRIDE + c
                
#                 A_rc_idx = (c - 1) * STRIDE + r

#                 L_ck = @inbounds tile[L_ck_idx]

#                 @inbounds tile[A_rc_idx] = muladd(-L_rk, L_ck, tile[A_rc_idx])
#             end
            
#             # move to next row
#             r += MAX_THREADS
#         end

#         @synchronize
#     end

#     # write to global mem
#     idx = tx
#     while idx <= total_elements
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1
#         s_idx = (c - 1) * STRIDE + r
#         @inbounds A[r, c] = tile[s_idx]
#         idx += MAX_THREADS
#     end
# end



@kernel cpu=false inbounds=true unsafe_indices=false function chol_kernel_register!(A, ::Val{N}) where N
    # ---------------------------------------------------------------------
    # 0. thread mapping
    # ---------------------------------------------------------------------
    # ok so we need to figure out which part of the matrix this specific thread owns.
    # we have 768 threads total (24 warps) and we're splitting the 64x64 matrix 
    # into vertical strips that are 4 columns wide.
    
    tx = @index(Global, Linear)
    
    warp_id = (tx - 1) ÷ 32
    lane_id = (tx - 1) % 32
    
    my_row = 0
    strip_idx = 0
    
    # logic for warps 0-15 (cols 1-32):
    # these columns are "tall" (full height), so one warp isn't enough.
    # we use 2 warps per strip here. even warps take the top half, odd warps take the bottom.
    if warp_id < 16
        strip_idx = warp_id ÷ 2
        is_bottom = (warp_id % 2) == 1
        my_row = lane_id + 1 + (is_bottom ? 32 : 0)

    # logic for warps 16-23 (cols 33-64):
    # these columns are "short" because it's a lower triangular matrix.
    # rows 1-32 are just zeros here, so we only care about rows 33-64.
    # one warp is enough to cover that.
    else
        strip_idx = 8 + (warp_id - 16)
        my_row = lane_id + 33 
    end

    # calculating the start index of the 4 columns this thread is responsible for
    col_start = (strip_idx * STRIP_WIDTH) + 1
    
    # ---------------------------------------------------------------------
    # 1. load global -> registers
    # ---------------------------------------------------------------------
    # defining our private register array to hold the 4 values.
    # using ka's native @private syntax instead of mvector to keep the compiler happy.
    my_vals = @private eltype(A) (4,)
    
    # grabbing the data from global memory.
    # we gotta make sure we don't read out of bounds.
    if my_row <= N
        for i in 1:STRIP_WIDTH
            c = col_start + (i - 1)
            if c <= N 
                @inbounds my_vals[i] = A[my_row, c]
            else
                # if we're padding or out of bounds, just fill with zero
                my_vals[i] = zero(eltype(A))
            end
        end
    end

    # allocating a sliver of shared memory.
    # we only need enough space to hold ONE column (the active one) at a time.
    tile = @localmem eltype(A) N

    # making sure everyone is loaded up before we start mathing
    @synchronize

    # ---------------------------------------------------------------------
    # 2. main loop
    # ---------------------------------------------------------------------
    # iterating down the diagonal...
    for k in 1:N
        
        # --- a. broadcast active column ---
        # if i own the data for the current column k, i need to share it with the class.
        # copying from my private register to the shared memory tile.
        if k >= col_start && k < (col_start + STRIP_WIDTH)
            local_idx = (k - col_start) + 1
            if my_row <= N
                @inbounds tile[my_row] = my_vals[local_idx]
            end
        end

        # wait for the owner to finish writing to shared mem
        @synchronize
        
        # --- b. sqrt & scale ---
        # step 1: square root the diagonal element (just one thread does this)
        if tx == 1
            @inbounds tile[k] = sqrt(tile[k])
        end
        
        @synchronize

        # step 2: normalize the rest of the column by dividing by the diagonal.
        # all threads help out here to make it fast.
        if tx > k && tx <= N
            @inbounds tile[tx] /= tile[k]
        end
        
        @synchronize

        # --- c. update registers ---
        
        # c1. update my own pocket
        # if i owned that column k originally, the values in shared mem just got updated/scaled.
        # i need to pull those new values back into my private register so i stay up to date.
        if k >= col_start && k < (col_start + STRIP_WIDTH)
            local_idx = (k - col_start) + 1
            if my_row <= N
                 @inbounds my_vals[local_idx] = tile[my_row]
            end
        end
        
        # c2. elimination (the outer product update)
        # this is the heavy lifting: A = A - L * L'
        # we only update if we are below the current diagonal (row > k)
        if my_row > k 
            # grabbing the multiplier for my row
            L_rk = @inbounds tile[my_row]
            
            for i in 1:STRIP_WIDTH
                c = col_start + (i - 1)
                # only update valid columns that are to the right of the diagonal
                # keeping it strictly lower triangular
                if c > k && my_row >= c 
                    L_ck = @inbounds tile[c]
                    # doing the math directly in registers. fast!
                    @inbounds my_vals[i] -= L_rk * L_ck
                end
            end
        end

        @synchronize
    end

    # ---------------------------------------------------------------------
    # 3. write back
    # ---------------------------------------------------------------------
    # all done. dumping the registers back to global memory.
    if my_row <= N
        for i in 1:STRIP_WIDTH
            c = col_start + (i - 1)
            if c <= N && my_row >= c
                @inbounds A[my_row, c] = my_vals[i]
            end
        end
    end
end

function cholesky_lower_left!(A)
    N = size(A, 1)
    backend = CUDABackend()
    
    # looping through the matrix in 64x64 blocks
    for k in 1:N_MATRIX:N
        k_end = min(k + N_MATRIX - 1, N)
        blk_len = k_end - k + 1
        
        # if not the first block, we need to update the current panel using previous results
        if k > 1
            L_prev_cols = view(A, k:N, 1:k-1) 
            L_prev_top  = view(A, k:k_end, 1:k-1)
            A_panel     = view(A, k:N, k:k_end)
            
            CUBLAS.gemm!('N', 'T', -one(eltype(A)), L_prev_cols, L_prev_top, one(eltype(A)), A_panel)
        end
        
        # grabbing the diagonal block to solve with our custom kernel
        A_diag = view(A, k:k_end, k:k_end)
        
        kernel = chol_kernel_register!(backend, REG_THREADS)
        # crucial: we must force workgroupsize to be 768 so all threads are in the same block
        kernel(A_diag, Val(blk_len); ndrange=REG_THREADS, workgroupsize=REG_THREADS)
        
        # update the panel to the right if we aren't at the end yet
        if k_end < N
            A_off_diag = view(A, (k_end + 1):N, k:k_end)
            unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A_diag, A_off_diag)
        end
    end

    KernelAbstractions.synchronize(backend)
    return A
end