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
    
    # thread mapping
    # we map 24 warps to vertical strips of the matrix. each strip is 4 columns wide. 64 / 4 = 16 strips.
    
    tx = @index(Global, Linear)
    
    # calculate Warp ID (0-23) and Lane ID (0-31)
    warp_id = (tx - 1) ÷ 32
    lane_id = (tx - 1) % 32
    
    # variables to determine which matrix elements this thread owns
    # my_row: the row index this thread is responsible for
    # strip_idx: which vertical strip (0-15) this thread owns
    
    my_row = 0
    strip_idx = 0
    
    # warps 0-15 handle the first 8 strips (cols 1-32). these are the tall strips.
    # we use 2 Warps per strip: even warps take top half, odd warps take bottom half.
    if warp_id < 16
        strip_idx = warp_id ÷ 2
        is_bottom = (warp_id % 2) == 1
        
        # Lane 0->Row 1, Lane 1->Row 2... (plus 32 if bottom warp)
        my_row = lane_id + 1 + (is_bottom ? 32 : 0)

    # warps 16-23 handle the last 8 strips (cols 33-64). these are short strips.
    # Because it is Lower Triangular, rows 1-32 are zero here. We only need rows 33-64.
    # One Warp is enough to cover these 32 rows.
    else
        strip_idx = 8 + (warp_id - 16)
        my_row = lane_id + 33 # Start at row 33
    end

    # The first column index this thread owns (1-based)
    col_start = (strip_idx * STRIP_WIDTH) + 1
    
    # ========================================================================================
    # STEP 1: LOAD GLOBAL -> REGISTERS (The Hoard)
    # ========================================================================================
    # Create our "Notepad" (Registers). We use MArray to force register usage.
    # Each thread owns 4 consecutive columns for its specific row.
    
    my_vals = @private MVector{4, eltype(A)}(undef)
    
    # Load from Global Memory
    # Check bounds just in case, though mapping should be perfect for N=64
    if my_row <= N
        for i in 1:STRIP_WIDTH
            c = col_start + (i - 1)
            # Only load if it's a valid matrix coordinate
            if c <= N 
                @inbounds my_vals[i] = A[my_row, c]
            else
                my_vals[i] = zero(eltype(A))
            end
        end
    end

    # shared memory for broadcasting - one column at a time
    tile = @localmem eltype(A) N

    @synchronize

    # ========================================================================================
    # STEP 2: THE MAIN LOOP
    # ========================================================================================
    
    for k in 1:N
        
        # ------------------------------------------------------------------------
        # PHASE A: BROADCAST (Put active column k into Shared Memory)
        # ------------------------------------------------------------------------
        
        # Am I the owner of the data for column k?
        # Check if k falls inside my strip [col_start, col_start + 3]
        if k >= col_start && k < (col_start + STRIP_WIDTH)
            # Calculate which local register holds column k
            local_idx = (k - col_start) + 1
            
            # Write my value to the shared memory whiteboard
            # Thread responsible for 'my_row' writes to 'tile[my_row]'
            if my_row <= N
                @inbounds tile[my_row] = my_vals[local_idx]
            end
        end

        @synchronize
        
        # ------------------------------------------------------------------------
        # PHASE B: DIAGONAL & COLUMN SCALE (Math on the Whiteboard)
        # ------------------------------------------------------------------------
        # We do this math on Shared Memory so everyone sees the updated factors.
        
        # 1. Square Root the Diagonal
        # Only one thread needs to do this (e.g., Thread 1)
        if tx == 1
            @inbounds tile[k] = sqrt(tile[k])
        end
        
        @synchronize

        # 2. Scale the column (Divide by diagonal)
        # We parallelize this. Threads 1..N each handle one row of the vector.
        # Note: We reuse 'tx' here as a simple linear index for the column array.
        if tx > k && tx <= N
            @inbounds tile[tx] /= tile[k]
        end
        
        @synchronize

        # ------------------------------------------------------------------------
        # PHASE C: UPDATE REGISTERS (The "Pull" & Elimination)
        # ------------------------------------------------------------------------

        # C1: Update my own register if I owned part of Column k
        # If I owned 'A[my_row, k]', I just scaled it in Shared Memory.
        # I must copy that new scaled value back into my pocket.
        if k >= col_start && k < (col_start + STRIP_WIDTH)
            local_idx = (k - col_start) + 1
            if my_row <= N
                 @inbounds my_vals[local_idx] = tile[my_row]
            end
        end
        
        # C2: The Elimination (The heavy math)
        # A[r, c] = A[r, c] - L[r, k] * L[c, k]'
        
        # L[r, k]: The factor for MY row. It is sitting in tile[my_row].
        # L[c, k]: The factor for MY columns. They are sitting in tile[c].
        
        if my_row > k # Only update rows below the current diagonal
            
            # Load the multiplier for my row (L_rk)
            L_rk = @inbounds tile[my_row]
            
            # Loop through my 4 columns
            for i in 1:STRIP_WIDTH
                c = col_start + (i - 1)
                
                # Only update if this column is to the right of the diagonal (and valid)
                # (Lower triangular update means we only touch cols <= my_row? 
                # Cholesky fills lower triangle. We update submatrix k+1:N)
                
                if c > k && my_row >= c # Ensure we stay in lower triangle
                    
                    # Load the multiplier for this specific column (L_ck)
                    L_ck = @inbounds tile[c]
                    
                    # The Register Math
                    @inbounds my_vals[i] -= L_rk * L_ck
                end
            end
        end

        @synchronize
    end

    # ========================================================================================
    # STEP 3: WRITE BACK (Empty Pockets)
    # ========================================================================================
    
    if my_row <= N
        for i in 1:STRIP_WIDTH
            c = col_start + (i - 1)
            
            # Write back only if it's a valid coordinate in the lower triangle
            if c <= N && my_row >= c
                @inbounds A[my_row, c] = my_vals[i]
            end
        end
    end
end

function cholesky_lower_left!(A)
    N = size(A, 1)
    backend = CUDABackend()
    
    for k in 1:BLOCK_SIZE:N
        k_end = min(k + BLOCK_SIZE - 1, N)
        blk_len = k_end - k + 1
        
        if k > 1
            L_prev_cols = view(A, k:N, 1:k-1) 
            L_prev_top  = view(A, k:k_end, 1:k-1)
            A_panel     = view(A, k:N, k:k_end)
            
            CUBLAS.gemm!('N', 'T', -one(eltype(A)), L_prev_cols, L_prev_top, one(eltype(A)), A_panel)
        end
        
        A_diag = view(A, k:k_end, k:k_end)
        
        kernel = chol_kernel_register!(backend, REG_THREADS)
        kernel(A_diag, Val(blk_len); ndrange=REG_THREADS, workgroupsize=REG_THREADS)
        # KernelAbstractions.synchronize(backend)
        
        if k_end < N
            A_off_diag = view(A, (k_end + 1):N, k:k_end)
            
            # CUBLAS.trsm!('R', 'L', 'T', 'N', one(eltype(A)), A_diag, A_off_diag)
            # RightUpperTRSM!(Transpose(A_diag), A_panel)
            unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A_diag, A_off_diag)
        end
    end

    KernelAbstractions.synchronize(backend)
    return A
end