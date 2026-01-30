using KernelAbstractions
using CUDA
using LinearAlgebra


const MAX_THREADS = 512 
const BLOCK_SIZE = 64
const PAD = 1
const STRIDE = BLOCK_SIZE + PAD

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


#right looking cholesky kernel
@kernel function chol_kernel_lower!(A, N)
    tx = @index(Global, Linear)

    # put block into shared memory 
    tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE) #stride prevents bank conflicts

    total_elements = N * N
    idx = tx

    #load into shared memory 
    while idx <= total_elements
        # julia is column-major
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1

        s_idx = (c - 1) * STRIDE + r
        
        @inbounds tile[s_idx] = A[r, c]
        idx += MAX_THREADS
    end

    @synchronize

    #iterate thru diagonal k
    for k in 1:N
        # one thread does sqrt
        diag_idx = (k - 1) * STRIDE + k
        if tx == 1
            @inbounds tile[diag_idx] = sqrt(tile[diag_idx])
        end

        @synchronize

        # division is now parallelized 
        # divide col by diag
        diag = @inbounds tile[diag_idx]
        idx = k + tx 
        while idx <= N
            s_idx = (k - 1) * STRIDE + idx
            @inbounds tile[s_idx] /= diag
            idx += MAX_THREADS
        end

        @synchronize

        # Elimination step
        # updates submatrix to right/bottom
        # treat the 2D submatrix as a 1D flat array to balance work evenly (quicker than doing it by row)
        
        # len = Int32(N - k) #size of submatrix
        # tx_32 = Int32(tx)
        # if len > 0
        #     limit = len * len #total items to process

        #     # map thread ID to the starting index in the flattened submatrix
        #     t_idx = tx_32 - Int32(1) 
        #     stride = Int32(MAX_THREADS)

        #     # initial r, c
        #     col_offset = div(t_idx, len)
        #     row_offset = rem(t_idx, len)

        #     # precalculate how much R/C change per stride to avoid division inside loop
        #     stride_c = div(stride, len)
        #     stride_r = rem(stride, len)

        #     #register for tile[idx_ck] (because it is repeated sometimes)
        #     last_c = Int32(-1)
        #     current_L_ck = zero(eltype(A))
            
        #     # loop until this thread has finished its share of the submatrix
        #     while t_idx < limit
        #         #actual r, c
        #         c = col_offset + Int32(k + 1)
        #         r = row_offset + Int32(k + 1)
                
        #         # the top multiplier (tile[k, c]) stays the same for a whole column
        #         # if 'c' hasn't changed, reuse the value from the register.
        #         if c != last_c
        #             idx_ck = (k - 1) * STRIDE + c
        #             current_L_ck = @inbounds tile[idx_ck]
        #             last_c = c
        #         end
        #         # indices for the Target (rc) and the Left Multiplier (rk)
        #         idx_rc = (c - 1) * STRIDE + r
        #         idx_rk = (k - 1) * STRIDE + r
                
        #         # perform the update: A[r,c] = A[r,c] - L[r,k] * L[c,k]
        #         # idx_ck = (k - 1) * STRIDE + c
        #         # use muladd instead of * and - for speed
        #         @inbounds tile[idx_rc] = muladd(-tile[idx_rk], current_L_ck, tile[idx_rc])
                
        #         # manual index updates to avoid modulo operations; update by stride
        #         t_idx += stride
        #         col_offset += stride_c
        #         row_offset += stride_r

        #         #wrap around logic
        #         if row_offset >= len
        #             row_offset -= len
        #             col_offset += Int32(1)
        #         end
                
        #     end
        # end

        @synchronize
    end

    # Zero out upper triangle - got rid of this bc unneccesary 
    # istart = (tx - 1) * ops_per_thread + 1
    # iend = min(N, istart + ops_per_thread - 1)

    # for i in istart:iend
    #     for j in (i+1):N
    #         A[i, j] = 0
    #     end
    # end

    # write results back to global memory 
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        
        s_idx = (c - 1) * STRIDE + r
        
        @inbounds A[r, c] = tile[s_idx]
        idx += MAX_THREADS
    end
end

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
        
        kernel = chol_kernel_lower!(backend, MAX_THREADS)
        kernel(A_diag, blk_len; ndrange=MAX_THREADS)
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