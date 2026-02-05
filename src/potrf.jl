using KernelAbstractions
using CUDA
using LinearAlgebra

const MAX_THREADS = 512
const MAX_SHARED_SIZE = 2048
const BLOCK_SIZE = 64
#padding for bank conflicts
const PAD = 1
const STRIDE = BLOCK_SIZE + PAD


# @kernel function chol_kernel_lower!(A, N)
#     tx = @index(Global, Linear)

#     # put block into shared memory 
#     tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE)

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

#     for k in 1:N
#         # one thread does sqrt
#         diag_idx = (k - 1) * STRIDE + k
#         if tx == 1
#             @inbounds tile[diag_idx] = sqrt(tile[diag_idx])
#         end

#         @synchronize

#         # division is now parallelized 
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
        
#         len = Int32(N - k)
#         tx_32 = Int32(tx)
#         if len > 0
#             limit = len * len
#             t_idx = tx_32 - Int32(1) 
#             stride = Int32(MAX_THREADS)

#             # precalculate offsets to avoid division inside loop
#             col_offset = div(t_idx, len)
#             row_offset = rem(t_idx, len)
#             stride_c = div(stride, len)
#             stride_r = rem(stride, len)
            
#             while t_idx < limit
#                 if row_offset >= col_offset
#                     r = row_offset + Int32(k + 1)
#                     c = col_offset + Int32(k + 1)
#                     idx_rc = (c - 1) * STRIDE + r
#                     idx_rk = (k - 1) * STRIDE + r
#                     idx_ck = (k - 1) * STRIDE + c
#                     # use muladd instead of * and - for speed
#                     @inbounds tile[idx_rc] = muladd(-tile[idx_rk], tile[idx_ck], tile[idx_rc])
#                 end
                
#                 # manual index updates to avoid modulo operations
#                 t_idx += stride
#                 col_offset += stride_c
#                 row_offset += stride_r

#                 if row_offset >= len
#                     row_offset -= len
#                     col_offset += Int32(1)
#                 end
#             end
#         end

#         @synchronize
#     end

#     # Zero out upper triangle
#     # istart = (tx - 1) * ops_per_thread + 1
#     # iend = min(N, istart + ops_per_thread - 1)

#     # for i in istart:iend
#     #     for j in (i+1):N
#     #         A[i, j] = 0
#     #     end
#     # end

#     # write results back to global memory 
#     idx = tx
#     while idx <= total_elements
#         c = div(idx - 1, N) + 1
#         r = rem(idx - 1, N) + 1
        
#         s_idx = (c - 1) * STRIDE + r
        
#         @inbounds A[r, c] = tile[s_idx]
#         idx += MAX_THREADS
#     end
# end


# function cholesky_lower!(A)
#     N = size(A, 1)
#     backend = CUDABackend()
    
#     #blocked algorithm - for sized bigger than 64x64 we do the trsm/gemm but not recursive.
#     for k in 1:BLOCK_SIZE:N
#         k_end = min(k + BLOCK_SIZE - 1, N)
#         blk_len = k_end - k + 1
#         A_diag = view(A, k:k_end, k:k_end)
#         kernel = chol_kernel_lower!(backend, MAX_THREADS)
#         kernel(A_diag, blk_len; ndrange=MAX_THREADS)
        
        
#         if k_end < N
#             A_panel = view(A, (k_end + 1):N, k:k_end)

#             # RightUpperTRSM!(Transpose(A_diag), A_panel)
#             unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A_diag, A_panel)
#             # CUBLAS.trsm!('R', 'L', 'T', 'N', one(eltype(A)), A_diag, A_panel)
            
#             A_trailing = view(A, (k_end + 1):N, (k_end + 1):N)
            
#             CUBLAS.gemm!('N', 'T', -one(eltype(A)), A_panel, A_panel, one(eltype(A)), A_trailing)
#             # CUBLAS.syrk!('L', 'N', -1.0, A_panel, 1.0, A_trailing)
#         end
#     end

#     KernelAbstractions.synchronize(backend)
#     return A
# end


#right looking cholesky kernel
@kernel cpu=false inbounds=true unsafe_indices=false function chol_kernel_lower!(A, ::Val{N}) where N
    tx = @index(Global, Linear) #change global to local? 

    # put block into shared memory 
    tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE) #stride prevents bank conflicts [64][65]

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
        
        len = Int32(N - k) #size of submatrix
        tx_32 = Int32(tx)
        if len > 0
            limit = len * len #total items to process

            # map thread ID to the starting index in the flattened submatrix
            t_idx = tx_32 - Int32(1) 
            stride = Int32(MAX_THREADS)

            # initial r, c
            col_offset = div(t_idx, len)
            row_offset = rem(t_idx, len)

            # precalculate how much R/C change per stride to avoid division inside loop
            stride_c = div(stride, len)
            stride_r = rem(stride, len)

            #register for tile[idx_ck] (because it is repeated sometimes)
            last_c = Int32(-1)
            current_L_ck = zero(eltype(A))
            
            # loop until this thread has finished its share of the submatrix
            while t_idx < limit
                
                #actual r, c
                c = col_offset + Int32(k + 1)
                r = row_offset + Int32(k + 1)
                
                # the top multiplier (tile[k, c]) stays the same for a whole column
                # if 'c' hasn't changed, reuse the value from the register.
                if c != last_c
                    idx_ck = (k - 1) * STRIDE + c
                    current_L_ck = @inbounds tile[idx_ck]
                    last_c = c
                end
                # indices for the Target (rc) and the Left Multiplier (rk)
                idx_rc = (c - 1) * STRIDE + r
                idx_rk = (k - 1) * STRIDE + r
                
                # perform the update: A[r,c] = A[r,c] - L[r,k] * L[c,k]
                # idx_ck = (k - 1) * STRIDE + c
                # use muladd instead of * and - for speed
                # @inbounds tile[idx_rc] = muladd(-tile[idx_rk], current_L_ck, tile[idx_rc])
                @inbounds tile[idx_rc] -= tile[idx_rk] * current_L_ck

                # manual index updates to avoid modulo operations; update by stride
                t_idx += stride
                col_offset += stride_c
                row_offset += stride_r

                #wrap around logic
                if row_offset >= len
                    row_offset -= len
                    col_offset += Int32(1)
                end
                
            end
        end

        @synchronize
    end

    # write results back to global memory - have a warp dedicated to write back (as write warps)??? but cant do it after sync?
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        
        s_idx = (c - 1) * STRIDE + r
        
        # @inbounds A[r, c] = tile[s_idx]

        if r >= c
            @inbounds A[r, c] = tile[s_idx]
        end
        idx += MAX_THREADS
    end
end

function cholesky_lower!(A)
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
        
        kernel = chol_kernel_lower!(backend, MAX_THREADS)
        # crucial: we must force workgroupsize to be 768 so all threads are in the same block
        kernel(A_diag, Val(blk_len); ndrange=MAX_THREADS)
        
        # update the panel to the right if we aren't at the end yet
        if k_end < N
            A_off_diag = view(A, (k_end + 1):N, k:k_end)
            unified_rectrxm!('R', 'L', 'T', 1.0, 'S', A_diag, A_off_diag)
        end
    end

    KernelAbstractions.synchronize(backend)
    return A
end

# function test_cholesky_lower(N)
#     println("Testing lower Cholesky for N = $N")
#     A = rand(Float16, N, N)
#     A = A * A' + N * I

#     A_gpu = CuArray(A)
#     t1 = @elapsed cholesky_lower!(A_gpu)

#     t2 = @elapsed L_ref = cholesky(A).L
#     L_gpu = Array(A_gpu)

#     rel_err = norm(L_gpu - L_ref) / norm(L_ref)
#     println("Relative error: $rel_err")
#     println((t1, t2))
# end

# test_cholesky_lower(256)