using KernelAbstractions
using CUDA
using LinearAlgebra


const MAX_THREADS = 512 
const BLOCK_SIZE = 64
const PAD = 1
const STRIDE = BLOCK_SIZE + PAD

# left looking cholesky kernel
@kernel function chol_kernel_left!(A, N)
    tx = @index(Global, Linear)
    
    # shared memory alloc
    tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE)

    # load into shmem
    total_elements = N * N
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        s_idx = (c - 1) * STRIDE + r
        
        @inbounds tile[s_idx] = A[r, c]
        idx += MAX_THREADS
    end

    @synchronize

    # main factorization
    for k in 1:N
        
        # sqrt diag
        if tx == 1
            idx_kk = (k - 1) * STRIDE + k
            @inbounds tile[idx_kk] = sqrt(tile[idx_kk])
        end
        @synchronize

        # take col k
        diag_val = tile[(k - 1) * STRIDE + k]
        
        # scale rows i > k in column k
        i = tx
        while i <= N
            if i > k
                idx_ik = (k - 1) * STRIDE + i
                @inbounds tile[idx_ik] /= diag_val
            end
            i += MAX_THREADS
        end
        @synchronize

        # update trailing submatrix A[i, j] = A[i, j] - L[i, k] * L[j, k]' for all i,j > k
        
        # assign threads to handle rows 'i' in the trailing submatrix
        i = tx
        while i <= N
            if i > k
                
                # load L[i, k] (the factor from the pivot column) into a register
                idx_ik = (k - 1) * STRIDE + i
                @private L_ik = @inbounds tile[idx_ik] 

                for j in (k + 1):i
                    idx_jk = (k - 1) * STRIDE + j  
                    idx_ij = (j - 1) * STRIDE + i  

                    @inbounds tile[idx_ij] = muladd(-L_ik, tile[idx_jk], tile[idx_ij])
                end
            end
            i += MAX_THREADS
        end
        @synchronize
    end

    # write back
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        s_idx = (c - 1) * STRIDE + r
        @inbounds A[r, c] = tile[s_idx]
        idx += MAX_THREADS
    end
end


#right looking cholesky kernel
@kernel function chol_kernel_lower!(A, N)
    tx = @index(Global, Linear)

    # put block into shared memory 
    tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE)

    #register for multiplier & diag
    multiplier = @private eltype(A) (1,)
    # diag_val = @private eltype(A) (1,)

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

    for k in 1:N
        # one thread does sqrt
        diag_idx = (k - 1) * STRIDE + k
        if tx == 1
            @inbounds tile[diag_idx] = sqrt(tile[diag_idx])
        end

        @synchronize

        # division is now parallelized 
        diag_val = @inbounds tile[diag_idx] #register holds the diagonal val
        idx = k + tx 
        while idx <= N
            s_idx = (k - 1) * STRIDE + idx
            @inbounds tile[s_idx] /= diag_val
            idx += MAX_THREADS
        end

        @synchronize

        # Elimination step
        # updates submatrix to right/bottom

        c = (k + 1) + (tx - 1) # Parallelize over columns
        
        while c <= N
            # load in register
            ck_idx = (k - 1) * STRIDE + c
            @inbounds multiplier[1] = tile[ck_idx]

            # update rows in col
            for r in c:N
                rc_idx = (c - 1) * STRIDE + r 
                rk_idx = (k - 1) * STRIDE + r 
                
                #use maladd for speed
                @inbounds tile[rc_idx] = muladd(-tile[rk_idx], multiplier[1], tile[rc_idx])
            end
            
            c += MAX_THREADS
        end
        
        # len = Int32(N - k)
        # tx_32 = Int32(tx)
        # if len > 0
        #     limit = len * len
        #     t_idx = tx_32 - Int32(1) 
        #     stride = Int32(MAX_THREADS)

        #     # precalculate offsets to avoid division inside loop
        #     col_offset = div(t_idx, len)
        #     row_offset = rem(t_idx, len)
        #     stride_c = div(stride, len)
        #     stride_r = rem(stride, len)
            
        #     while t_idx < limit
        #         if row_offset >= col_offset
        #             r = row_offset + Int32(k + 1)
        #             c = col_offset + Int32(k + 1)
        #             idx_rc = (c - 1) * STRIDE + r
        #             idx_rk = (k - 1) * STRIDE + r
        #             idx_ck = (k - 1) * STRIDE + c
        #             # use muladd instead of * and - for speed
        #             @inbounds tile[idx_rc] = muladd(-tile[idx_rk], tile[idx_ck], tile[idx_rc])
        #         end
                
        #         # manual index updates to avoid modulo operations
        #         t_idx += stride
        #         col_offset += stride_c
        #         row_offset += stride_r

        #         if row_offset >= len
        #             row_offset -= len
        #             col_offset += Int32(1)
        #         end
        #     end
        # end

        @synchronize
    end

    # Zero out upper triangle
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