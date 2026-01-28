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
    
    # shared memory w padding
    tile = @localmem eltype(A) (BLOCK_SIZE * STRIDE)

    # load tile into shmem
    total_elements = N * N
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        s_idx = (c - 1) * STRIDE + r
        
        @inbounds tile[s_idx] = A[r, c]
        idx += MAX_THREADS
    end

    # wait for finish for load
    @synchronize

    # factorization loop
    for j in 1:N
        
        # each thread handles one ROW 'i' for the current column 'j'.
        i = tx
        while i <= N
            # we only process the lower triangle (i >= j)
            if i >= j
                
                # load curr val into register (val)
                idx_ij = (j - 1) * STRIDE + i
                val = @inbounds tile[idx_ij]
                
                # accumulate updates from other columns
                for k in 1:(j-1)
                    idx_ik = (k - 1) * STRIDE + i # L[i,k]
                    idx_jk = (k - 1) * STRIDE + j # L[j,k] (transpose of L[k,j])
                    
                    # muladd happens in reg
                    val = muladd(-tile[idx_ik], tile[idx_jk], val)
                end
                
                # write result back to Shared Memory ONCE.
                @inbounds tile[idx_ij] = val
            end
            
            i += MAX_THREADS
        end
        
        @synchronize 

        #sqrt (diagonal factorization)
        if tx == 1
            idx_jj = (j - 1) * STRIDE + j
            @inbounds tile[idx_jj] = sqrt(tile[idx_jj])
        end
        
        @synchronize

        # scaling
        diag_val = @inbounds tile[(j - 1) * STRIDE + j]
        
        i = tx
        while i <= N
            if i > j
                idx_ij = (j - 1) * STRIDE + i
                # division happens here
                @inbounds tile[idx_ij] /= diag_val
            end
            i += MAX_THREADS
        end
        
        @synchronize
    end

    # write back to globalmem
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
        
        kernel = chol_kernel_left!(backend, MAX_THREADS)
        kernel(A_diag, blk_len; ndrange=MAX_THREADS)
        KernelAbstractions.synchronize(backend)
        
        if k_end < N
            A_off_diag = view(A, (k_end + 1):N, k:k_end)
            
            CUBLAS.trsm!('R', 'L', 'T', 'N', one(eltype(A)), A_diag, A_off_diag)
        end
    end

    KernelAbstractions.synchronize(backend)
    return A
end