using KernelAbstractions
using CUDA
using LinearAlgebra

const MAX_THREADS = 512
const MAX_SHARED_SIZE = 2048
const BLOCK_SIZE = 64


@kernel function chol_kernel_lower!(A, N)
    tx = @index(Global, Linear)

    # curr_col = @localmem eltype(A) MAX_SHARED_SIZE
    tile = @localmem eltype(A) (BLOCK_SIZE + 1, BLOCK_SIZE)

    total_elements = N * N
    idx = tx
    while idx <= total_elements
        # Map linear index to (row, col)
        # Julia is Column-Major: row changes fastest
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        
        @inbounds tile[r, c] = A[r, c]
        idx += MAX_THREADS
    end

    @synchronize

    for k in 1:N
        # Thread 0 does sqrt and division
        if tx == 1
            @inbounds tile[k, k] = sqrt(tile[k, k])
        end

        @synchronize

        diag = @inbounds tile[k, k]
        idx = k + tx 
        while idx <= N
            @inbounds tile[idx, k] /= diag
            idx += MAX_THREADS
        end

        @synchronize

        # Elimination step
        
        len = Int32(N - k)
        tx_32 = Int32(tx)
        if len > 0
            limit = len * len
            t_idx = tx_32 - Int32(1) 
            stride = Int32(MAX_THREADS)
            
            while t_idx < limit
                col_offset = div(t_idx, len)
                row_offset = rem(t_idx, len)

                if row_offset >= col_offset
                    r = row_offset + Int32(k + 1)
                    c = col_offset + Int32(k + 1)
                    # @inbounds tile[r, c] -= tile[r, k] * tile[c, k]
                    @inbounds tile[r, c] = muladd(-tile[r, k], tile[c, k], tile[r, c])
                end
                
                t_idx += stride
            end
        end

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
    idx = tx
    while idx <= total_elements
        c = div(idx - 1, N) + 1
        r = rem(idx - 1, N) + 1
        
        @inbounds A[r, c] = tile[r, c]
        idx += MAX_THREADS
    end
end


function cholesky_lower!(A)
    N = size(A, 1)
    backend = CUDABackend()
    
    for k in 1:BLOCK_SIZE:N
        k_end = min(k + BLOCK_SIZE - 1, N)
        blk_len = k_end - k + 1
        A_diag = view(A, k:k_end, k:k_end)
        kernel = chol_kernel_lower!(backend, MAX_THREADS)
        kernel(A_diag, blk_len; ndrange=MAX_THREADS)
        KernelAbstractions.synchronize(backend)
        
        if k_end < N
            A_panel = view(A, (k_end + 1):N, k:k_end)

            RightUpperTRSM!(Transpose(A_diag), A_panel)
            # CUBLAS.trsm!('R', 'L', 'T', 'N', one(eltype(A)), A_diag, A_panel)
            
            A_trailing = view(A, (k_end + 1):N, (k_end + 1):N)
            
            CUBLAS.gemm!('N', 'T', -one(eltype(A)), A_panel, A_panel, one(eltype(A)), A_trailing)
            # CUBLAS.syrk!('L', 'N', -1.0, A_panel, 1.0, A_trailing)
        end
    end
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