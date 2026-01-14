using KernelAbstractions
using CUDA
using LinearAlgebra

const MAX_THREADS = 512
const MAX_SHARED_SIZE = 2048

@kernel function chol_kernel_lower!(A, N)
    tx = @index(Global, Linear)

    curr_col = @localmem eltype(A) MAX_SHARED_SIZE

    for k in 1:N
        # Thread 0 does sqrt and division
        if tx == 1
            @inbounds A[k, k] = sqrt(A[k, k])
        end

        @synchronize

        diag = @inbounds A[k, k]
        idx = k + tx 
        while idx <= N
            @inbounds A[idx, k] /= diag
            idx += MAX_THREADS
        end

        @synchronize

        idx = k + tx
        while idx <= N
            @inbounds curr_col[idx] = A[idx, k]
            idx += MAX_THREADS
        end
        
        if tx == 1
            @inbounds curr_col[k] = diag
        end
        @synchronize

        # Elimination step
        # for col in (k + 1):N
        #     mul = curr_col[col] 
        #     row = col + (tx - 1)
        #     while row <= N
        #         A[row, col] -= curr_col[row] * mul
        #         row += MAX_THREADS
        #     end
            
        # end
        len = N - k
        if len > 0
            limit = len * len
            
            t_idx = tx - 1 
            
            while t_idx < limit
                col_offset = div(t_idx, len)
                row_offset = rem(t_idx, len)
                
                if row_offset >= col_offset
                    r = row_offset + k + 1
                    c = col_offset + k + 1
                    @inbounds A[r, c] -= curr_col[r] * curr_col[c]
                end
                
                t_idx += MAX_THREADS
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
end


function cholesky_lower!(A)
    N = size(A, 1)
    num_threads = MAX_THREADS
    # ops_per_thread = cld(N, num_threads)

    backend = CUDABackend()
    kernel = chol_kernel_lower!(backend, num_threads)

    kernel(A, N; ndrange = num_threads)
    KernelAbstractions.synchronize(backend)
    return A
end

function test_cholesky_lower(N)
    println("Testing lower Cholesky for N = $N")
    A = rand(Float16, N, N)
    A = A * A' + N * I

    A_gpu = CuArray(A)
    t1 = @elapsed cholesky_lower!(A_gpu)

    t2 = @elapsed L_ref = cholesky(A).L
    L_gpu = Array(A_gpu)

    rel_err = norm(L_gpu - L_ref) / norm(L_ref)
    println("Relative error: $rel_err")
    println((t1, t2))
end

# test_cholesky_lower(256)