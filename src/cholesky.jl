using KernelAbstractions
using CUDA
using LinearAlgebra

const MAX_THREADS = 512

@kernel function chol_kernel_lower!(A, N, ops_per_thread)
    tx = @index(Global, Linear)

    for k in 1:N
        # Thread 0 does sqrt and division
        if tx == 1
            A[k, k] = sqrt(A[k, k])
            for j in (k+1):N
                A[j, k] /= A[k, k]
            end
        end

        @synchronize

        # Elimination step
        istart = (k + 1) + (tx - 1) * ops_per_thread
        iend = min(N, istart + ops_per_thread - 1)

        for i in istart:iend
            for j in i:N
                A[j, i] -= A[j, k] * A[i, k]
            end
        end

        @synchronize
    end

    # Zero out upper triangle
    istart = (tx - 1) * ops_per_thread + 1
    iend = min(N, istart + ops_per_thread - 1)

    for i in istart:iend
        for j in (i+1):N
            A[i, j] = 0
        end
    end
end


function cholesky_lower!(A)
    N = size(A, 1)
    num_threads = min(N, MAX_THREADS)
    ops_per_thread = cld(N, num_threads)

    backend = CUDABackend()
    kernel = chol_kernel_lower!(backend, num_threads)

    kernel(A, N, ops_per_thread; ndrange = num_threads)
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