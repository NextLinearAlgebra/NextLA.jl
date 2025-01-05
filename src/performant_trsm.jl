using LinearAlgebra
using KernelAbstractions
using CUDA
#bad

@kernel function gpu_trsm_kernel_parallel!(A, B, n, m)
    # Get the column index handled by this thread
    j = @index(Global)

    # Sequential solve for the j-th column - cannot be parallelized as far as I can see
    for i in 1:n
        B[i, j] /= A[i, i]
        for k in i+1:n
            B[k, j] -= A[k, i] * B[i, j]
        end
    end
end

function performant_trsm!(side::Char, uplo::Char, transpose::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    # focusing on the case 'L', 'L', 'N'
    if side == 'L' && uplo == 'L' && transpose == 'N'
        # Get matrix dimensions
        n = size(A, 1)
        m = size(B, 2)

        # Ensure dimensions of A and B are compatible (check)
        @assert n == size(A, 2) "Matrix A must be square"
        @assert n == size(B, 1) "Incompatible dimensions between A and B"

        # Get the backend (GPU)
        backend = get_backend(A)

        # Launch the kernel with one thread per column
        gpu_trsm_kernel_parallel!(backend, m)(A, B, n, m, ndrange=m) #there are m columns and that's the number of threads

        # Synchronize to ensure the GPU completes the computation
        KernelAbstractions.synchronize(backend)

        return B
    else
        error("Only 'L', 'L', 'N' case is supported for now.....")
    end
end
