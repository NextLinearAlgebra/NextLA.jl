using LinearAlgebra
using KernelAbstractions
using CUDA

# Include the performant_trsm.jl file for the base case 
include("performant_trsm_2 copy.jl")
# include("performant_trsm_2.jl")

function partition_matrices!(A, B, n)
    # mid = div(n, 2)
    
    # A11 = A[1:mid, 1:mid]
    # A22 = A[mid+1:n, mid+1:n]
    # A21 = A[mid+1:n, 1:mid]
    # B1 = B[1:mid, :]
    # B2 = B[mid+1:n, :]

    mid = div(n, 2)
    A11 = view(A, 1:mid, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)
    B1 = view(B, 1:mid, :)
    B2 = view(B, mid+1:n, :)
    A21 = view(A, mid+1:n, 1:mid)  # For lower triangular
    A12 = view(A, 1:mid, mid+1:n)  # For upper triangular
    
    return A11, A22, A21, B1, B2
end

# Recursive function for rectangular triangular solve
function performant_rectrsm!(A::AbstractMatrix{T}, n::Int, B::AbstractMatrix{T}, side::AbstractChar = 'L', k::Int=1;
                  uplo::AbstractChar='L', transpose::AbstractChar='N', threshold::Int=1024) where T
    
    if n <= threshold
        # Base case: choose one of the following - performant trsm is almost sequential, very simple
        # performant_trsm_2_2!(side, uplo, transpose, A, B)
        performant_trsm_2_2!(side, uplo, transpose, A, B)
        return B
    end

    # Check if n is a power of 2 by checking if log2(n) is an integer
    if isinteger(log2(n))
        # Power of 2 case: Split into 4 equal submatrices
        
        A11, A22, A21, B1, B2 = partition_matrices!(A, B, n)

        # Solve the first part of the system recursively
        performant_rectrsm!(A11, div(n, 2), B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Update B2 = B2 - A21 * B1 using GPU-accelerated GEMM (modifying B2 in place)
        CUDA.CUBLAS.gemm!('N', 'N', -1, A21, B1, 1, B2)
        #OR gpu_gemm_update!(A21, B1, B2, mid, n - mid)

        # Solve the second part of the system recursively
        performant_rectrsm!(A22, n - div(n, 2), B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Explicitly set B to the concatenation of B1 and B2
        # B[1:div(n, 2), :] .= B1
        # B[div(n, 2)+1:n, :] .= B2
    
    else
        # Non-power-of-2 case: Standard recursive partition
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        # Partition `A` and `B` into subarrays
        A11 = A[1:M1, 1:M1]
        A22 = A[M1+1:n, M1+1:n]
        A21 = A[M1+1:n, 1:M1]
        B1 = B[1:M1, :]
        B2 = B[M1+1:n, :]

        # Print the matrix size at this level
        println("Processing matrix of size $n x $n")

        # Solve the first part of the system recursively
        performant_rectrsm!(A11, M1, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Update B2 = B2 - A21 * B1 using GPU-accelerated GEMM (modifying B2 in place)
        CUDA.CUBLAS.gemm!('N', 'N', -1, A21, B1, 1, B2)

        # Solve the second part of the system recursively
        performant_rectrsm!(A22, M2, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Explicitly set B to the concatenation of B1 and B2
        B[1:M1, :] .= B1
        B[M1+1:n, :] .= B2
    end
    
    return B
end

# GEMM Update kernel
@kernel function gpu_gemm_kernel!(A, B, C, m, n, k)
    row, col = @index(Global, NTuple)
    if row <= m && col <= n
        sum = zero(eltype(C))
        for i in 1:k
            sum += A[row, i] * B[i, col]
        end
        C[row, col] -= sum  # Modify C (B2) in place here
    end
end

# GEMM Update function
function gpu_gemm_update!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}, m::Int, n::Int) where T
    backend = get_backend(A)
    workgroup_size = (16, 16)
    gpu_gemm_kernel!(backend, workgroup_size)(A, B, C, m, n, size(A, 2), ndrange=(size(C)))
    KernelAbstractions.synchronize(backend)
end


