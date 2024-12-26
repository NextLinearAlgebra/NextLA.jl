using LinearAlgebra
using KernelAbstractions
using CUDA

# Include the performant_trsm.jl file for the base case 
include("performant_trsm.jl")

# Recursive function for rectangular triangular solve
function performant_rectrsm!(A::CuArray{T}, n::Int, B::CuArray{T}, side::AbstractChar = 'L', k::Int=1;
                  uplo::AbstractChar='L', transpose::AbstractChar='N', threshold::Int=16) where T
    
    if n <= threshold
        # Base case
        performant_trsm!(side, uplo, transpose, A, B)
        return B
    end

    # Check if n is a power of 2 by checking if log2(n) is an integer
    if isinteger(log2(n))
        # Power of 2 case: Split into 4 equal submatrices
        mid = div(n, 2)
        
        A11 = A[1:mid, 1:mid]
        A22 = A[mid+1:n, mid+1:n]
        A21 = A[mid+1:n, 1:mid]
        A12 = A[1:mid, mid+1:n]

        B1 = B[1:mid, :]
        B2 = B[mid+1:n, :]
        
        # Solve the first part of the system recursively
        performant_rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Update B2 = B2 - A21 * B1 using GPU-accelerated GEMM (modifying B2 in place)
        gpu_gemm_update!(A21, B1, B2, mid, n - mid)

        # Solve the second part of the system recursively
        performant_rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Explicitly set B to the concatenation of B1 and B2
        B[1:mid, :] .= B1
        B[mid+1:n, :] .= B2
    
    else
        # Non-power-of-2 case: Standard recursive partition
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        # Partition `A` and `B` into subarrays
        A11 = A[1:M1, 1:M1]
        A22 = A[M1+1:n, M1+1:n]
        A21 = A[M1+1:n, 1:M1]
        A12 = A[1:M1, M1+1:n]

        B1 = B[1:M1, :]
        B2 = B[M1+1:n, :]
        
        # Solve the first part of the system recursively
        performant_rectrsm!(A11, M1, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        # Update B2 = B2 - A21 * B1 using GPU-accelerated GEMM (modifying B2 in place)
        gpu_gemm_update!(A21, B1, B2, M2, size(B2, 2))

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
