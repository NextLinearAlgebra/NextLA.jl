using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays
include("performant_trsm_2 copy.jl")

const TILE_DIM = 32


@kernel function coalesced_matmul_kernel!(
    output, @Const(input1), @Const(input2), N, R, M,
    ::Val{BANK} = Val(1),
) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]

    tile1 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)
    tile2 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)

    outval = @private eltype(output) 1
    @inbounds outval[1] = zero(eltype(output))

    @uniform NUM_TILES = ceil(Int, R / TILE_DIM)

    for t in 0:(NUM_TILES - 1)
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        K = t * TILE_DIM + j

        if I <= N && K <= R
            @inbounds tile1[i, j] = input1[I, K]
        else
            @inbounds tile1[i, j] = zero(eltype(output))
        end

        K = t * TILE_DIM + i
        if K <= R && J <= M
            @inbounds tile2[i, j] = input2[K, J]
        else
            @inbounds tile2[i, j] = zero(eltype(output))
        end

        @synchronize

        if I <= N && J <= M
            out = zero(eltype(output))
            @simd for k in 1:TILE_DIM
                @inbounds out += tile1[i, k] * tile2[k, j]
            end
            outval[1] += out
        end

        @synchronize
    end

    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    if I <= N && J <= M
        @inbounds output[I, J] -= outval[1]
    end
end




# Kernel function for solving lower triangular system Ax = b
@kernel function lower_left_kernel(A, B, n)
    col = @index(Group)
    row = @index(Local)

    # Allocate shared memory for diagonal, B column, and A column
    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024
    A_col = @localmem eltype(A) 1024

    # Initialize diagonal and B column
    if row <= n
        @inbounds diag[row] = A[row, row]
        @inbounds B_c[row] = B[row, col] / diag[row]
    end

    # Forward substitution
    for i in 1:n
        @synchronize
        if row > i
            @inbounds A_col[i] = A[i, row] / diag[row]
            @inbounds B_c[row] -= A_col[i] * B_c[i]
        end
    end

    # Write result back to global memory
    if row <= n
        @inbounds B[row, col] = B_c[row]
    end
end

# Kernel function for solving upper triangular system Ax = b
@kernel function upper_left_kernel(A, B, n)
    col = @index(Group)
    row = @index(Local)

    # Allocate shared memory for diagonal, B column, and A column
    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024
    A_col = @localmem eltype(A) 1024

    # Initialize diagonal and B column
    if row <= n
        @inbounds diag[row] = A[row, row]
        @inbounds B_c[row] = B[row, col] / diag[row]
    end

    # Backward substitution
    for i in n:-1:1
        @synchronize
        if row < i
            @inbounds A_col[i] = A[i, row] / diag[row]
            @inbounds B_c[row] -= A_col[i] * B_c[i]
        end
    end

    # Write result back to global memory
    if row <= n
        @inbounds B[row, col] = B_c[row]
    end
end

# Kernel function for solving lower triangular system xA = b
@kernel function right_lower_kernel(A, B, n)
    row = @index(Group)
    col = @index(Local)

    # Allocate shared memory for diagonal, B row, and A row
    diag = @localmem eltype(A) 1024
    B_r = @localmem eltype(B) 1024
    A_row = @localmem eltype(A) 1024

    # Initialize diagonal and B row
    if col <= n
        @inbounds diag[col] = A[col, col]
        @inbounds B_r[col] = B[row, col] / diag[col]
    end

    # Backward substitution
    for i in n:-1:1
        @synchronize
        if col < i
            @inbounds A_row[i] = A[col, i] / diag[col]
            @inbounds B_r[col] -= B_r[i] * A_row[i] 
        end
    end

    # Write result back to global memory
    if col <= n
        @inbounds B[row, col] = B_r[col]
    end
end

# Kernel function for solving upper triangular system xA = b
@kernel function right_upper_kernel(A, B, n)
    row = @index(Group)
    col = @index(Local)

    # Allocate shared memory for diagonal, B row, and A row
    diag = @localmem eltype(A) 1024
    B_r = @localmem eltype(B) 1024
    A_row = @localmem eltype(A) 1024

    # Initialize diagonal and B row
    if col <= n
        @inbounds diag[col] = A[col, col]
        @inbounds B_r[col] = B[row, col] / diag[col]
    end
    
    # Forward substitution
    for i in 1:n
        @synchronize
        if col > i
            @inbounds A_row[i] = A[col, i] / diag[col]
            @inbounds B_r[col] -= B_r[i] * A_row[i]
        end
    end

    # Write result back to global memory
    if col <= n
        @inbounds B[row, col] = B_r[col]
    end
end

# Main recursive triangular solve function
function performant_rectrsm!(A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, side::AbstractChar = 'L';
    uplo::AbstractChar='L', transpose::AbstractChar='N', threshold::Int=256) where T <: AbstractFloat
    
    backend = get_backend(A)

    # Base case: Use kernel functions for small matrices
    if n <= threshold
        n, m = size(B)
    
        if side == 'L' && uplo == 'L' && transpose == 'N'
            lower_left_kernel(backend, (n,))(Transpose(A), B, n, ndrange=(n, m))
        elseif side == 'L' && uplo == 'U' && transpose == 'N'
            upper_left_kernel(backend, (n,))(Transpose(A), B, n, ndrange=(n, m))
        elseif side == 'R' && uplo == 'L' && transpose == 'N'
            right_lower_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
        elseif side == 'R' && uplo == 'U' && transpose == 'N'
            right_upper_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
        else
            error("Unsupported combination of side, uplo, and transposed parameters.")
        end
        return B
    end
    
    # Recursive case: Split the problem into smaller subproblems
    if side == 'L' && uplo == 'L' && transpose == 'N'
        if isinteger(log2(n))
            mid = div(n, 2)
            A11 = view(A, 1:mid, 1:mid)
            A22 = view(A, mid+1:n, mid+1:n)
            A21 = view(A, mid+1:n, 1:mid)
            B1 = view(B, 1:mid, :)
            B2 = view(B, mid+1:n, :)
    
            # Solve the first half
            performant_rectrsm!(A11, mid, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the second half
            N, R, M = size(B2, 1), size(A21, 2), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the second half
            performant_rectrsm!(A22, n - mid, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
        else
            # Handle non-power-of-two sizes
            largest_pow2 = 2 ^ floor(Int, log2(n))
            M1 = largest_pow2
            M2 = n - M1
            
            A11 = view(A, 1:M1, 1:M1)
            A22 = view(A, M1+1:n, M1+1:n)
            A21 = view(A, M1+1:n, 1:M1)
            B1 = view(B, 1:M1, :)
            B2 = view(B, M1+1:n, :)
    
            # Solve the first part
            performant_rectrsm!(A11, M1, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the second part
            N, R, M = size(B2, 1), size(A21, 2), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the second part
            performant_rectrsm!(A22, M2, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
        end
    elseif side == 'L' && uplo == 'U' && transpose == 'N'
        if isinteger(log2(n))
            mid = div(n, 2)
            A11 = view(A, 1:mid, 1:mid)
            A22 = view(A, mid+1:n, mid+1:n) 
            A12 = view(A, 1:mid, mid+1:n)
            B1 = view(B, 1:mid, :)
            B2 = view(B, mid+1:n, :)
    
            # Solve the first half
            performant_rectrsm!(A11, mid, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the second half
            N, R, M = size(B2, 1), size(A12, 2), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B1, A12, B2, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the second half
            performant_rectrsm!(A22, n - mid, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
        else
            # Handle non-power-of-two sizes
            largest_pow2 = 2 ^ floor(Int, log2(n))
            M1 = largest_pow2
            M2 = n - M1
            
            A11 = view(A, 1:M1, 1:M1)
            A22 = view(A, M1+1:n, M1+1:n)
            A12 = view(A, 1:M1, M1+1:n)       
            B1 = view(B, 1:M1, :)
            B2 = view(B, M1+1:n, :)
    
            # Solve the first part
            performant_rectrsm!(A11, M1, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the second part
            N, R, M = size(B2, 1), size(A12, 2), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B1, A12, B2, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the second part
            performant_rectrsm!(A22, M2, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
        end
    elseif side == 'R' && uplo == 'L' && transpose == 'N'
        if isinteger(log2(n))
            mid = div(n, 2)
            A11 = view(A, 1:mid, 1:mid)
            A22 = view(A, mid+1:n, mid+1:n)
            A21 = view(A, mid+1:n, 1:mid)
            B1 = view(B, :, 1:mid)
            B2 = view(B, :, mid+1:n)
    
            # Solve the second half
            performant_rectrsm!(A22, n - mid, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the first half
            N, R, M = size(B2, 1), size(A21, 1), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B1, B2, A21, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the first half
            performant_rectrsm!(A11, mid, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
        else
            # Handle non-power-of-two sizes
            largest_pow2 = 2 ^ floor(Int, log2(n))
            M1 = largest_pow2
            M2 = n - M1
            
            A11 = view(A, 1:M1, 1:M1)
            A22 = view(A, M1+1:n, M1+1:n)
            A21 = view(A, M1+1:n, 1:M1)
            B1 = view(B, :, 1:M1)
            B2 = view(B, :, M1+1:n)
    
            # Solve the second part
            performant_rectrsm!(A22, M2, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the first part
            N, R, M = size(B2, 1), size(A21, 1), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B1, B2, A21, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the first part
            performant_rectrsm!(A11, M1, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
        end
    elseif side == 'R' && uplo == 'U' && transpose == 'N'
        if isinteger(log2(n))
            mid = div(n, 2)
            A11 = view(A, 1:mid, 1:mid)
            A22 = view(A, mid+1:n, mid+1:n)
            A12 = view(A, 1:mid, mid+1:n)
            B1 = view(B, :, 1:mid)
            B2 = view(B, :, mid+1:n)
    
            # Solve the first half
            performant_rectrsm!(A11, mid, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the second half
            N, R, M = size(B2, 1), size(A12, 2), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B2, B1, A12, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the second half
            performant_rectrsm!(A22, n - mid, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
        else
            # Handle non-power-of-two sizes
            largest_pow2 = 2 ^ floor(Int, log2(n))
            M1 = largest_pow2
            M2 = n - M1
            
            A11 = view(A, 1:M1, 1:M1)
            A22 = view(A, M1+1:n, M1+1:n)
            A12 = view(A, 1:M1, M1+1:n)
            B1 = view(B, :, 1:M1)
            B2 = view(B, :, M1+1:n)
    
            # Solve the first part
            performant_rectrsm!(A11, M1, B1, side; uplo=uplo, transpose=transpose, threshold=threshold)
    
            # Update the second part
            N, R, M = size(B2, 1), size(A12, 2), size(B2, 2)
            coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B2, B1, A12, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
    
            # Solve the second part
            performant_rectrsm!(A22, M2, B2, side; uplo=uplo, transpose=transpose, threshold=threshold)
        end
    end
    
    
    return B
end



# Brief Pseudocode
# Assumptions: A is lower triangular, solving Ax = B; A is nxn, B is nxm;
# We are solving for x and mutating B to be x

# Recursion:
# We are performing a recursion with a threshold in which we use the base case.

# If n is less than or equal to the threshold:
#     Call the base case kernel function to solve the small matrix problem.

# Split the matrix A into 4 equal-sized submatrices:
#     A11: Top left (lower triangular)
#     A22: Bottom right (lower triangular)
#     A21: Bottom left (full matrix)
#     A12: Top right (empty)

# Split matrix B into two halves:
#     B1: Top half
#     B2: Bottom half

# Call recursion on the top left submatrix (A11) with size n/2 x n/2

# Perform GEMM: Update bottom half of B: B2 = B2 - (A21 * B1)

# Call recursion on the bottom right submatrix (A22) with size n/2 x n/2

# Base Case of the Triangular Solve (TRSM):

# For each row in B: Get the diagonal element from A for that row, 
# then update the corresponding entry in B by dividing it by the diagonal element.

# Then for each row below the current row, update B by subtracting contributions from rows above it.

# Store the updated value back into B.