"""
GPU-accelerated Triangular Matrix Multiplication (TRMM) Operations

This module provides GPU kernel implementations for triangular matrix multiplication
operations, supporting both left and right sided operations with upper and lower
triangular matrices.

The kernels are optimized for GPU architectures with:
- Shared memory tiling for improved memory access patterns  
- Bank conflict avoidance through memory padding
- Vectorized inner loops for computational efficiency
- Bounds checking for non-square matrix operations

All kernels perform in-place operations: B := A * B or B := B * A
where A is triangular and B is a general matrix.
"""

export LeftLowerTRMM!, LeftUpperTRMM!, RightLowerTRMM!, RightUpperTRMM!

# Performs in-place TRMM: B := A * B 
# where A is an N×N lower triangular matrix and B is an N×M matrix
# A is limited to matrix size 16×16 due to shared memory constraints

"""
    LeftLowerTRMM_kernel!(A, B, ::Val{BANK}=Val(1))

GPU kernel for left-sided lower triangular matrix multiplication.

Performs the operation B := A * B where A is lower triangular.
Uses shared memory tiling with configurable bank offset to avoid conflicts.

# Arguments
- `A::AbstractMatrix`: N×N lower triangular coefficient matrix
- `B::AbstractMatrix`: N×M target matrix (modified in-place) 
- `BANK::Int`: Memory bank offset to avoid conflicts (default: 1)

# Implementation Notes
- Tile size limited to 16×16 due to shared memory constraints
- Uses private variables for accumulation to enable vectorization
- Includes bounds checking for non-square input matrices
- Synchronization points ensure correct shared memory access patterns
"""
@kernel function LeftLowerTRMM_kernel!(A,B,
                            ::Val{BANK} = Val(1)) where BANK
    
    # Get thread and block indices
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # Tile dimension kept at 16×16 due to shared memory constraints
    TILE_DIM = @uniform @groupsize()[1]

    # Allocate shared memory for sub-matrix product calculation
    # BANK padding added to avoid bank conflicts from irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)  # For matrix A
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)  # For matrix B

    # Private variable to accumulate the result of sub-matrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = zero(eltype(B))

    # Get matrix dimensions
    @uniform N = size(A, 1)  # Matrix A dimensions
    @uniform R = size(A, 2)  # Matrix A dimensions  
    @uniform M = size(B, 2)  # Matrix B column count

    # Calculate global thread indices (cannot use @index(Global) with custom ndrange)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # Load input matrix A into shared memory tile with bounds checking
    if i <= N && j <= N
        @inbounds tile1[i, j] = A[i, j]
    else
        @inbounds tile1[i, j] = zero(eltype(A))
    end

    # Load input/output matrix B into shared memory tile with bounds checking
    if I <= R && J <= M
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = zero(eltype(B))
    end

    # Synchronize to ensure all tiles are loaded before computation
    @synchronize

    # Calculate triangular matrix-vector product for lower triangular A
    # For lower triangular: only use elements A[i,k] where k <= i
    out = zero(eltype(B))
    @simd for k in 1:i
        @inbounds out += tile1[i, k] * tile2[k, j]
    end
    B_sub[1] += out

    # Synchronize before writing results
    @synchronize
    
    # Recalculate global indices after synchronization
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # Write result back to global memory if within bounds
    if I <= N && J <= M
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize

end





# A is an NxN upper triangular matrix and B is an NxM matrix
@kernel function LeftUpperTRMM_kernel!(A,B,
                            ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if I <= N && j <= R
        @inbounds tile1[i, j] = A[I, j]
    else
        @inbounds tile1[i, j] = 0.0
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= R && J <= M
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize

    # get global values again (because of synchronize?)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in i:N
        @inbounds out += tile1[i, k] * tile2[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize
end








@kernel function RightLowerTRMM_kernel!(A,B,
                ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform M = size(B, 1)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if i <= N && j <= N
        @inbounds tile1[i, j] = A[i, j]
    else
        @inbounds tile1[i, j] = 0.0
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= M && J <= N
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize


    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in j:N
        @inbounds out += tile2[i, k] * tile1[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= M && J <= N
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize

end

@kernel function RightUpperTRMM_kernel!(A,B,
    ::Val{BANK} = Val(1)) where BANK
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    # kept at 16x16 due to shmem constraints
    TILE_DIM = @uniform @groupsize()[1]

    # allocating shared memory for the sub matrix product calculation
    # BANK = 1, added to avoid banck coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    B_sub = @private eltype(B) 1
    @inbounds B_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform M = size(B, 1)

    # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # load input A into tile, with bounds checking for non-square matrices
    if i <= N && j <= N
        @inbounds tile1[i, j] = A[i, j]
    else
        @inbounds tile1[i, j] = 0.0
    end

    # load input/output B into tiles, with bounds checking for non-square matrices
    if I <= M && J <= N
        @inbounds tile2[i, j] = B[I, J]
    else
        @inbounds tile2[i, j] = 0.0
    end

    # wait for all tiles to be loaded
    @synchronize


    # calculate value of spot in output, use temporary value to allow for vectorization
    out = zero(eltype(B))
    @simd for k in 1:j
        @inbounds out += tile2[i, k] * tile1[k, j]
    end
    B_sub[1] += out

    @synchronize
    
    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= M && J <= N
        @inbounds B[I, J] = B_sub[1]
    end
    @synchronize
end




"""
    LeftLowerTRMM!(A, B; n_threads=(16,16))

Perform left-sided lower triangular matrix multiplication: B := A * B

# Arguments
- `A::AbstractMatrix`: N×N lower triangular coefficient matrix
- `B::AbstractMatrix`: N×M target matrix (modified in-place)
- `n_threads::Tuple`: Thread block dimensions (default: (16,16))

# Implementation Notes
- Uses GPU acceleration with optimized kernel
- Thread block size should not exceed hardware limits
- NDRange is padded to handle boundary conditions
"""
function LeftLowerTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    # Calculate NDRange with padding to handle boundary threads
    Ndrange = max(size(A), size(B))
    Ndrange = (Ndrange[1] + 16, Ndrange[2] + 16)
    LeftLowerTRMM_kernel!(backend, n_threads)(A, B, ndrange = Ndrange)
end

"""
    LeftUpperTRMM!(A, B; n_threads=(16,16))

Perform left-sided upper triangular matrix multiplication: B := A * B

# Arguments
- `A::AbstractMatrix`: N×N upper triangular coefficient matrix  
- `B::AbstractMatrix`: N×M target matrix (modified in-place)
- `n_threads::Tuple`: Thread block dimensions (default: (16,16))
"""
function LeftUpperTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    Ndrange = max(size(A), size(B))
    Ndrange = (Ndrange[1] + 16, Ndrange[2] + 16)
    LeftUpperTRMM_kernel!(backend, n_threads)(A, B, ndrange = Ndrange)
end

"""
    RightLowerTRMM!(A, B; n_threads=(16,16))

Perform right-sided lower triangular matrix multiplication: B := B * A

# Arguments
- `A::AbstractMatrix`: N×N lower triangular coefficient matrix
- `B::AbstractMatrix`: M×N target matrix (modified in-place)  
- `n_threads::Tuple`: Thread block dimensions (default: (16,16))
"""
function RightLowerTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    Ndrange = max(size(A), size(B))
    Ndrange = (Ndrange[1] + 16, Ndrange[2] + 16)
    RightLowerTRMM_kernel!(backend, n_threads)(A, B, ndrange = Ndrange)
end

"""
    RightUpperTRMM!(A, B; n_threads=(16,16))

Perform right-sided upper triangular matrix multiplication: B := B * A

# Arguments  
- `A::AbstractMatrix`: N×N upper triangular coefficient matrix
- `B::AbstractMatrix`: M×N target matrix (modified in-place)
- `n_threads::Tuple`: Thread block dimensions (default: (16,16))
"""
function RightUpperTRMM!(A, B; n_threads = (16,16))
    backend = get_backend(A)
    Ndrange = max(size(A), size(B))
    Ndrange = (Ndrange[1] + 16, Ndrange[2] + 16)
    RightUpperTRMM_kernel!(backend, n_threads)(A, B, ndrange = Ndrange)
end

"""
    trmm(side, uplo, transa, diag, A, B, alpha=1.0) -> B

Performs triangular matrix multiplication with automatic parameter detection.
This is a simplified interface for triangular matrix multiplication operations.

# Arguments
- `side::Char`: Operation side
    - 'L': B := alpha*op(A)*B (left multiplication)
    - 'R': B := alpha*B*op(A) (right multiplication)
- `uplo::Char`: Triangular part specification
    - 'U': A is upper triangular
    - 'L': A is lower triangular  
- `transa::Char`: Operation on matrix A
    - 'N': op(A) = A (no transpose)
    - 'T': op(A) = A^T (transpose)
    - 'C': op(A) = A^H (conjugate transpose)
- `diag::Char`: Diagonal type (currently unused in GPU implementation)
    - 'N': non-unit diagonal
    - 'U': unit diagonal
- `A::AbstractMatrix`: Triangular coefficient matrix
- `B::AbstractMatrix`: Target matrix (modified in-place)
- `alpha`: Scalar multiplier (default: 1.0)

# Returns
- Updated matrix B (same as input B, modified in-place)

# Example
```julia
A = complex.(triu(randn(4, 4)), triu(randn(4, 4)))
B = complex.(randn(4, 3), randn(4, 3))
C = trmm('L', 'U', 'N', 'N', A, copy(B))
```

# Implementation Notes
- Currently supports 'N' (no transpose) operations only
- Uses GPU-accelerated kernels for computation
- The transa and diag parameters are provided for interface compatibility
"""
function trmm(side, uplo, transa, diag, A, B, alpha=one(eltype(A)))
    # Apply scaling if alpha != 1
    if alpha != one(eltype(A))
        B .*= alpha
    end
    
    # Dispatch to appropriate GPU kernel based on operation parameters
    if side == 'L' && uplo == 'L'
        LeftLowerTRMM!(A, B)
    elseif side == 'L' && uplo == 'U' 
        LeftUpperTRMM!(A, B)
    elseif side == 'R' && uplo == 'L'
        RightLowerTRMM!(A, B)
    elseif side == 'R' && uplo == 'U'
        RightUpperTRMM!(A, B)
    else
        error("Unsupported combination of side='$side', uplo='$uplo'")
    end
end