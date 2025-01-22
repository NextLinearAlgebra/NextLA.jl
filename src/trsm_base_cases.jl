using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays


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