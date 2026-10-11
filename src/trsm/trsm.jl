
export LeftLowerTRSM!, LeftUpperTRSM!, RightLowerTRSM!, RightUpperTRSM!

# Kernel function for solving lower triangular system Ax = b
@kernel function lower_left_kernel(A, B, n, ::Val{U}) where {U}
    col = @index(Group)
    row = @index(Local)

    # Allocate shared memory for diagonal, B column, and A column
    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024
    A_col = @localmem eltype(A) 1024

    # Initialize diagonal and B column
    if row <= n
        @inbounds diag[row] = U ? one(eltype(A)) : A[row, row]
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
@kernel function upper_left_kernel(A, B, n, ::Val{U}) where {U}
    col = @index(Group)
    row = @index(Local)

    # Allocate shared memory for diagonal, B column, and A column
    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024
    A_col = @localmem eltype(A) 1024

    # Initialize diagonal and B column
    if row <= n
        @inbounds diag[row] = U ? one(eltype(A)) : A[row, row]
        @inbounds B_c[row] = B[row, col] / diag[row]
    end

    # Backward substitution
    for i in n:-1:1
        @synchronize
        if row < i
            @inbounds A_col[i] = A[row, i] / diag[row]
            @inbounds B_c[row] -= A_col[i] * B_c[i]
        end
    end

    # Write result back to global memory
    if row <= n
        @inbounds B[row, col] = B_c[row]
    end
end

# Kernel function for solving lower triangular system xA = b
@kernel function right_lower_kernel(A, B, n, ::Val{U}) where {U}
    row = @index(Group)
    col = @index(Local)

    # Allocate shared memory for diagonal, B row, and A row
    diag = @localmem eltype(A) 1024
    B_r = @localmem eltype(B) 1024
    A_row = @localmem eltype(A) 1024

    # Initialize diagonal and B row
    if col <= n
        @inbounds diag[col] = U ? one(eltype(A)) : A[col, col]
        @inbounds B_r[col] = B[row, col] / diag[col]
    end

    # Backward substitution
    for i in n:-1:1
        @synchronize
        if col < i
            @inbounds A_row[i] = A[i, col] / diag[col]
            @inbounds B_r[col] -= B_r[i] * A_row[i] 
        end
    end

    # Write result back to global memory
    if col <= n
        @inbounds B[row, col] = B_r[col]
    end
end

# Kernel function for solving upper triangular system xA = b
@kernel function right_upper_kernel(A, B, n, ::Val{U}) where {U}
    row = @index(Group)
    col = @index(Local)

    # Allocate shared memory for diagonal, B row, and A row
    diag = @localmem eltype(A) 1024
    B_r = @localmem eltype(B) 1024
    A_row = @localmem eltype(A) 1024

    # Initialize diagonal and B row
    if col <= n
        @inbounds diag[col] = U ? one(eltype(A)) : A[col, col]
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

function LeftLowerTRSM!(A, B, unitdiag::Bool=false)
    n, m = size(B)
    backend = get_backend(A)
    lower_left_kernel(backend, (n,))(Transpose(A), B, n, Val(unitdiag), ndrange=(n, m))
end

function LeftUpperTRSM!(A, B, unitdiag::Bool=false)
    n, m = size(B)
    backend = get_backend(A)
    upper_left_kernel(backend, (n,))(A, B, n, Val(unitdiag), ndrange=(n, m))
end

function RightLowerTRSM!(A, B, unitdiag::Bool=false)
    n, m = size(B)
    backend = get_backend(A)
    right_lower_kernel(backend, (m,))(A, B, m, Val(unitdiag), ndrange=(m, n))
end

function RightUpperTRSM!(A, B, unitdiag::Bool=false)
    n, m = size(B)
    backend = get_backend(A)
    right_upper_kernel(backend, (m,))(Transpose(A), B, m, Val(unitdiag), ndrange=(m, n))
end

"""
    trsm(side, uplo, transa, diag, A, B, alpha=1.0) -> B

Solves triangular matrix systems with automatic parameter detection.
This is a simplified interface for triangular system solving.

# Arguments
- 'side': 
    - 'L': solve op(A)*X = alpha*B
    - 'R': solve X*op(A) = alpha*B
- 'uplo':
    - 'U': A is upper triangular
    - 'L': A is lower triangular  
- 'transa': operation on A
    - 'N': op(A) = A
    - 'T': op(A) = A^T
    - 'C': op(A) = A^H
- 'diag': diagonal type
    - 'N': non-unit diagonal
    - 'U': unit diagonal
- 'A': triangular matrix
- 'B': right-hand side matrix (will be overwritten with solution)
- 'alpha': scalar multiplier (default: 1.0)

# Returns
- Updated matrix B containing the solution

# Example
```julia
A = complex.(triu(randn(4, 4)), triu(randn(4, 4)))
B = complex.(randn(4, 3), randn(4, 3))
X = trsm('L', 'U', 'N', 'N', A, copy(B))
```
"""
function trsm(side, uplo, transa, diag, A, B, alpha=one(eltype(A)))
    # Scale B if alpha != 1
    if alpha != one(eltype(A))
        B .*= alpha
    end
    
    # Apply the appropriate kernel based on parameters
    if side == 'L' && uplo == 'L'
        LeftLowerTRSM!(A, B)
    elseif side == 'L' && uplo == 'U' 
        LeftUpperTRSM!(A, B)
    elseif side == 'R' && uplo == 'L'
        RightLowerTRSM!(A, B)
    elseif side == 'R' && uplo == 'U'
        RightUpperTRSM!(A, B)
    else
        error("Unsupported combination of side='$side', uplo='$uplo'")
    end
    
end

# The BLAS-order entry point. It lives here, beside the kernels, rather than in
# rectrxm.jl: unified_rectrxm! is the engine (recursion plus vendor fast paths)
# and trsm! is the public name over it.
"""
    trsm!(side, uplo, transa, diag, alpha, A, B) -> B

Triangular solve in the BLAS argument order, in place: `B` is overwritten with
the `X` that solves `op(A) * X = alpha * B` (`side = 'L'`) or
`X * op(A) = alpha * B` (`side = 'R'`), where `op(A)` is `A`, `transpose(A)` or
`adjoint(A)` for `transa = 'N'`, `'T'` or `'C'`, `A` is the `uplo` triangle and
`diag = 'U'` takes its diagonal as ones without reading it.

A thin entry point over [`unified_rectrxm!`](@ref) with `func = 'S'`, so it runs
wherever that does, for real and complex element types. On `CuMatrix` and
`ROCMatrix` of `Float32`, `Float64`, `ComplexF32` or `ComplexF64` it
reaches cuBLAS / rocBLAS `trsm!` through that function's extension methods; on
the CPU, oneAPI, Metal, `Float16` and views it runs the portable recursive
kernels. `LinearAlgebra.BLAS.trsm!` takes the same arguments but is CPU BLAS only.
"""
trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number,
      A::AbstractMatrix, B::AbstractMatrix) =
    unified_rectrxm!(side, uplo, transa, diag, alpha, 'S', A, B)

export trsm!
