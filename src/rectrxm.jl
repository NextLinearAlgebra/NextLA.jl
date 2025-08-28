"""
    unified_rectrxm!(side, uplo, transpose, alpha, func, A, B)

Unified recursive function for triangular matrix solve (TRSM) and multiply (TRMM) operations.

This function supports both solving triangular systems of equations and performing triangular matrix multiplications
using recursive algorithms that are cache-friendly and numerically stable.

# Arguments
- `side::Char`: Specifies the side of the operation ('L' for left, 'R' for right)
    - 'L': Left multiplication (A * B or inv(A) * B)
    - 'R': Right multiplication (B * A or B * inv(A))
- `uplo::Char`: Specifies the triangular part of the matrix to reference
    - 'U': Use the upper triangle
    - 'L': Use the lower triangle
- `transpose::Char`: Specifies the transposition operation
    - 'N': No transpose
    - 'T': Transpose
    - 'C': Conjugate transpose
- `alpha::Number`: Scalar multiplier applied to the operation
- `func::Char`: Specifies the function type
    - 'S': Solve (TRSM, A * X = alpha * B)
    - 'M': Multiply (TRMM, Update B = alpha * A * B or alpha * B * A)
- `A::AbstractMatrix`: The triangular matrix
- `B::AbstractMatrix`: The matrix to multiply or solve for (modified in-place)

# Returns
- Updated matrix `B` after performing the specified operation

# Algorithm
Uses recursive divide-and-conquer approach that:
1. Partitions matrices into 2x2 block structure
2. Applies operations recursively on subblocks
3. Handles base cases with optimized kernel functions
4. Maintains numerical stability through careful ordering

# Implementation Notes
- The function modifies `B` in place for efficiency
- Uses different thresholds for TRSM (256) vs TRMM (16) operations
- Automatically handles transpose operations by adjusting matrix views
- Recursive partitioning adapts to matrix size for optimal performance
"""
function unified_rectrxm!(
        side::Char, 
        uplo::Char, 
        transpose::Char, 
        alpha::Number, 
        func::Char, 
        A::AbstractMatrix, 
        B::AbstractMatrix
    )
    threshold = 16  # Default threshold for TRMM operations
    n = size(A, 1)

    # Handle transpose operations by adjusting matrix view and uplo flag
    if transpose == 'T' || transpose == 'C'
        A = (transpose == 'T') ? Transpose(A) : Adjoint(A)
        uplo = (uplo == 'L') ? 'U' : 'L'
    end    
    
    # TRSM operations require different handling and larger threshold
    if func == 'S'
        threshold = 256  # Larger threshold for solve operations
        B .= alpha .* B  # Apply scaling before solve
    end
    
    # Call recursive kernel
    unified_rec(func, side, uplo, A, n, B, threshold)
    
    # TRMM operations apply scaling after multiplication
    if func == 'M'
        B .= alpha .* B
    end
    
    return B
end

"""
    unified_rec(func, side, uplo, A, n, B, threshold)

Recursive kernel for unified triangular matrix operations.

This function implements the divide-and-conquer recursive algorithm that partitions
matrices into 2x2 block structure and applies the appropriate sequence of operations.

# Arguments
- `func::Char`: Operation type ('S' for solve, 'M' for multiply)
- `side::Char`: Operation side ('L' for left, 'R' for right)  
- `uplo::Char`: Triangular part ('U' for upper, 'L' for lower)
- `A::AbstractMatrix{T}`: Triangular coefficient matrix
- `n::Int`: Matrix dimension to process
- `B::AbstractMatrix{T}`: Target matrix (modified in-place)
- `threshold::Int`: Recursion base case threshold (default: 256)

# Algorithm
The recursion follows different orderings based on the operation type:
1. For forward substitution: A11 → GEMM → A22
2. For backward substitution: A22 → GEMM → A11
This ensures numerical stability and correctness of the triangular solve.
"""
function unified_rec(func::Char, side::Char, uplo::Char, A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, threshold::Int=256) where T <: AbstractFloat
    # Base case: use optimized kernel functions for small matrices
    if n <= threshold
        if func == 'S'  # Solve operations (TRSM)
            if side == 'L' && uplo == 'L'
                LeftLowerTRSM!(A, B)
            elseif side == 'L' && uplo == 'U'
                LeftUpperTRSM!(A, B)
            elseif side == 'R' && uplo == 'L'
                RightLowerTRSM!(A, B)
            else
                RightUpperTRSM!(A, B)
            end
        else  # Multiply operations (TRMM)
            if side == 'L' && uplo == 'L'
                LeftLowerTRMM!(A, B)
            elseif side == 'L' && uplo == 'U'
                LeftUpperTRMM!(A, B)
            elseif side == 'R' && uplo == 'L'
                RightLowerTRMM!(A, B)
            else
                RightUpperTRMM!(A, B)
            end
        end
        return B
    end

    # Determine partition size for optimal cache performance
    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    mid_remainder = n - mid

    # Create 2x2 block partition of matrix A
    A11 = view(A, 1:mid, 1:mid)                    # Upper-left block
    A22 = view(A, mid+1:n, mid+1:n)                # Lower-right block  
    A21 = view(A, mid+1:n, 1:mid)                  # Lower-left block
    A12 = view(A, 1:mid, mid+1:n)                  # Upper-right block

    # Partition matrix B based on operation side
    if side == 'L'
        B1 = view(B, 1:mid, :)        # Upper block rows
        B2 = view(B, mid+1:n, :)      # Lower block rows
    else
        B1 = view(B, :, 1:mid)        # Left block columns
        B2 = view(B, :, mid+1:n)      # Right block columns
    end

    # Apply recursive algorithm with correct ordering for numerical stability
    # Different operation types require different orderings to maintain correctness
    if (side == 'L' && uplo == 'L' && func == 'S') || 
        (side == 'R' && uplo == 'U' && func == 'S') || 
        (side == 'L' && uplo == 'U' && func == 'M') || 
        (side == 'R' && uplo == 'L' && func == 'M')
        
        # Forward substitution ordering: A11 → GEMM → A22
        unified_rec(func, side, uplo, A11, mid, B1, threshold)
        
        # Apply rank-k update between recursive calls
        if side == 'L'
            if func == 'S'
                GEMM_SUB!(B2, A21, B1)  # B2 := B2 - A21 * B1
            else
                GEMM_ADD!(A12, B2, B1)  # B1 := B1 + A12 * B2
            end
        else
            if func == 'S'
                GEMM_SUB!(B2, B1, A12)  # B2 := B2 - B1 * A12
            else
                GEMM_ADD!(B2, A21, B1)  # B2 := B2 + A21 * B1
            end
        end
        
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold)
    else
        # Backward substitution ordering: A22 → GEMM → A11
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold)
        
        # Apply rank-k update between recursive calls
        if side == 'L'
            if func == 'S'
                GEMM_SUB!(B1, A12, B2)  # B1 := B1 - A12 * B2
            else
                GEMM_ADD!(A21, B1, B2)  # B2 := B2 + A21 * B1
            end
        else
            if func == 'S'
                GEMM_SUB!(B1, B2, A21)  # B1 := B1 - B2 * A21
            else
                GEMM_ADD!(B1, A12, B2)  # B1 := B1 + A12 * B2
            end
        end
        
        unified_rec(func, side, uplo, A11, mid, B1, threshold)
    end
end

export unified_rectrxm!