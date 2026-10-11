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
    return unified_rectrxm!(side, uplo, transpose, 'N', alpha, func, A, B)
end

"""
    unified_rectrxm!(side, uplo, transpose, diag, alpha, func, A, B)

As above, with `diag` selecting whether `A` has a unit diagonal.

- `diag::Char`: `'N'` reads the stored diagonal, `'U'` treats it as all ones and
  never reads it, matching the BLAS convention.

The seven-argument form delegates here with `'N'`, so existing callers are
unaffected.
"""
function unified_rectrxm!(
        side::Char,
        uplo::Char,
        transpose::Char,
        diag::Char,
        alpha::Number,
        func::Char,
        A::AbstractMatrix,
        B::AbstractMatrix
    )
    if diag != 'N' && diag != 'U'
        throw(ArgumentError("diag must be 'N' or 'U', got '$diag'"))
    end
    unitdiag = diag == 'U'
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
    unified_rec(func, side, uplo, A, n, B, threshold; unitdiag=unitdiag)
    
    # TRMM operations apply scaling after multiplication
    if func == 'M'
        B .= alpha .* B
    end
    
    return B
end

"""
    unified_rec(func, side, uplo, A, n, B, threshold; unitdiag=false)

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
# The base case, with the kernels called at the element type they are given.
function _trxm_base!(func::Char, side::Char, uplo::Char, A::AbstractMatrix,
                     B::AbstractMatrix, unitdiag::Bool)
    if func == 'S'  # Solve operations (TRSM)
        if side == 'L' && uplo == 'L'
            LeftLowerTRSM!(A, B, unitdiag)
        elseif side == 'L' && uplo == 'U'
            LeftUpperTRSM!(A, B, unitdiag)
        elseif side == 'R' && uplo == 'L'
            RightLowerTRSM!(A, B, unitdiag)
        else
            RightUpperTRSM!(A, B, unitdiag)
        end
    else  # Multiply operations (TRMM)
        if side == 'L' && uplo == 'L'
            LeftLowerTRMM!(A, B, unitdiag)
        elseif side == 'L' && uplo == 'U'
            LeftUpperTRMM!(A, B, unitdiag)
        elseif side == 'R' && uplo == 'L'
            RightLowerTRMM!(A, B, unitdiag)
        else
            RightUpperTRMM!(A, B, unitdiag)
        end
    end
    return B
end

# Half precision accumulates in Float32 and is rounded once, on store.
#
# The kernels substitute and multiply in their argument's own type, so a Float16
# base case sums Float16 terms into a Float16 running value. Measured on a
# 128-wide lower-triangular solve, that is 0.0052 relative error against 0.00021
# for the same inputs worked in Float32 -- and it grows with the block, where the
# Float32 figure is flat at the rounding of Float16 storage. The operands are
# exactly as accurate either way; only the accumulation changes.
#
# vicki-development reached the same conclusion in dispatch_trsm!/dispatch_trmm!
# (src/rectrxm.jl), which promoted to Float32 around a vendor BLAS call. Those
# two go through wrappers.jl, which this repo does not take, so the promotion is
# done here around the portable kernels instead. It is the same treatment
# GEMM_ADD!/GEMM_SUB! give a Float16 destination.
#
# This costs two temporaries, for Float16 only. Every other element type reaches
# the kernels unchanged, which is why the method below dispatches on the type
# rather than branching inside the hot path.
#
# A transposed operand is widened through its parent and the transpose put back
# on the copy. For 'T' and 'C', unified_rectrxm! wraps A in Transpose/Adjoint and
# the recursion then takes views of that wrapper; broadcasting over a view of a
# transposed GPU array has no device method and falls back to scalar indexing.
# The parent, or the matching block of it, is a plain device array or view.
_widen32(A::AbstractMatrix) = Float32.(A)
_widen32(A::Transpose) = transpose(_widen32(parent(A)))
_widen32(A::Adjoint) = adjoint(_widen32(parent(A)))
_widen32(A::SubArray{<:Any,2,<:Transpose}) =
    transpose(_widen32(view(parent(parent(A)), reverse(parentindices(A))...)))
_widen32(A::SubArray{<:Any,2,<:Adjoint}) =
    adjoint(_widen32(view(parent(parent(A)), reverse(parentindices(A))...)))

function _trxm_base!(func::Char, side::Char, uplo::Char, A::AbstractMatrix{Float16},
                     B::AbstractMatrix{Float16}, unitdiag::Bool)
    A32 = _widen32(A)
    B32 = Float32.(B)
    _trxm_base!(func, side, uplo, A32, B32, unitdiag)
    B .= Float16.(B32)      # broadcast, so the copy back stays on the device
    return B
end

function unified_rec(func::Char, side::Char, uplo::Char, A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, threshold::Int=256; unitdiag::Bool=false) where T <: Union{AbstractFloat, Complex{<:AbstractFloat}}
    # src/trsm/trmm.jl's kernels index one TILE_DIM = 16 tile, so a multiply's base
    # case is valid only up to 16 rows: a larger one reads the wrong elements and
    # returns a silently wrong result (0.52 relative error at n = 64). Clamping here
    # covers every caller, not only unified_rectrxm!, which already passes 16.
    func == 'M' && (threshold = min(threshold, 16))
    # Base case: use optimized kernel functions for small matrices
    if n <= threshold
        return _trxm_base!(func, side, uplo, A, B, unitdiag)
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
        unified_rec(func, side, uplo, A11, mid, B1, threshold; unitdiag=unitdiag)
        
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
        
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold; unitdiag=unitdiag)
    else
        # Backward substitution ordering: A22 → GEMM → A11
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold; unitdiag=unitdiag)
        
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
        
        unified_rec(func, side, uplo, A11, mid, B1, threshold; unitdiag=unitdiag)
    end
end

export unified_rectrxm!