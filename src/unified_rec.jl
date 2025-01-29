include("unifrec.jl")
include("matmul.jl")
include("trsm_base_cases.jl")

"""
Unified recursive function for triangular matrix solve (TRSM) and multiply (TRMM) operations.

This function supports both solving triangular systems of equations and performing triangular matrix multiplications.

Arguments:
- side::Char: Specifies the side of the operation:
    - 'L': Left multiplication (A * B or inv(A) * B).
    - 'R': Right multiplication (B * A or B * inv(A)).
- uplo::Char: Specifies the triangular part of the matrix to reference:
    - 'U': Use the upper triangle.
    - 'L': Use the lower triangle.
- transpose::Char: Specifies the transposition operation:
    - 'N': No transpose.
    - 'T': Transpose.
    - 'C': Conjugate transpose.
- alpha::Number: Scalar multiplier applied to the operation.
- func::Char: Specifies the function type:
    - 'S': Solve (TRSM, A * X = alpha * B).
    - 'M': Multiply (TRMM, Update B = alpha * A * B or alpha * B * A).
- A::AbstractMatrix: The triangular matrix.
- B::AbstractMatrix: The matrix to multiply or solve for.

Returns:
- Updated matrix `B` after performing the specified operation.

Notes:
- The function modifies `B` in place.
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
    threshold = 16
    n = size(A, 1)

    if transpose == 'T' || transpose == 'C'
        A = (transpose == 'T') ? Transpose(A) : Adjoint(A)
        uplo = (uplo == 'L') ? 'U' : 'L'
    end    
    
    # TRSM: Triangular Solve
    if func == 'S'
        threshold = 256
        B .= alpha .* B
    end
    unified_rec(func, side, uplo, A, n, B, threshold)
    if func == 'M'
        B .= alpha .* B
    end
    return B
end


