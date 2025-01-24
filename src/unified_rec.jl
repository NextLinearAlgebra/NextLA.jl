include("matmul.jl")
include("trsm_base_cases.jl")
include("rectrsm_cases.jl")
include("rectrmm_cases.jl")

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
    backend = get_backend(A)
    n = size(A, 1)

    if transpose == 'T' || transpose == 'C'
        A = (transpose == 'T') ? Transpose(A) : Adjoint(A)
        uplo = (uplo == 'L') ? 'U' : 'L'
    end
    

    
    # TRSM: Triangular Solve
    if func == 'S'
        threshold = 256
        # Scale B with alpha before solving
        B .= alpha .* B

        if side == 'L' && uplo == 'L'
            return lower_left_rectrsm!(A, n, B, backend, threshold)
        elseif side == 'L' && uplo == 'U'
            return upper_left_rectrsm!(A, n, B, backend, threshold)
        elseif side == 'R' && uplo == 'L'
            return lower_right_rectrsm!(A, n, B, backend, threshold)
        elseif side == 'R' && uplo == 'U'
            return upper_right_rectrsm!(A, n, B, backend, threshold)
        else
            error("Unsupported combination of side, uplo, and transpose parameters.")
        end

    # TRMM: Triangular Multiply
    elseif func == 'M'
        threshold = 16
        if side == 'L' && uplo == 'L'
            lower_left_rectrmm!(A, n, B, backend, threshold)
        elseif side == 'L' && uplo == 'U'
            upper_left_rectrmm!(A, n, B, backend, threshold)
        elseif side == 'R' && uplo == 'L'
            lower_right_rectrmm!(A, n, B, backend, threshold)
        elseif side == 'R' && uplo == 'U'
            upper_right_rectrmm!(A, n, B, backend, threshold)
        else
            error("Unsupported combination of side, uplo, and transpose parameters.")
        end

        # Scale B with alpha after multiplication
        B .= alpha .* B
        return B
    else
        error("Invalid operation type. Use 'S' for TRSM or 'M' for TRMM.")
    end
end


