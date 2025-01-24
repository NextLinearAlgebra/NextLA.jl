using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays
include("matmul.jl")
include("trmm_base_cases.jl")
const TILE_DIM = 32

export perf_trmm!
export recTRMM_LL!
export recTRMM_UL!
export recTRMM_LR!


# implementing the base kernels, dependent on the occupied side of A,
# in place, store result in B





# the recursive trmm functions
# The first letter is the triangle referencing, Upper/ Lower
# The second letter is the position of the A argument

function lower_left_rectrmm!(A, n, B, backend, threshold = 16)
    if n <= threshold
        LeftLowerTRMM!(A, B)
    else
        split = div(n, 2)
        # Step 1: subdivide the two matrices into segments
        A11 = @view(A[1:split, 1:split])
        A21 = @view(A[split+1:end, 1:split])
        A22 = @view(A[split+1:end, split+1:end])
        

        B1 = @view(B[1:split, 1:end])
        B2 = @view(B[split+1:end, 1:end])
         
        
        # Step 2. operate recursively on B2 : B2 = A22 * B2
        lower_left_rectrmm!(A22, split, B2, backend, threshold)

        # Step 3: GEMM AND ADDITION: B2 = A21*B1 + B2
        GEMM_ADD!(A21, B1, B2)

        # Step 4: operate recursively on B1 : B1 = A11 * B1
        lower_left_rectrmm!(A11, n-split, B1, backend, threshold)
        

    end
end



function upper_left_rectrmm!(A, n, B, backend, threshold = 16)
    size_A = size(A, 1)
    if size_A <= threshold
        LeftUpperTRMM!(A, B)
    else
        split = div(size_A, 2)
        # Step 1: subdivide the two matrices into segments
        A11 = @view(A[1:split, 1:split])
        A12 = @view(A[1:split, split+1:end])
        A22 = @view(A[split+1:end, split+1:end])
        

        B1 = @view(B[1:split, 1:end])
        B2 = @view(B[split+1:end, 1:end])
         
        
        # Step 2. operate recursively on B1 : B1 = A11 * B1
        upper_left_rectrmm!(A11, n, B1, backend, threshold)

        # Step 3: GEMM AND ADDITION: B1 = A12*B2 + B1
        GEMM_ADD!(A12, B2, B1)

        # Step 4: operate recursively on B2 : B2 = A22 * B2
        upper_left_rectrmm!(A22, n, B2, backend, threshold)
        

    end
end


# splits are now vertical in matrix B
# splits in A maintain their structure
function lower_right_rectrmm!(A, n, B, backend, threshold = 16)
    if n <= threshold
        RightLowerTRMM!(A, B)
    else
        split = div(n, 2)
        # Step 1: subdivide the two matrices into segments
        A11 = @view(A[1:split, 1:split])
        A21 = @view(A[split+1:end, 1:split])
        A22 = @view(A[split+1:end, split+1:end])
        

        B1 = @view(B[1:end, 1:split])
        B2 = @view(B[1:end, split+1:end])
         
        
        # Step 2. operate recursively on B1 : B1 = A11 * B1
        lower_right_rectrmm!(A11, n, B1, backend, threshold)

        # Step 3: GEMM AND ADDITION: B1 = B2*A21 + B1: remember to switch order of args
        GEMM_ADD!(B2, A21, B1)

        # Step 4: operate recursively on B2 : B2 = A22 * B2
        lower_right_rectrmm!(A22, n, B2, backend, threshold)
        

    end
end
# TO DO: implement recTRMM_LR and rec_TRMM_UR





# the main trmm call.

# CHARACTER ARGUMENTS
# Multiplication Order
# side	Meaning
# 'L'	The argument goes on the left side of a matrix-matrix operation.
# 'R'	The argument goes on the right side of a matrix-matrix operation.

# Triangle Referencing
# uplo/ul	Meaning
# 'U'	    Only the upper triangle of the matrix will be used.
# 'L'	    Only the lower triangle of the matrix will be used.

# Transposition Operation
# trans/tX	Meaning
# 'N'	    The input matrix X is not transposed or conjugated.
# 'T'	    The input matrix X will be transposed.
# 'C'	    The input matrix X will be conjugated and transposed.

# Unit Diagonal
# diag/dX	Meaning
# 'N'	The diagonal values of the matrix X will be read.
# 'U'	The diagonal of the matrix X is assumed to be all ones.

# TRMM/ TRSM ?
# 'S'       Solve
# 'M'       Multiply



# Update B as alpha*A*B
# Return the updated B
function perf_trmm!(side, ul, tA, dA, alpha, A, B)
    # assume dA = 'N'
    # call the appropriate TRMM recursive function
    if side == 'L' && ul == 'L' && tA == 'N'
        recTRMM_LL!(A, B)
    else
        error("Unsupported combination of parameters")
    end


end









# Formulation
# Assumptions: A is lower triangular, solving B = AB; A is nxn, B is nxm;

# Recursion:
# We are performing a recursion with a threshold in which we use the base case TRMM.

# If n is less than or equal to the threshold:
#     Call the base case kernel function to carry out the TRMM in place.

# Split the matrix A into 4 equal-sized submatrices, but focus on the 3 with non-zero elements:
#     A11: Top left (lower triangular)
#     A22: Bottom right (lower triangular)
#     A21: Bottom left (full matrix)
#     A12: Top right (empty) *not used in computation

# Split matrix B into two halves:
#     B1: Top half
#     B2: Bottom half

# Call recursion on the top left submatrix (A11) with size n/2 x n/2: recTRMM(A11, B1) : B1 = A11*B1

# Perform GEMM and addition: Update TOP half of B: B1 = A21*B2 + B1

# Call recursion on the bottom right submatrix (A22) with size n/2 x n/2: recTRMM(A22, B2)

# Base Case of the Triangular Solve (TRSM): TO DO

# For each row in B: Get the diagonal element from A for that row, 
# then update the corresponding entry in B by dividing it by the diagonal element.

# Then for each row below the current row, update B by subtracting contributions from rows above it.

# Store the updated value back into B.
