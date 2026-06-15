export potrf!, trsm!, trmm!, syrk!, gemm!, gemmEx!

using CUDA
using AMDGPU
using LinearAlgebra

# ==============================================================================
# --- POTRF (Cholesky Factorization) ---
# ==============================================================================

potrf!(uplo::Char, A::StridedCuMatrix) = CUSOLVER.potrf!(uplo, A)
potrf!(uplo::Char, A::StridedROCMatrix) = AMDGPU.rocSOLVER.potrf!(uplo, A)

"""
    potrf!(A::AnyGPUArray)

High-level wrapper for Cholesky factorization. Defaults to Lower triangular ('L').
Handles memory-taxing Float16 spoofing by temporarily casting to Float32.
"""
function potrf!(A::AbstractMatrix{T}) where T
    if eltype(A) == Float16
        A_f32 = Float32.(A)
        potrf!('L', A_f32)
        A .= Float16.(A_f32)
    else
        potrf!('L', A)
    end
end

# ==============================================================================
# --- TRSM (Triangular Solve) ---
# ==============================================================================

trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T = CUBLAS.trsm!(side, uplo, transa, diag, alpha, A, B)

trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::StridedROCMatrix{T}, B::StridedROCMatrix{T}) where T = AMDGPU.rocBLAS.trsm!(side, uplo, transa, diag, T.(alpha), A, B)

# ==============================================================================
# --- TRMM (Triangular Multiply) ---
# ==============================================================================

trmm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, C::StridedCuMatrix{T}) where T = CUBLAS.trmm!(side, uplo, transa, diag, alpha, A, B, C)

trmm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::StridedROCMatrix{T}, B::StridedROCMatrix{T}, C::StridedROCMatrix{T}) where T = AMDGPU.rocBLAS.trmm!(side, uplo, transa, diag, T.(alpha), A, B, C)

# ==============================================================================
# --- SYRK (Symmetric Rank-k Update) ---
# ==============================================================================

syrk!(uplo::Char, trans::Char, alpha, A::StridedCuVecOrMat{T}, beta, C::StridedCuMatrix{T}) where T = CUBLAS.syrk!(uplo, trans, alpha, A, beta, C)

syrk!(uplo::Char, trans::Char, alpha, A::StridedROCVecOrMat{T}, beta, C::StridedROCMatrix{T}) where T = AMDGPU.rocBLAS.syrk!(uplo, trans, T.(alpha), A, T.(beta), C)

# ==============================================================================
# --- GEMM (General Matrix Multiply) ---
# ==============================================================================

gemm!(transA::Char, transB::Char, alpha, A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}, beta, C::StridedCuVecOrMat{T}) where T = CUBLAS.gemm!(transA, transB, alpha, A, B, beta, C)

gemm!(transA::Char, transB::Char, alpha::T, A::StridedROCVecOrMat{T}, B::StridedROCVecOrMat{T}, beta::T, C::StridedROCVecOrMat{T}) where T = AMDGPU.rocBLAS.gemm!(transA, transB, alpha, A, B, beta, C)

# ==============================================================================
# --- GEMM EX (Mixed-Precision General Matrix Multiply) ---
# ==============================================================================

gemmEx!(transA::Char, transB::Char, alpha, A::StridedCuVecOrMat, B::StridedCuVecOrMat, beta, C::StridedCuVecOrMat) = CUBLAS.gemmEx!(transA, transB, alpha, A, B, beta, C)

gemmEx!(transA::Char, transB::Char, alpha, A::StridedROCVecOrMat, B::StridedROCVecOrMat, beta, C::StridedROCVecOrMat{T}) where T = AMDGPU.rocBLAS.gemm!(transA, transB, T.(alpha), T.(A), T.(B), T.(beta), C)