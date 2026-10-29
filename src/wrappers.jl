export potrf!, trsm!, trmm!, syrk!, gemm!, gemmEx!

using CUDA
using LinearAlgebra

# ==============================================================================
# --- POTRF (Cholesky Factorization) ---
# ==============================================================================

potrf!(uplo::Char, A::StridedCuMatrix) = CUSOLVER.potrf!(uplo, A)
if isdefined(@__MODULE__, :AMDGPU)
    potrf!(uplo::Char, A::AMDGPU.StridedROCMatrix) = AMDGPU.rocSOLVER.potrf!(uplo, A)
end

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

if isdefined(@__MODULE__, :AMDGPU)
    trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::AMDGPU.StridedROCMatrix{T}, B::AMDGPU.StridedROCMatrix{T}) where T = AMDGPU.rocBLAS.trsm!(side, uplo, transa, diag, T.(alpha), A, B)
end

# ==============================================================================
# --- TRMM (Triangular Multiply) ---
# ==============================================================================

trmm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, C::StridedCuMatrix{T}) where T = CUBLAS.trmm!(side, uplo, transa, diag, alpha, A, B, C)

if isdefined(@__MODULE__, :AMDGPU)
    trmm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha, A::AMDGPU.StridedROCMatrix{T}, B::AMDGPU.StridedROCMatrix{T}, C::AMDGPU.StridedROCMatrix{T}) where T = AMDGPU.rocBLAS.trmm!(side, uplo, transa, diag, T.(alpha), A, B, C)
end

# ==============================================================================
# --- SYRK (Symmetric Rank-k Update) ---
# ==============================================================================

syrk!(uplo::Char, trans::Char, alpha, A::StridedCuVecOrMat{T}, beta, C::StridedCuMatrix{T}) where T = CUBLAS.syrk!(uplo, trans, alpha, A, beta, C)

if isdefined(@__MODULE__, :AMDGPU)
    syrk!(uplo::Char, trans::Char, alpha, A::AMDGPU.StridedROCVecOrMat{T}, beta, C::AMDGPU.StridedROCMatrix{T}) where T = AMDGPU.rocBLAS.syrk!(uplo, trans, T.(alpha), A, T.(beta), C)
end

# ==============================================================================
# --- GEMM (General Matrix Multiply) ---
# ==============================================================================

gemm!(transA::Char, transB::Char, alpha, A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}, beta, C::StridedCuVecOrMat{T}) where T = CUBLAS.gemm!(transA, transB, alpha, A, B, beta, C)

if isdefined(@__MODULE__, :AMDGPU)
    gemm!(transA::Char, transB::Char, alpha::T, A::AMDGPU.StridedROCVecOrMat{T}, B::AMDGPU.StridedROCVecOrMat{T}, beta::T, C::AMDGPU.StridedROCVecOrMat{T}) where T = AMDGPU.rocBLAS.gemm!(transA, transB, alpha, A, B, beta, C)
end

# ==============================================================================
# --- GEMM EX (Mixed-Precision General Matrix Multiply) ---
# ==============================================================================

gemmEx!(transA::Char, transB::Char, alpha, A::StridedCuVecOrMat, B::StridedCuVecOrMat, beta, C::StridedCuVecOrMat) = CUBLAS.gemmEx!(transA, transB, alpha, A, B, beta, C)

if isdefined(@__MODULE__, :AMDGPU)
    gemmEx!(transA::Char, transB::Char, alpha, A::AMDGPU.StridedROCVecOrMat, B::AMDGPU.StridedROCVecOrMat, beta, C::AMDGPU.StridedROCVecOrMat{T}) where T = AMDGPU.rocBLAS.gemm!(transA, transB, T.(alpha), T.(A), T.(B), T.(beta), C)
end
