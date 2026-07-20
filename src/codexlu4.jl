export getrf_recursive!
 
using CUDA
include("wrappers.jl")
include("rectrxm.jl")
include("matmul.jl")
 
const LU_BASECASE = 4096
 
function lu_basecase_nopiv!(A::AbstractMatrix)
    getrf_nopiv!(A)
    return A
end
 
function getrf_recursive!(A::AbstractMatrix, block_size::Int=LU_BASECASE)
    n = size(A, 1)
    @assert n == size(A, 2) "LU requires a square matrix"
 
    if n <= block_size
        lu_basecase_nopiv!(A)
        return A
    end
 
    n1 = 2^floor(Int, log2(n)) ÷ 2
 
    A11 = @view A[1:n1,       1:n1]
    A12 = @view A[1:n1,       n1+1:end]
    A21 = @view A[n1+1:end,   1:n1]
    A22 = @view A[n1+1:end,   n1+1:end]
 
    getrf_recursive!(A11, block_size)
 
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A11, A12)
 
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A11, A21)
 
    recgemm!(-1.0, A21, A12, 1.0, A22)
 
    getrf_recursive!(A22, block_size)
 
    return A
end
 
function getrf_recursive!(A::FullMixedPrec)
    if A.BaseCase !== nothing
        getrf_recursive!(A.BaseCase, LU_BASECASE)
        return A
    end
 
    getrf_recursive!(A.A11)
 
    unified_rectrxm!('L', 'L', 'N', 'U', 1.0, 'S', A.A11, A.A12)
 
    unified_rectrxm!('R', 'U', 'N', 'N', 1.0, 'S', A.A11, A.A21)
 
    recgemm!(-1.0, A.A21, A.A12, 1.0, A.A22)
 
    getrf_recursive!(A.A22)
 
    return A
end