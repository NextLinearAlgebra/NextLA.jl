using LinearAlgebra
using KernelAbstractions
using CUDA

include("performant_trsm_2 copy.jl")

function performant_rectrsm!(A::AbstractMatrix{T}, n::Int, B::AbstractMatrix{T}, side::AbstractChar = 'L', k::Int=1;
                  uplo::AbstractChar='L', transpose::AbstractChar='N', threshold::Int=1024) where T
    
    if n <= threshold
        performant_trsm_2_2!(side, uplo, transpose, A, B)
        return B
    end

    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end

    A11 = view(A, 1:mid, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)
    A21 = view(A, mid+1:n, 1:mid)
    B1 = view(B, 1:mid, :)
    B2 = view(B, mid+1:n, :)

    performant_rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)
    CUDA.CUBLAS.gemm!('N', 'N', -1, A21, B1, 1, B2)
    performant_rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)
    
    return B
end

@kernel function gpu_gemm_kernel!(A, B, C, m, n, k)
    row, col = @index(Global, NTuple)
    if row <= m && col <= n
        sum = zero(eltype(C))
        for i in 1:k
            sum += A[row, i] * B[i, col]
        end
        C[row, col] -= sum
    end
end

function gpu_gemm_update!(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}, m::Int, n::Int) where T
    backend = get_backend(A)
    workgroup_size = (16, 16)
    gpu_gemm_kernel!(backend, workgroup_size)(A, B, C, m, n, size(A, 2), ndrange=(size(C)))
    KernelAbstractions.synchronize(backend)
end
