using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays
include("matmul.jl")
include("trsm_base_cases.jl")

function unified_rectrsm!(side::Char, uplo::Char, A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, backend, threshold::Int=256) where T <: AbstractFloat
    # Base case: Small matrix handling
    if n <= threshold
        n, m = size(B)

        if side == 'L'
            if uplo == 'L'
                lower_left_kernel(backend, (n,))(Transpose(A), B, n, ndrange=(n, m))
            else
                upper_left_kernel(backend, (n,))(A, B, n, ndrange=(n, m))
            end
        elseif side == 'R'
            if uplo == 'L'
                right_lower_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
            else 
                right_upper_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
            end
        end
        return B
    end

    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    mid_remainder = n - mid

    A11 = view(A, 1:mid, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)
    A21 = view(A, mid+1:n, 1:mid)
    A12 = view(A, 1:mid, mid+1:n)

    if side == 'L'
        B1 = view(B, 1:mid, :)
        B2 = view(B, mid+1:n, :)
    else
        B1 = view(B, :, 1:mid)
        B2 = view(B, :, mid+1:n)
    end

    if (side == 'L' && uplo == 'L') || (side == 'R' && uplo == 'U')
        unified_rectrsm!(side, uplo, A11, mid, B1, backend, threshold)
        if side == 'L'
            GEMM_SUB!(B2, A21, B1)
        else
            GEMM_SUB!(B2, B1, A12)
        end
        unified_rectrsm!(side, uplo, A22, mid_remainder, B2, backend, threshold)
    else
        unified_rectrsm!(side, uplo, A22, mid_remainder, B2, backend, threshold)
        if side == 'L' 
            GEMM_SUB!(B1, A12, B2)
        else
            GEMM_SUB!(B1, B2, A21)
        end
        unified_rectrsm!(side, uplo, A11, mid, B1, backend, threshold)
    end
    return B
end
