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
        if side == 'L' && uplo == 'L'
            lower_left_kernel(backend, (n,))(Transpose(A), B, n, ndrange=(n, m))
        elseif side == 'L' && uplo == 'U'
            upper_left_kernel(backend, (n,))(Transpose(A), B, n, ndrange=(n, m))
        elseif side == 'R' && uplo == 'L'
            right_lower_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
        elseif side == 'R' && uplo == 'U'
            right_upper_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
        end
        return B
    end

    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    mid_remainder = n - mid

    if side == 'L'
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n)
        B1 = view(B, 1:mid, :)
        B2 = view(B, mid+1:n, :)

        if uplo == 'L'
            A21 = view(A, mid+1:n, 1:mid)
            unified_rectrsm!('L', 'L', A11, mid, B1, backend, threshold)
            matmul!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, size(B2, 1), size(A21, 2), size(B2, 2), ndrange = (ceil(Int, size(B2, 1) / TILE_DIM) * TILE_DIM, ceil(Int, size(B2, 2) / TILE_DIM) * TILE_DIM))
            unified_rectrsm!('L', 'L', A22, mid_remainder, B2, backend, threshold)
        else
            A12 = view(A, 1:mid, mid+1:n)
            unified_rectrsm!('L', 'U', A22, mid_remainder, B2, backend, threshold)
            matmul!(backend, (TILE_DIM, TILE_DIM))(B1, A12, B2, size(B1, 1), size(A12, 2), size(B1, 2), ndrange = (ceil(Int, size(B1, 1) / TILE_DIM) * TILE_DIM, ceil(Int, size(B1, 2) / TILE_DIM) * TILE_DIM))
            unified_rectrsm!('L', 'U', A11, mid, B1, backend, threshold)
        end
    else
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n)
        B1 = view(B, :, 1:mid)
        B2 = view(B, :, mid+1:n)

        if uplo == 'L'
            A21 = view(A, mid+1:n, 1:mid)
            unified_rectrsm!('R', 'L', A22, mid_remainder, B2, backend, threshold)
            matmul!(backend, (TILE_DIM, TILE_DIM))(B1, B2, A21, size(B1, 1), size(A21, 1), size(B1, 2), ndrange = (ceil(Int, size(B1, 1) / TILE_DIM) * TILE_DIM, ceil(Int, size(B1, 2) / TILE_DIM) * TILE_DIM))
            unified_rectrsm!('R', 'L', A11, mid, B1, backend, threshold)
        else
            A12 = view(A, 1:mid, mid+1:n)
            unified_rectrsm!('R', 'U', A11, mid, B1, backend, threshold)
            matmul!(backend, (TILE_DIM, TILE_DIM))(B2, B1, A12, size(B2, 1), size(A12, 1), size(B2, 2), ndrange = (ceil(Int, size(B2, 1) / TILE_DIM) * TILE_DIM, ceil(Int, size(B2, 2) / TILE_DIM) * TILE_DIM))
            unified_rectrsm!('R', 'U', A22, mid_remainder, B2, backend, threshold)
        end
    end
    return B
end
