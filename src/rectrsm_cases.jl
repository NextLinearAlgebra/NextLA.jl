using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays
include("matmul.jl")
include("trsm_base_cases.jl")

# Function for the lower-left case (side == 'L' and uplo == 'L')
function lower_left_rectrsm!(A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, backend, threshold::Int=256) where T <: AbstractFloat
    # Base case: Small matrix handling
    if n <= threshold
        n, m = size(B)
        lower_left_kernel(backend, (n,))(Transpose(A), B, n, ndrange=(n, m))
        return B
    end
    
    if isinteger(log2(n))
        mid = div(n, 2)
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n)
        A21 = view(A, mid+1:n, 1:mid)
        B1 = view(B, 1:mid, :)
        B2 = view(B, mid+1:n, :)

        # Solve the first half
        lower_left_rectrsm!(A11, mid, B1, backend, threshold)

        # Update the second half
        N, R, M = size(B2, 1), size(A21, 2), size(B2, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the second half
        lower_left_rectrsm!(A22, n - mid, B2, backend, threshold)
    else
        # Handle non-power-of-two sizes
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        A11 = view(A, 1:M1, 1:M1)
        A22 = view(A, M1+1:n, M1+1:n)
        A21 = view(A, M1+1:n, 1:M1)
        B1 = view(B, 1:M1, :)
        B2 = view(B, M1+1:n, :)

        # Solve the first part
        lower_left_rectrsm!(A11, M1, B1, backend, threshold)

        # Update the second part
        N, R, M = size(B2, 1), size(A21, 2), size(B2, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the second part
        lower_left_rectrsm!(A22, M2, B2, backend, threshold)
    end
    return B
end

# Function for the upper-left case (side == 'L' and uplo == 'U')
function upper_left_rectrsm!(A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, backend, threshold::Int=256) where T <: AbstractFloat
    # Base case: Small matrix handling
    if n <= threshold
        n, m = size(B)
        upper_left_kernel(backend, (n,))(A, B, n, ndrange=(n, m))
        return B
    end
    if isinteger(log2(n))
        mid = div(n, 2)
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n) 
        A12 = view(A, 1:mid, mid+1:n)
        B1 = view(B, 1:mid, :)
        B2 = view(B, mid+1:n, :)

        # Solve the first half
        upper_left_rectrsm!(A22, n - mid, B2, backend, threshold)
        
        # Update the second half
        N, R, M = size(B1, 1), size(A12, 2), size(B1, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B1, A12, B2, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the second half
        upper_left_rectrsm!(A11, mid, B1, backend, threshold)
    else
        # Handle non-power-of-two sizes
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        A11 = view(A, 1:M1, 1:M1)
        A22 = view(A, M1+1:n, M1+1:n)
        A12 = view(A, 1:M1, M1+1:n)       
        B1 = view(B, 1:M1, :)
        B2 = view(B, M1+1:n, :)

        # Solve the first part
        upper_left_rectrsm!(A22, M2, B2, backend, threshold)

        # Update the second part
        N, R, M = size(B1, 1), size(A12, 2), size(B1, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B1, A12, B2, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the second part
        upper_left_rectrsm!(A11, M1, B1, backend, threshold)
    end

    return B
end

# Function for the lower-right case (side == 'R' and uplo == 'L')
function lower_right_rectrsm!(A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, backend, threshold::Int=256) where T <: AbstractFloat
    # Base case: Small matrix handling
    if n <= threshold
        n, m = size(B)
        right_lower_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
        return B
    end
    
    if isinteger(log2(n))
        mid = div(n, 2)
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n)
        A21 = view(A, mid+1:n, 1:mid)
        B1 = view(B, :, 1:mid)
        B2 = view(B, :, mid+1:n)

        # Solve the second half
        lower_right_rectrsm!(A22, n - mid, B2, backend, threshold)

        # Update the first half
        N, R, M = size(B1, 1), size(A21, 1), size(B1, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B1, B2, A21, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the first half
        lower_right_rectrsm!(A11, mid, B1, backend, threshold)
    else
        # Handle non-power-of-two sizes
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        A11 = view(A, 1:M1, 1:M1)
        A22 = view(A, M1+1:n, M1+1:n)
        A21 = view(A, M1+1:n, 1:M1)
        B1 = view(B, :, 1:M1)
        B2 = view(B, :, M1+1:n)

        # Solve the second part
        lower_right_rectrsm!(A22, M2, B2, backend, threshold)
        # Update the first part
        N, R, M = size(B1, 1), size(A21, 1), size(B1, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B1, B2, A21, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the first part
        lower_right_rectrsm!(A11, M1, B1, backend, threshold)
    end
    return B
end

# Function for the upper-right case (side == 'R' and uplo == 'U')
function upper_right_rectrsm!(A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, backend, threshold::Int=256) where T <: AbstractFloat
    # Base case: Small matrix handling
    if n <= threshold
        n, m = size(B)
        right_upper_kernel(backend, (m,))(Transpose(A), B, m, ndrange=(m, n))
        return B
    end
    
    if isinteger(log2(n))
        mid = div(n, 2)
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n)
        A12 = view(A, 1:mid, mid+1:n)
        B1 = view(B, :, 1:mid)
        B2 = view(B, :, mid+1:n)

        # Solve the first half
        upper_right_rectrsm!(A11, mid, B1, backend, threshold)

        # Update the second half
        N, R, M = size(B2, 1), size(A12, 2), size(B2, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B2, B1, A12, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the second half
        upper_right_rectrsm!(A22, n - mid, B2, backend, threshold)
    else
        # Handle non-power-of-two sizes
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        A11 = view(A, 1:M1, 1:M1)
        A22 = view(A, M1+1:n, M1+1:n)
        A12 = view(A, 1:M1, M1+1:n)
        B1 = view(B, :, 1:M1)
        B2 = view(B, :, M1+1:n)

        # Solve the first part
        upper_right_rectrsm!(A11, M1, B1, backend, threshold)

        # Update the second part
        N, R, M = size(B2, 1), size(A12, 1), size(B2, 2)
        matmul!(backend, (TILE_DIM, TILE_DIM))(B2, B1, A12, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        # Solve the second part
        upper_right_rectrsm!(A22, M2, B2, backend, threshold)
    end

    return B
end
