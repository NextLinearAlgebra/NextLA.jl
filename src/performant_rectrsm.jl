using LinearAlgebra
using KernelAbstractions
using CUDA
using StaticArrays

const TILE_DIM = 32

@kernel function both_steps_parallel(A, B, n)
    col = @index(Group)
    row = @index(Local)

    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024

    if row <= n
        diag[row] = A[row, row]
        B_c[row] = B[row, col] / diag[row]
    end

    for i in 1:n
        @synchronize
        if row > i
            B_c[row] -= (A[i, row] / diag[row]) * B_c[i]
        end
    end

    if row <= n
        B[row, col] = B_c[row]
    end
end

@kernel function upper_left_kernel(A, B, n)
    col = @index(Group)
    row = @index(Local)

    diag = @localmem eltype(A) 1024
    B_c = @localmem eltype(B) 1024

    if row <= n
        diag[row] = A[row, row]
        B_c[row] = B[row, col] / diag[row]
    end

    for i in n:-1:1
        @synchronize
        if row < i
            B_c[row] -= (A[i, row] / diag[row]) * B_c[i]
        end
    end

    if row <= n
        B[row, col] = B_c[row]
    end
end

@kernel function coalesced_matmul_kernel!(
        output, @Const(input1), @Const(input2), N, R, M,
        ::Val{BANK} = Val(1),
    ) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]

    tile1 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)
    tile2 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)

    outval = @private eltype(output) 1
    @inbounds outval[1] = zero(eltype(output))

    @uniform NUM_TILES = ceil(Int, R / TILE_DIM)

    for t in 0:(NUM_TILES - 1)
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        K = t * TILE_DIM + j

        if I <= N && K <= R
            @inbounds tile1[i, j] = input1[I, K]
        else
            @inbounds tile1[i, j] = zero(eltype(output))
        end

        K = t * TILE_DIM + i
        if K <= R && J <= M
            @inbounds tile2[i, j] = input2[K, J]
        else
            @inbounds tile2[i, j] = zero(eltype(output))
        end

        @synchronize

        if I <= N && J <= M
            out = zero(eltype(output))
            @simd for k in 1:TILE_DIM
                @inbounds out += tile1[i, k] * tile2[k, j]
            end
            outval[1] += out
        end

        @synchronize
    end

    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    if I <= N && J <= M
        @inbounds output[I, J] -= outval[1]
    end
end

function performant_rectrsm!(A::AbstractMatrix{T}, n::Int, B::AbstractMatrix{T}, side::AbstractChar = 'L', k::Int=1;
    uplo::AbstractChar='L', transpose::AbstractChar='N', threshold::Int=256) where T <: AbstractFloat

    backend = get_backend(A)

    if n <= threshold
        n, m = size(B)

        if side == 'L' && uplo == 'L' && transpose == 'N'
            both_steps_parallel(backend, (1024,))(Transpose(A), B, n, ndrange=(n, m))
        elseif side == 'L' && uplo == 'U' && transpose == 'N'
            upper_left_kernel(backend, (1024,))(Transpose(A), B, n, ndrange=(n, m))
        else
            error("Unsupported combination of side, uplo, and transposed parameters.")
        end

        return B
    end
    
    if isinteger(log2(n))
        mid = div(n, 2)
        A11 = view(A, 1:mid, 1:mid)
        A22 = view(A, mid+1:n, mid+1:n)
        A21 = view(A, mid+1:n, 1:mid)
        B1 = view(B, 1:mid, :)
        B2 = view(B, mid+1:n, :)

        performant_rectrsm!(A11, mid, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        N, R, M = size(B2, 1), size(A21, 2), size(B2, 2)
        coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        performant_rectrsm!(A22, n - mid, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)
    else
        largest_pow2 = 2 ^ floor(Int, log2(n))
        M1 = largest_pow2
        M2 = n - M1
        
        A11 = view(A, 1:M1, 1:M1)
        A22 = view(A, M1+1:n, M1+1:n)
        A21 = view(A, M1+1:n, 1:M1)
        B1 = view(B, 1:M1, :)
        B2 = view(B, M1+1:n, :)

        performant_rectrsm!(A11, M1, B1, side, k; uplo=uplo, transpose=transpose, threshold=threshold)

        N, R, M = size(B2, 1), size(A21, 2), size(B2, 2)
        coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))(B2, A21, B1, N, R, M, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))

        performant_rectrsm!(A22, M2, B2, side, k; uplo=uplo, transpose=transpose, threshold=threshold)
    end
    
    return B
end
