export GEMM_ADD!, GEMM_SUB!

const TILE_DIM = 32

@kernel function matmul!(
    output, input1, input2, N::Int, R::Int, M::Int, alpha::Float64 = 1.0, transA::Char = 'N', transB::Char = 'N',
    ::Val{BANK} = Val(1)
) where {BANK}

    gi, gj = @index(Group,   NTuple)
    i,  j  = @index(Local,   NTuple)

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
            @inbounds tile1[i, j] =
                (transA == 'N' || transA == 'n') ? input1[I, K] :
                (transA == 'T' || transA == 't') ? input1[K, I] :
                                                   conj(input1[K, I])
        else
            @inbounds tile1[i, j] = zero(eltype(output))
        end

        K = t * TILE_DIM + i
        if K <= R && J <= M
            @inbounds tile2[i, j] =
                (transB == 'N' || transB == 'n') ? input2[K, J] :
                (transB == 'T' || transB == 't') ? input2[J, K] :
                                                   conj(input2[J, K])
        else
            @inbounds tile2[i, j] = zero(eltype(output))
        end

        @synchronize

        if I <= N && J <= M
            tmp = zero(eltype(output))
            @simd for k in 1:TILE_DIM
                @inbounds tmp += tile1[i, k] * tile2[k, j]
            end
            outval[1] += tmp
        end

        @synchronize
    end

    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    if I <= N && J <= M
        @inbounds output[I, J] += alpha * outval[1]
    end
end

# wrapper function for the GEMM_ADD kernel
function GEMM_ADD!(A, B, C; nthreads = (16, 16))
    backend = get_backend(A)
    TILE_DIM = 32
    N, R, M = size(A, 1), size(B, 1), size(B, 2)
    matmul!(backend, (TILE_DIM, TILE_DIM))(C, A, B, N, R, M, 1.0, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
end

function GEMM_SUB!(A, B, C)
    backend = get_backend(A)
    TILE_DIM = 32
    N, R, M = size(A, 1), size(C, 1), size(A, 2)
    matmul!(backend, (TILE_DIM, TILE_DIM))(A, B, C, N, R, M, -1.0, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
end