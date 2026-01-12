export SYRK_KERNEL!

const TILE_DIM = 32

@kernel function syrk_kernel!(
    uplo::Char, trans::Char, alpha::Number, A::AbstractArray, beta::Number, C::AbstractArray,
    ::Val{BANK} = Val(1)
) where {BANK}

    N = size(C, 1);
    M = (trans == 'N') ? size(A, 2) : size(A, 1)

    gi, gj = @index(Group,   NTuple)
    i,  j  = @index(Local,   NTuple)

    # if gj > gi
    #     return
    # end
    if gj <= gi
        TILE_DIM = @uniform @groupsize()[1]

        tile1 = @localmem eltype(C) (TILE_DIM + BANK, TILE_DIM)
        tile2 = @localmem eltype(C) (TILE_DIM + BANK, TILE_DIM)

        outval = @private eltype(C) 1
        @inbounds outval[1] = zero(eltype(C))

        @uniform NUM_TILES = ceil(Int, M / TILE_DIM)

        for t in 0:(NUM_TILES - 1)
            I = (gi - 1) * TILE_DIM + i
            J = (gj - 1) * TILE_DIM + j
            K = t * TILE_DIM + j

            if I <= N && K <= M
                @inbounds tile1[i, j] = trans in ('N', 'n') ? A[I, K] : A[K, I]
            else
                @inbounds tile1[i, j] = zero(eltype(C))
            end
            

            K = t * TILE_DIM + i
            if K <= M && J <= N
                @inbounds tile2[i, j] = trans in ('N', 'n') ? A[J, K] : A[K, J]
            else
                @inbounds tile2[i, j] = zero(eltype(C))
            end
            

            @synchronize

            if I <= N && J <= N
                tmp = zero(eltype(C))
                @simd for k in 1:TILE_DIM
                    @inbounds tmp += tile1[i, k] * tile2[k, j]
                end
                outval[1] += tmp
            end

            @synchronize
        end

        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j

        if I <= N && J <= N && I >= J
            @inbounds C[I, J] = alpha * outval[1] + beta * C[I, J]
        end
    end
end

# wrapper function for the GEMM_ADD kernel
function SYRK_KERNEL!(uplo::Char, trans::Char, alpha::Number, A::AbstractArray, beta::Number, C::AbstractArray)
    backend = get_backend(A)
    N = size(C, 1)
    grid_range = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, N / TILE_DIM) * TILE_DIM)
    syrk_kernel!(backend, (TILE_DIM, TILE_DIM))(
        uplo, trans, alpha, A, beta, C,
        ndrange = grid_range
    )
end

# function GEMM_SUB!(A, B, C)
#     backend = get_backend(A)
#     TILE_DIM = 32
#     N, R, M = size(A, 1), size(C, 1), size(A, 2)
#     matmul!(backend, (TILE_DIM, TILE_DIM))(A, B, C, N, R, M, -1.0, ndrange = (ceil(Int, N / TILE_DIM) * TILE_DIM, ceil(Int, M / TILE_DIM) * TILE_DIM))
# end