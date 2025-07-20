using NextLA
using Test
using LinearAlgebra, Random
using CUDA


function parameter_creation(path::String, imat::Int, M::Int, N::Int)
    # Defaults
    kl = 0
    ku = 0
    anorm = 1.0
    mode = :decay
    cndnum = 10.0
    dist = :uniform
    type = 'N'

    if path == "GE"  # General dense
        type = 'N'
        if imat == 1
            mode = :decay; cndnum = 2.0
        elseif imat == 2
            mode = :decay; cndnum = 1e2
        elseif imat == 3
            mode = :one_large; cndnum = 1e6
        elseif imat == 4
            kl = 2; ku = 2
            mode = :decay; cndnum = 1e2
        end
        # TODO: expand for future tests
    end

    return (; type, kl, ku, anorm, mode, cndnum, dist)
end

function matrix_generation(type, M, N; dist=:uniform, mode=:decay, 
                                     cndnum=10.0, anorm=1.0, 
                                     kl=0, ku=0, seed=1234)

    rng = MersenneTwister(seed)

    # Handle edge cases
    if M == 0 || N == 0
        return zeros(type, M, N)
    end

    # Step 1: Generate singular values (σ_i)
    k = min(M, N)
    σ = zeros(real(type), k)
    
    if mode == :one_large
        σ .= 1.0
        if k > 0
            σ[1] = cndnum
        end
    elseif mode == :decay
        if k == 1
            σ[1] = anorm
        else
            for i in 1:k
                σ[i] = anorm * (1/cndnum)^((i - 1)/(k - 1))
            end
        end
    elseif mode == :random
        σ .= rand(rng, real(type), k)
        if maximum(σ) > 0
            σ .*= anorm / maximum(σ)
        end
    else
        error("Unsupported mode")
    end

    U, _ = qr(randn(rng, type, M, M))
    V, _ = qr(randn(rng, type, N, N))

    # Step 3: Build A = U * Σ * V'
    Σ = Diagonal(σ)
    A = U[:, 1:k] * Σ * V[:, 1:k]'

    # Step 4: Impose bandwidth (optional)
    if kl > 0 || ku > 0
        for i in 1:M, j in 1:N
            if j > i + ku || i > j + kl
                A[i, j] = zero(type)
            end
        end
    end

    return A
end

include("NextLAMatrix.jl")
include("lu.jl")
include("unified_rectrxm.jl")
include("trsm.jl")
include("lauum.jl")

include("axpy.jl")
include("gerc.jl")
include("zlarfg.jl")
include("zlarf.jl")
#include("zlarft.jl") #TODO: implement sub-tests for zlarft  (indirect test in zgeqrt)
#include("zlarfb.jl") #TODO: implement sub-tests for zlarfb (indirect test in zunmqr)
include("zgeqr2.jl")
include("zgeqrt.jl")
include("zunmqr.jl")
include("ztsqrt.jl")
include("ztsmqr.jl")
include("zparfb.jl")
#include("zpamm.jl") #TODO: implement sub-tests for zpamm (indirect test in zparfb)
#include("zpemv.jl") #TODO: implement sub-tests for zpemv (indirect test in zttqrt)
include("zttqrt.jl")
include("zttmqr.jl")

