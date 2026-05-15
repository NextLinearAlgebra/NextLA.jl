#!/usr/bin/env julia
# Burner: run NextLA.geqrf_2p5d! on LAPACK DCHKQR-style *parameters* (DLATB4 path "QR", IMAT 1:8).
#
# NOTE: Reference `DLATMS` lives in LAPACK TESTING/MATGEN and is usually *not* linked into
# libblastrampoline. This script reproduces:
#   - DLATM1 MODE=3 singular values + DMAX scaling (same as DLATMS uses for DCHKQR),
#   - A = U Σ Vᵀ with random orthogonal U,V (DLATMS "method A" dense core),
#   - zeroing outside the (KL,KU) band using the same envelope as DLATMS post-processing
#     (for IMAT 2–3 this can differ slightly from LAPACK's Householder bandwidth reduction).
#
# Run:
#   julia --project=NextLA.jl scripts/burner_geqrf_2p5d_lapack_matrices.jl

using LinearAlgebra
using Random

const NEXTLA_ROOT = joinpath(@__DIR__, "..", "NextLA.jl")
if !isdir(NEXTLA_ROOT)
    error("Expected NextLA.jl at $NEXTLA_ROOT")
end
push!(LOAD_PATH, NEXTLA_ROOT)
using NextLA

function dlamch_like()
    tenth = 0.1
    epsm = eps(Float64)
    badc2 = tenth / epsm
    badc1 = sqrt(badc2)
    one = 1.0
    sml = floatmin(Float64)
    large = one / sml
    shrink = 0.25
    small = shrink * (sml / epsm)
    return (; epsm, badc1, badc2, small, large)
end

"""Mirror DLATB4 PATH='DQR' (characters 2:3 = 'QR') for IMAT = 1:8."""
function dlatb4_dqr_params(imat::Int, m::Int, n::Int)
    (1 ≤ imat ≤ 8) || throw(ArgumentError("imat must be 1..8, got $imat"))
    two = 2.0
    one = 1.0
    (; badc1, badc2, small, large) = dlamch_like()

    dist = 'S'
    mode = 3
    typ = 'N'

    kl, ku = if imat == 1
        (0, 0)
    elseif imat == 2
        (0, max(n - 1, 0))
    elseif imat == 3
        (max(m - 1, 0), 0)
    else
        (max(m - 1, 0), max(n - 1, 0))
    end

    cndnum = if imat == 5
        badc1
    elseif imat == 6
        badc2
    else
        two
    end

    anorm = if imat == 7
        small
    elseif imat == 8
        large
    else
        one
    end

    return (; dist, mode, typ, kl, ku, cndnum, anorm)
end

"""DLATM1 MODE=3 (positive), IRSIGN=0: σ₁=dmax, geometric decay to σ_mn ∝ dmax/cond, then scale."""
function dlatm1_mode3!(d::Vector{Float64}, cond::Float64, dmax::Float64)
    mn = length(d)
    mn == 0 && return
    if mn == 1
        d[1] = dmax
        return
    end
    alpha = cond^(-1.0 / (mn - 1))
    d[1] = 1.0
    @inbounds for i in 2:mn
        d[i] = alpha^(i - 1)
    end
    s = maximum(abs, d)
    invs = dmax / s
    @inbounds for i in 1:mn
        d[i] *= invs
    end
    return
end

"""Dense SVD ensemble + band envelope (see file header)."""
function jl_dlatms_like_qr!(
    A::Matrix{Float64},
    m::Int,
    n::Int,
    cond::Float64,
    dmax::Float64,
    kl::Int,
    ku::Int,
    rng::AbstractRNG,
)
    fill!(A, 0.0)
    mn = min(m, n)
    d = Vector{Float64}(undef, mn)
    dlatm1_mode3!(d, cond, dmax)
    U = randn(rng, m, m)
    V = randn(rng, n, n)
    Qu = Matrix(qr(U).Q)
    Qv = Matrix(qr(V).Q)
    @views mul!(A[1:m, 1:n], Qu[:, 1:mn] * Diagonal(d), transpose(Qv[:, 1:mn]))
    @inbounds for j in 1:n, i in 1:m
        if j > i + ku || i > j + kl
            A[i, j] = 0.0
        end
    end
    return A
end

"""LAPACK DQRT02-style thin Q (m×k), k=min(m,n); assumes m ≥ n."""
function dqrt02_style_ratios(A_orig::Matrix{Float64}, A_fact::Matrix{Float64}, R_acc::Matrix{Float64})
    m, n = size(A_fact)
    m >= n || error("burner assumes m >= n")
    k = min(m, n)
    Q = @view A_fact[1:m, 1:k]
    R = copy(@view R_acc[1:k, 1:n])
    ΔR = R .- Q' * A_orig[1:m, 1:n]
    anorm = opnorm(A_orig[1:m, 1:n], 1)
    resid_r = opnorm(ΔR, 1)
    epsm = eps(Float64)
    result1 = anorm > 0 ? (resid_r / max(1, m)) / anorm / epsm : 0.0

    G = Matrix{Float64}(I, k, k)
    BLAS.gemm!('T', 'N', -1.0, Q, Q, 1.0, G)
    resid_o = opnorm(Symmetric(G, :U), 1)
    result2 = (resid_o / max(1, m)) / epsm
    return (; result1, result2, anorm, resid_r, resid_o)
end

function one_case(m::Int, n::Int, imat::Int, rng::AbstractRNG; b::Int = 8)
    p = dlatb4_dqr_params(imat, m, n)
    A = zeros(Float64, m, n)
    jl_dlatms_like_qr!(A, m, n, p.cndnum, p.anorm, p.kl, p.ku, rng)

    A_orig = copy(A)
    R_acc = zeros(Float64, n, n)
    tau = zeros(Float64, n)
    NextLA.geqrf_2p5d!(m, n, A, R_acc, tau; b = b)
    met = dqrt02_style_ratios(A_orig, A, R_acc)
    return (; imat, p..., met..., condA = cond(A_orig))
end

function main()
    sizes = [(32, 24), (48, 32), (64, 40)]
    println("=== burner: geqrf_2p5d! on LAPACK DQR-style (DLATB4 + DLATM1-mode3 SVD) matrices ===")
    println("  imat | res_ratio (DQRT02 RESULT(1) style) | orth_ratio (RESULT(2)) | cond(A)")
    println("  Ratios are O(1) for reference Householder+ORGQR on these ensembles; large ⇒ gap.\n")

    for (m, n) in sizes
        println("--- (m,n) = ($m,$n) ---")
        for imat in 1:8
            rng = MersenneTwister(UInt64(hash((m, n, imat, 19881123))))
            r = try
                one_case(m, n, imat, rng; b = 8)
            catch e
                println("  imat=$imat  FAILED ($(typeof(e))) — $(sprint(showerror, e))")
                continue
            end
            println(
                "  imat=$(r.imat)  res_ratio=$(r.result1)  orth_ratio=$(r.result2)  cond(A)=$(r.condA)",
            )
        end
        println()
    end
    println("Done.")
end

main()
