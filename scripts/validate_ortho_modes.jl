#!/usr/bin/env julia
# Validate ortho=:fast and ortho=:safe across condition numbers and matrix sizes.
#
# Run: julia --project=NextLA.jl -t 4 scripts/validate_ortho_modes.jl

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA, KernelAbstractions

function make_matrix(::Type{T}, m, n; cnd, seed=42) where {T}
    rng = MersenneTwister(seed)
    k = min(m, n)
    sv = T[T(cnd)^(-(i-1)/(k-1)) for i in 1:k]
    U, _ = qr(randn(rng, T, m, m)); U = Matrix(U)
    V, _ = qr(randn(rng, T, n, n)); V = Matrix(V)
    return U[:,1:k] * Diagonal(sv) * V[:,1:k]'
end

function metrics(A0, A, R)
    n = size(A, 2)
    Q = A[:, 1:n]
    G = Q' * Q - I
    denom = max(norm(A0), eps(real(eltype(A0))))
    return (
        res = norm(A0 - Q * R) / denom,
        orth = norm(G),
        inter_panel = norm(tril(Q' * Q, -1)),
    )
end

function check(::Type{T}, m, n, b, kappa, ortho::Symbol; passΓ=true) where {T}
    A0 = make_matrix(T, m, n; cnd=kappa, seed=99)
    A = copy(A0); R = zeros(T, n, n); tau = zeros(T, n)
    be = CPU()
    p = compute_params(be, T, n; b=b, c=1)
    NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p, ortho=ortho)
    met = metrics(A0, A, R)
    u = eps(real(T))
    # Per-panel Fukaya bound (κ-independent)
    fukaya = 50 * (m * b + b * (b + 1)) * u
    # Inter-panel: :safe → O(u), :fast → O(κ·u)
    inter_bound = ortho === :safe ? fukaya : 50 * kappa * u
    res_bound = 100 * u
    passed = met.res < res_bound && met.orth < (ortho === :safe ? fukaya : max(fukaya, inter_bound))
    pflag = passed ? "OK" : "WARN"
    @printf("  T=%-9s κ=%.0e b=%-3d ortho=%-5s  res=%.2e orth=%.2e  inter=%.2e  %s\n",
            T, kappa, b, ortho, met.res, met.orth, met.inter_panel, pflag)
    return passed
end

println("══════════════════════════════════════════════════════════════")
println(" Ortho mode validation: :fast vs :safe across κ")
println("══════════════════════════════════════════════════════════════")

results = Bool[]

for T in (Float64, Float32, ComplexF64)
    println("\n── $T ─────────────────")
    for kappa in (1.0, 1e2, 1e6, 1e10, 1e14)
        T === Float32 && kappa > 1e6 && continue
        for ortho in (:fast, :safe)
            push!(results, check(T, 128, 64, 16, kappa, ortho))
        end
    end
end

println("\n── Default keyword check ─────────────────")
# Confirm geqrf_2p5d!(A) (no ortho kwarg) uses :fast.
let T = Float64, m = 64, n = 32, b = 8
    A0 = make_matrix(T, m, n; cnd=1e2, seed=11)
    A = copy(A0); R = zeros(T, n, n); tau = zeros(T, n)
    be = CPU(); p = compute_params(be, T, n; b=b, c=1)
    NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p)
    met_default = metrics(A0, A, R)

    A2 = copy(A0); R2 = zeros(T, n, n); tau2 = zeros(T, n)
    NextLA.geqrf_2p5d!(m, n, A2, R2, tau2; params=p, ortho=:fast)
    met_fast = metrics(A0, A2, R2)
    same = isapprox(met_default.res, met_fast.res; rtol=1e-12) &&
           isapprox(met_default.orth, met_fast.orth; rtol=1e-12)
    println("  default ≡ :fast?  ", same ? "yes" : "no",
            "  (res default=$(met_default.res), :fast=$(met_fast.res))")
end

println("\n── Two-arg wrapper check ─────────────────")
let T = Float64, m = 32, n = 24
    A0 = make_matrix(T, m, n; cnd=1e2, seed=7)
    A = copy(A0)
    Aout, R = NextLA.geqrf_2p5d!(A; b=8)
    met = metrics(A0, Aout, R)
    @printf("  geqrf_2p5d!(A; b=8) [default ortho]  res=%.2e orth=%.2e\n", met.res, met.orth)
end

@printf("\nTotal: %d/%d expected behaviors confirmed.\n", sum(results), length(results))
