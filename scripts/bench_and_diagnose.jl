#!/usr/bin/env julia
# bench_and_diagnose.jl
# Phase 1: Baseline benchmark on 1000×1000
# Phase 2: Orthogonality sweep across condition numbers
# Phase 3: Inter-panel Q^H Q off-diagonal measurement (root-cause)
#
# Run: julia --project=NextLA.jl -t 1 scripts/bench_and_diagnose.jl

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using NextLA
using KernelAbstractions

# ── helpers ──────────────────────────────────────────────────────────────────

function make_matrix(::Type{T}, m, n; cnd, seed=42) where {T}
    rng = MersenneTwister(seed)
    k = min(m, n)
    sv = T[cnd^(-(i-1)/(k-1)) for i in 1:k]
    U, _ = qr(randn(rng, T, m, m)); U = Matrix(U)
    V, _ = qr(randn(rng, T, n, n)); V = Matrix(V)
    return U[:,1:k] * Diagonal(sv) * V[:,1:k]'
end

function fukaya_metrics(A0, A_fact, R)
    n = size(A_fact, 2)
    Q = A_fact[:, 1:n]
    G = Q'Q - I
    denom = max(norm(A0), eps(eltype(real(A0))))
    res_fro  = norm(A0 - Q*R)  / denom
    orth_fro = norm(G)
    orth2    = opnorm(G, 2)
    # Off-diagonal inter-panel orthogonality (strip the block-diagonal)
    Goff = copy(G)
    for i in axes(Goff, 1); Goff[i,i] = 0; end
    orth_off = norm(Goff)
    return (; res_fro, orth_fro, orth2, orth_off)
end

# Run geqrf_2p5d! with explicit c=1 and user-specified b, bypassing auto-param.
function run_qr(A0::Matrix{T}; b::Int) where T
    m, n = size(A0)
    A = copy(A0)
    R = zeros(T, n, n)
    tau = zeros(T, n)
    be = KernelAbstractions.CPU()
    p  = compute_params(be, T, n; b=b, c=1)
    # Clamp b to what compute_params allows on this hardware
    buse = p.b
    NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p)
    return A, R, buse
end

# ── Phase 1: Baseline benchmark on 1000×1000 ─────────────────────────────────
println("\n══════════════════════════════════════════")
println("Phase 1 — Baseline timing, Float64 1000×1000")
println("══════════════════════════════════════════")

let T = Float64, m = 1000, n = 1000
    A0 = make_matrix(T, m, n; cnd=10.0, seed=7)
    # warm-up
    A_w, R_w, buse = run_qr(A0; b=64)
    println("  Panel width used: $buse")
    met = fukaya_metrics(A0, A_w, R_w)
    @printf("  [warm-up] res_fro=%.2e  orth_fro=%.2e  orth2=%.2e\n",
            met.res_fro, met.orth_fro, met.orth2)

    # timed runs
    times = Float64[]
    for _ in 1:10
        A = copy(A0); R = zeros(T, n, n); tau = zeros(T, n)
        be = KernelAbstractions.CPU()
        p  = compute_params(be, T, n; b=buse, c=1)
        t  = @elapsed NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p)
        push!(times, t)
    end
    sort!(times)
    @printf("  Timing (10 runs): min=%.2fms  median=%.2fms  max=%.2fms\n",
            1e3*times[1], 1e3*times[6], 1e3*times[end])
    println("  Target: 1.5 ms")
end

# ── Phase 2: Orthogonality vs condition number ───────────────────────────────
println("\n══════════════════════════════════════════")
println("Phase 2 — Orthogonality sweep, Float64 128×64 b=16")
println("══════════════════════════════════════════")
println("  cnd          res_fro     orth_fro    orth2       orth_off    PASS?")

let T = Float64, m = 128, n = 64, b = 16
    u = eps(T)
    C_res = 8; C_orth_f = 48; C_orth_2 = 24
    for log10k in [0, 2, 4, 6, 8, 10, 12, 14, 15]
        cnd = 10.0^log10k
        cnd > 1/u && (cnd = 0.9/u)
        A0 = make_matrix(T, m, n; cnd=cnd, seed=99)
        A_f, R_f, buse = run_qr(A0; b=b)
        met = fukaya_metrics(A0, A_f, R_f)
        pass = met.res_fro < C_res*u && met.orth_fro < C_orth_f*u && met.orth2 < C_orth_2*u
        @printf("  κ=10^%-5d  %.2e  %.2e  %.2e  %.2e  %s\n",
                log10k, met.res_fro, met.orth_fro, met.orth2, met.orth_off,
                pass ? "OK" : "FAIL")
    end
end

# ── Phase 3: Root-cause — per-step inter-panel orthogonality ─────────────────
println("\n══════════════════════════════════════════")
println("Phase 3 — Root-cause: inter-panel ‖Q_j^H Q_k‖_F per step")
println("══════════════════════════════════════════")

function diagnose_interpanel(::Type{T}, m, n, b; cnd) where T
    A0 = make_matrix(T, m, n; cnd=cnd, seed=99)
    be = KernelAbstractions.CPU()
    p  = compute_params(be, T, n; b=b, c=1)
    buse = p.b

    A      = copy(A0)
    R      = zeros(T, n, n)
    tau    = zeros(T, n)

    k_eff  = min(m, n)
    panels = UnitRange{Int}[]
    ks     = 1
    while ks <= k_eff
        push!(panels, ks:min(ks+buse-1, k_eff))
        ks += buse
    end

    # Run full geqrf, then inspect Q columns
    NextLA.geqrf_2p5d!(m, n, A, R, tau; params=p)
    Q = A[:, 1:n]

    np = length(panels)
    if np < 2
        println("  Only 1 panel, no inter-panel check possible.")
        return
    end

    println("  κ=$cnd, b=$buse, panels=$np")
    println("  j  k  ‖Q_j^H Q_k‖_F     ‖Q_k^H Q_k - I‖_F")
    for j in 1:np
        rj = panels[j]
        Qj = Q[:, rj]
        # Self-orthogonality
        self_err = norm(Qj'Qj - I(length(rj)))
        for k in (j+1):np
            rk = panels[k]
            Qk = Q[:, rk]
            off = norm(Qj' * Qk)
            @printf("  %d  %d  %.3e           %.3e\n", j, k, off, self_err)
        end
    end

    # Also measure trailing residual drift after each step
    println("\n  Trailing residual drift ‖A_trailing_actual - A_trailing_exact‖_F / ‖A0‖_F:")
    A_cur = copy(A0)
    A_ref = copy(A0)   # exact Schur complement maintained via LAPACK QR
    for (pidx, rng) in enumerate(panels[1:end-1])
        # Exact orthonormal basis for this column range (via LAPACK)
        Q_ref, _ = qr(A_ref[:, rng])
        Qex = Matrix(Q_ref)[:, 1:length(rng)]
        # Trailing part of exact Schur
        tr_rng = (rng.stop+1):n
        W_ref = Qex' * A_ref[:, tr_rng]
        A_ref[:, tr_rng] .-= Qex * W_ref

        # geqrf result for this panel
        Qk    = Q[:, rng]
        W_act = Qk' * A0[:, tr_rng]
        A_cur[:, tr_rng] .-= Qk * W_act

        drift = norm(A_cur[:, tr_rng] - A_ref[:, tr_rng]) / norm(A0)
        @printf("  After panel %d: drift = %.3e\n", pidx, drift)
    end
end

println("\n  Well-conditioned (κ=10²):")
diagnose_interpanel(Float64, 128, 64, 16; cnd=1e2)
println("\n  Ill-conditioned (κ=10¹⁰):")
diagnose_interpanel(Float64, 128, 64, 16; cnd=1e10)
println("\n  Near-singular (κ=10¹⁴):")
diagnose_interpanel(Float64, 128, 64, 16; cnd=1e14)

println("\nDone.")
