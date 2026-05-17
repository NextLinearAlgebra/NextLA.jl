export geqrf_2p5d_lookahead!

"""
    geqrf_2p5d_lookahead!(m, n, A, R_acc, tau; params=nothing, b=nothing,
                          ortho::Symbol=:fast, n_streams::Int=2)

Variant 5 — sCQR3-2.5D with **look-ahead pipelining** (Phase Q5 of
[`qr_schur_xpartition.tex`](qr_schur_xpartition.tex), §A.1).

Schedules the panel/trailing-update sequence across `n_streams` CUDA streams
so that panel-(k+1)'s Phase Q1 begins as soon as panel-k's S_4, S_5 have
finished on the *next-panel slice* `A[:, k+b : k+2b-1]`, rather than waiting
for the full trailing update on `A[:, k+b : N]`. The remaining trailing
update on `A[:, k+2b : N]` runs concurrently with panel-(k+1)'s factorization
on a separate stream, then both streams synchronize at the step boundary.

X-partition derivation (Phase Q5a–b of the paper): the trailing update's
two column slabs `A_next = A[:, k+b:k+2b-1]` (width `b`) and
`A_rest = A[:, k+2b:N]` (width `N - k - 2b`) are column-disjoint, so they
carry no data dependency between them. The cDAG partial order
   `S_4(A_next); S_5(A_next)  ≺  Panel-(k+1)  ≺  S_4(slab_{k+2}); S_5(slab_{k+2})`
is enforced by one CUDA event per stream boundary. Each statement's access
function `ϕ_j` is unchanged, so the DAAP classification (Path~(s):
DAAP-proper; Path~(h): Quasi-DAAP) is preserved.

Critical-path speedup vs sequential schedule (Phase Q5 derivation):
   `η_LA = 1 + t_panel / t_tail^rest  ∈  [1, 1 + b/N]`.
For `b = N/√P_1` on a single H200 with `P_1 ≈ 132`, this is asymptotically
`1 + 1/√P_1 ≈ 1.09`. The visible single-GPU benefit is largest in the
small-step regime (`b/N` not vanishing): ~1.3× at N=4000 b=363.

`n_streams ≥ 2` is required for any look-ahead. Deeper look-ahead with
`n_streams ≤ √P_1` overlaps multiple panels but shrinks each stream's
X-partition cube to `√(M/s)`, so the optimum is `n_streams = 2` (paper
§A.1, Phase Q5b).
"""
function geqrf_2p5d_lookahead!(m::Integer, n::Integer,
                                A::AbstractMatrix{T},
                                R_acc::AbstractMatrix{T},
                                tau::AbstractVector{T};
                                params::Union{DeviceParams{T}, Nothing} = nothing,
                                b::Union{Integer, Nothing} = nothing,
                                ortho::Symbol = :fast,
                                n_streams::Int = 2) where {T}
    n_streams >= 1 || throw(ArgumentError("n_streams must be ≥ 1"))
    m = Int(m); n = Int(n)
    (m == 0 || n == 0) && return nothing

    be = KernelAbstractions.get_backend(A)
    k_eff = min(m, n)
    N_budget = max(m, n)

    p = if params === nothing
        bval = b === nothing ? nothing : max(1, Int(b))
        compute_params(be, T, N_budget; b = bval, c = nothing)
    else
        params
    end
    b_full = p.b
    b_full >= 1 || throw(ArgumentError("DeviceParams.b must be ≥ 1"))
    tile = _geqrf_tile(be, b_full)
    c_eff = effective_c(p)

    # Scratch (identical to geqrf_2p5d!).
    G_buf    = similar(A, b_full, b_full)
    R_buf    = similar(A, b_full, b_full)
    info_buf = fill!(similar(A, Int, 1), 0)
    W_buf    = similar(A, b_full, n > b_full ? n - b_full : 1)

    fill!(R_acc, zero(T))

    # When n_streams=1, the look-ahead degenerates to the sequential schedule
    # (geqrf_2p5d!'s body). For n_streams≥2 we use the implementation below.
    _lookahead_run!(be, m, n, A, R_acc, tau, p, tile, c_eff, b_full,
                    G_buf, R_buf, info_buf, W_buf, n_streams, ortho)
    return nothing
end

# Default backend has no notion of "streams" — fall back to sequential.
function _lookahead_run!(be::KernelAbstractions.CPU, m, n, A, R_acc, tau, p, tile,
                          c_eff, b_full, G_buf, R_buf, info_buf, W_buf,
                          n_streams::Int, ortho::Symbol)
    # CPU fallback: just call sequential geqrf_2p5d!.
    geqrf_2p5d!(m, n, A, R_acc, tau; params=p, ortho=ortho)
end

# Backend hook overridden in ext/cudaext.jl for actual stream-based look-ahead.
function _lookahead_run!(be, m, n, A, R_acc, tau, p, tile, c_eff, b_full,
                          G_buf, R_buf, info_buf, W_buf, n_streams::Int, ortho::Symbol)
    # Generic fallback: sequential.
    geqrf_2p5d!(m, n, A, R_acc, tau; params=p, ortho=ortho)
end
