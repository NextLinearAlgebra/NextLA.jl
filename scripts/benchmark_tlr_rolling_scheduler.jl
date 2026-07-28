using CUDA
using LinearAlgebra
using Random
using Statistics
using NextLA

const TLRM = NextLA.TLRmodule

function graded_tlr(::Type{T}, n, b, rmax; seed) where {T}
    rng = MersenneTwister(seed)
    X = NextLA.PaddedFTLRMatrix(CUDA.zeros(T, n, n), b, rmax)
    U = randn(rng, T, size(X.int_U))
    V = randn(rng, T, size(X.int_V))
    ntile = length(X.ranks)
    ranks = [1 + mod(3k + seed, rmax) for k in 1:ntile]
    for k in 1:ntile
        U[:, (ranks[k] + 1):end, k] .= 0
        V[:, (ranks[k] + 1):end, k] .= 0
    end
    copyto!(X.int_U, U)
    copyto!(X.int_V, V)
    X.ranks .= ranks
    return X
end

function summarize(label, ws, stats, elapsed)
    active = stats.active_counts
    waves = stats.retirement_waves
    println((
        label=label,
        capacity=ws.key.capacity,
        elapsed_ms=elapsed / 1e6,
        passes=stats.passes,
        admissions=stats.admissions,
        mean_active=isempty(active) ? 0.0 : mean(active),
        minimum_active=isempty(active) ? 0 : Base.minimum(active),
        mean_retirement_wave=isempty(waves) ? 0.0 : mean(waves),
        padded_projection_columns=stats.padded_projection_columns,
        discarded_terminal_columns=stats.discarded_terminal_columns,
        contraction_ms=stats.sampling_ns / 1e6,
        orthogonalization_ms=stats.orthogonalization_ns / 1e6,
        finalization_ms=stats.finalization_ns / 1e6,
        admission_ms=stats.admission_ns / 1e6,
    ))
end

T = Float64
q = parse(Int, get(ENV, "NEXTLA_TLR_Q", "12"))
b = parse(Int, get(ENV, "NEXTLA_TLR_B", "64"))
rA = parse(Int, get(ENV, "NEXTLA_TLR_RA", "16"))
rB = parse(Int, get(ENV, "NEXTLA_TLR_RB", "16"))
rC = parse(Int, get(ENV, "NEXTLA_TLR_RC", "48"))
block = parse(Int, get(ENV, "NEXTLA_TLR_BLOCK", "8"))
n = q * b

A = graded_tlr(T, n, b, rA; seed=41)
B = graded_tlr(T, n, b, rB; seed=73)
C = NextLA.PaddedFTLRMatrix(CUDA.zeros(T, n, n), b, rC)
minimum = NextLA.tlr_gemm_minimum_workspace_bytes(C, A, B; block)
maximum = NextLA.tlr_gemm_maximum_workspace_bytes(C, A, B; block)

for (label, bytes) in (
        ("minimum", minimum),
        ("midpoint", minimum + (maximum - minimum) ÷ 2),
        ("maximum", maximum),
    )
    ws = NextLA.TLRGemmWorkspace(C, A, B; bytes, block)
    # Warm all kernels and backend solver paths before recording scheduler time.
    TLRM.gemm!(
        C, A, B; tol=1e-6, rel=true, eps_rel=1e-6,
        r_required=4, block, workspace=ws)
    TLRM._tlr_gemm_schedule_stats!(
        C, A, B; tol=1e-6, rel=true, eps_rel=1e-6,
        r_required=4, block, workspace=ws)
    CUDA.synchronize()
    t0 = time_ns()
    stats = TLRM._tlr_gemm_schedule_stats!(
        C, A, B; tol=1e-6, rel=true, eps_rel=1e-6,
        r_required=4, block, workspace=ws)
    CUDA.synchronize()
    summarize(label, ws, stats, time_ns() - t0)
end
