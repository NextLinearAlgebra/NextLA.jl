# CPU-only roofline model for the Stage-1/2 fusion/layout trade-off.
#
#   julia --project=experiments experiments/stage2_layout_tradeoff.jl
#   LAYOUT_Q=32 LAYOUT_ROWS=1,4,8,32 julia --project=experiments experiments/stage2_layout_tradeoff.jl
#
# This intentionally models bytes pessimistically (each GEMM operand is charged
# once) and copy traffic explicitly.  It is a filter for deciding which GPU
# prototype is worth writing, not a replacement for measurement.

using Printf, Random, Statistics

const Q = parse(Int, get(ENV, "LAYOUT_Q", "32"))
const BM = parse(Int, get(ENV, "LAYOUT_BM", "256"))
const BN = parse(Int, get(ENV, "LAYOUT_BN", string(BM)))
const RMAX = parse(Int, get(ENV, "LAYOUT_RMAX", "64"))
const EBYTES = parse(Int, get(ENV, "LAYOUT_ELEMENT_BYTES", "2"))
const PEAK = parse(Float64, get(ENV, "LAYOUT_PEAK_TFLOPS", "989")) * 1e12
const BW = parse(Float64, get(ENV, "LAYOUT_BW_TBPS", "3.35")) * 1e12
const ROWS = parse.(Int, split(get(ENV, "LAYOUT_ROWS", "1,4,8,32"), ','))

q8(r) = 8cld(r, 8)

function ranks(q, rmax, dist, seed)
    rng = MersenneTwister(seed)
    raw = if dist == :uniform
        rand(rng, 1:rmax, q, q)
    elseif dist == :skewed
        [clamp(round(Int, 1 + (rmax - 1) * rand(rng)^2), 1, rmax) for _ in 1:q, _ in 1:q]
    elseif dist == :decay
        [clamp(round(Int, rmax * exp(-2.5abs(i - j) / q)) + 1, 1, rmax) for i in 1:q, j in 1:q]
    else
        error("unknown distribution $dist")
    end
    return q8.(raw)
end

roof(flops, bytes) = max(flops / PEAK, bytes / BW)
copytime(bytes) = bytes / BW
ai(flops, bytes) = bytes == 0 ? Inf : flops / bytes

"Return current and proposed FoldRight costs for the contiguous rows `is`."
function foldright(A, B, is)
    q = size(A, 2)
    sigma = [sum(@view B[k, :]) for k in 1:q]
    asum = [sum(A[i, k] for i in is) for k in 1:q]

    # Stage 1 today: one (i,k) GEMM, fused only across output columns j.
    f1 = 2sum(A[i, k] * BM * sigma[k] for i in is, k in 1:q)
    b1 = EBYTES * sum(A[i, k] * BM + BM * sigma[k] + A[i, k] * sigma[k]
                      for i in is, k in 1:q)

    # Proposed coupled Stage 1: one GEMM per k, fused over both i and j.
    # With V column-packed and W row-packed this directly emits the S matrix
    # needed by fixed-(k,j), i-fused Stage 2; no S permutation is required.
    b1f = EBYTES * sum(asum[k] * BM + BM * sigma[k] + asum[k] * sigma[k] for k in 1:q)

    f2 = 2sum(A[i, k] * B[k, j] * BN for i in is, k in 1:q, j in 1:q)
    b2 = EBYTES * sum(A[i, k] * B[k, j] + B[k, j] * BN + A[i, k] * BN
                      for i in is, k in 1:q, j in 1:q)
    b2f = EBYTES * sum(asum[k] * B[k, j] + B[k, j] * BN + asum[k] * BN
                       for k in 1:q, j in 1:q)

    # Proposed Stage 2 is (k,j,i)-owned; current wide Stage 3 is row-owned.
    telems = sum(A[i, k] * BN for i in is, k in 1:q, _ in 1:q)
    tcopy = 2EBYTES * telems

    # Changing Stage 2 alone also needs S: (i,k,j) -> (k,j,i).
    selems = sum(A[i, k] * B[k, j] for i in is, k in 1:q, j in 1:q)
    scopy = 2EBYTES * selems

    # Stage 3 is identical after T is restored to row-owned layout.
    f3 = 2sum(BM * sum(@view A[i, :]) * (q * BN) for i in is)
    b3 = EBYTES * sum(BM * sum(@view A[i, :]) + sum(@view A[i, :]) * q * BN + BM * q * BN for i in is)

    current = roof(f1, b1) + roof(f2, b2) + roof(f3, b3)
    coupled = roof(f1, b1f) + roof(f2, b2f) + copytime(tcopy) + roof(f3, b3)
    stage2only = roof(f1, b1) + roof(f2, b2f) + copytime(scopy + tcopy) + roof(f3, b3)
    return (; f1, b1, b1f, f2, b2, b2f, tcopy, scopy, current, coupled, stage2only)
end

"FoldLeft mirror: Stage 2 fuses j for fixed (i,k), reusing U (not Z)."
function foldleft(A, B, is)
    q = size(A, 2)
    sigma = [sum(@view B[k, :]) for k in 1:q]
    f2 = 2sum(BM * A[i, k] * B[k, j] for i in is, k in 1:q, j in 1:q)
    b2 = EBYTES * sum(BM * A[i, k] + A[i, k] * B[k, j] + BM * B[k, j]
                      for i in is, k in 1:q, j in 1:q)
    b2f = EBYTES * sum(BM * A[i, k] + A[i, k] * sigma[k] + BM * sigma[k]
                       for i in is, k in 1:q)
    # Output changes from k-major panels to the j-major k-stack consumed by
    # FoldLeft Stage 3, so T is read and written once.
    telems = sum(BM * B[k, j] for _ in is, k in 1:q, j in 1:q)
    tcopy = 2EBYTES * telems
    return (; f2, b2, b2f, tcopy,
            current=roof(f2, b2), fused=roof(f2, b2f) + copytime(tcopy))
end

println("Stage-1/2 layout roofline: Q=$Q BM=$BM BN=$BN RMAX=$RMAX bytes/el=$EBYTES")
println("peak=$(PEAK / 1e12) TF/s, bandwidth=$(BW / 1e12) TB/s; times include Stage 3 for FoldRight")
@printf("%-8s %4s | %8s %8s | %8s %8s | %9s | %8s %8s\n",
        "dist", "rows", "AI1 old", "AI1 new", "AI2 old", "AI2 new", "Tcopy MB", "S2-only", "coupled")

for dist in (:uniform, :skewed, :decay)
    A, B = ranks(Q, RMAX, dist, 7), ranks(Q, RMAX, dist, 11)
    for nr in ROWS
        nr <= Q || continue
        rs = foldright(A, B, 1:nr)
        @printf("%-8s %4d | %8.1f %8.1f | %8.1f %8.1f | %9.2f | %7.3fx %7.3fx\n",
                string(dist), nr, ai(rs.f1, rs.b1), ai(rs.f1, rs.b1f),
                ai(rs.f2, rs.b2), ai(rs.f2, rs.b2f), rs.tcopy / 2.0^20,
                rs.current / rs.stage2only, rs.current / rs.coupled)
    end
end

println("""

Interpretation:
  S2-only  changes only Stage 2 and pays both S and T permutations.
  coupled  also fuses Stage 1 across scheduled rows, emits S in the required
           layout, and pays only the T permutation before today's Stage 3.
  Values above 1 predict a win. Confirm any promising case on the GPU.

FoldLeft sanity summary (uniform ranks):
""")
A, B = ranks(Q, RMAX, :uniform, 7), ranks(Q, RMAX, :uniform, 11)
for nr in ROWS
    nr <= Q || continue
    x = foldleft(A, B, 1:nr)
    @printf("  rows=%2d  Stage2 AI %.1f -> %.1f, Tcopy %.2f MiB, predicted Stage2+copy speedup %.3fx\n",
            nr, ai(x.f2, x.b2), ai(x.f2, x.b2f), x.tcopy / 2.0^20, x.current / x.fused)
end

