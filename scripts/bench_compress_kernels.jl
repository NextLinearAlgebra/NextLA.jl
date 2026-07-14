# Microbenchmark for the compression helper kernels:
#   _tile_norm_sq_kernel!   (per-tile squared Frobenius norm)
#   _cholqr_shift_kernel!   (per-slab FKNYY diagonal shift)
#   _prune_rank_kernel  (rank detection + in-place compaction)
#
# Run:
#   julia --project=../gpuenv bench_compress_kernels.jl

using CUDA, KernelAbstractions, Random, Printf, Statistics
CUDA.functional() || error("CUDA required")

using NextLA
const TLR = NextLA.TLRmodule
using NextLA: SUBGROUP_SIZE, unwrap

const BK = CUDABackend()

function bench(f; warmup=5, iters=50)
    for _ in 1:warmup; f(); end
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:iters
        CUDA.synchronize()
        t = CUDA.@elapsed f()
        push!(ts, t)
    end
    return minimum(ts), median(ts)
end

function bench_with_setup(f, setup; warmup=3, iters=20)
    for _ in 1:warmup
        setup()
        f()
    end
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:iters
        setup()
        CUDA.synchronize()
        push!(ts, CUDA.@elapsed f())
    end
    return minimum(ts), median(ts)
end

# ---- norm kernel (dense source) ----
function bench_norm(n, b, S)
    A = CUDA.randn(Float32, n, n)
    mt = cld(n, b)
    coords = [(i, j) for j in 1:mt, i in 1:mt if i != j]
    count = length(coords)
    p0s = CuArray(Int32[(c[1]-1)*b + 1 for c in coords])
    q0s = CuArray(Int32[(c[2]-1)*b + 1 for c in coords])
    out = CUDA.zeros(Float64, count)
    W, _, NT = TLR._norm_launch(BK, b)
    run() = TLR._tile_norm_sq_kernel!(BK, NT)(out, A, p0s, q0s, b, b, Val{W}(), Val{NT}();
        ndrange=(NT*count,), workgroupsize=NT)
    run(); CUDA.synchronize()
    # correctness vs CPU reference
    Ac = Array(A); ref = [sum(abs2 ∘ Float64, view(Ac, (c[1]-1)*b+1:(c[1]-1)*b+b, (c[2]-1)*b+1:(c[2]-1)*b+b)) for c in coords]
    err = maximum(abs.(Array(out) .- ref) ./ ref)
    mn, md = bench(run)
    bytes = count * b * b * sizeof(Float32)
    return (; count, mn, md, gbps = bytes/mn/1e9, err)
end

# ---- shift kernel ----
function bench_shift(S, count)
    G0 = CUDA.rand(Float64, S, S, count)  # positive diagonals
    mult = CUDA.ones(Float64, count)
    coeff = TLR._cholqr_shift_coeff(Float64, 512, S)
    nt = TLR._reduce_threads(S)
    G = copy(G0)
    run() = TLR._cholqr_shift_kernel!(BK, nt)(G, coeff, mult, Val{nt}();
        ndrange=(nt*count,), workgroupsize=nt)
    copyto!(G, G0); run(); CUDA.synchronize()
    # correctness: shift added = coeff*max(diag) to each diagonal entry
    Gc = Array(G); G0c = Array(G0)
    err = 0.0
    for k in 1:count
        mx = maximum(real(G0c[i,i,k]) for i in 1:S)
        want = coeff*mx
        got = maximum(Gc[i,i,k] - G0c[i,i,k] for i in 1:S)
        err = max(err, abs(got-want)/want)
    end
    # time the kernel alone (relaunching on the same G just keeps adding shift;
    # irrelevant for timing, and matches how the baseline was measured)
    mn, md = bench(run)
    return (; mn, md, err)
end

# ---- fused rank detection + in-place compaction ----
function bench_truncate(tm, tn, S, count, keep, nsub)
    rng = MersenneTwister(1000 + tm + tn + S + count + keep)
    U0 = CuArray(randn(rng, Float32, tm, S, count))
    Vcpu = randn(rng, Float32, tn, S, count)
    # Separate the energies so the hard cap has deterministic choices and a
    # representative mix of holes throughout the first K positions.
    for j in 1:S
        Vcpu[:, j, :] .*= Float32(0.25 + j / S)
    end
    V0 = CuArray(Vcpu)
    U = similar(U0)
    V = similar(V0)
    rk = CUDA.zeros(Int32, count)
    norm0 = CuArray([sum(abs2, Float64.(view(Vcpu, :, :, k))) for k in 1:count])
    norm_err_sq = similar(norm0)

    W = unwrap(SUBGROUP_SIZE(typeof(BK)))
    nt = W * nsub
    kernel! = TLR._prune_rank_kernel(BK, nt)
    run() = kernel!(U, V, rk, norm_err_sq, 0.0, false, keep, 0.0,
        Val{S}(), Val{W}(); ndrange=(nt * count,), workgroupsize=nt)
    setup() = (copyto!(U, U0); copyto!(V, V0); copyto!(norm_err_sq, norm0); nothing)

    setup()
    run()
    CUDA.synchronize()
    ranks_ok = all(==(keep), Array(rk))
    tail_ok = all(iszero, Array(view(U, :, keep+1:S, :))) &&
        all(iszero, Array(view(V, :, keep+1:S, :)))
    mn, md = bench_with_setup(run, setup)
    bytes_moved = count * (tm + tn) * S * sizeof(Float32)
    return (; mn, md, gbps=bytes_moved / mn / 1e9, ok=ranks_ok && tail_ok)
end

println("GPU = ", CUDA.name(CUDA.device()))
println("W (subgroup) = ", unwrap(SUBGROUP_SIZE(typeof(BK))))
println()

cases = [(4096, 512, 128), (4096, 256, 128), (8192, 512, 128), (2048, 128, 64)]
@printf("%-22s %10s %10s %9s %10s\n", "norm  (n,b,S)", "min_us", "med_us", "GB/s", "relerr")
for (n, b, S) in cases
    r = bench_norm(n, b, S)
    @printf("%-22s %10.2f %10.2f %9.1f %10.1e   (ntiles=%d)\n",
        "($n,$b,$S)", r.mn*1e6, r.md*1e6, r.gbps, r.err, r.count)
end
println()
@printf("%-22s %10s %10s %10s\n", "shift (S,count)", "min_us", "med_us", "relerr")
for (S, count) in [(128, 56), (128, 240), (64, 992), (128, 56*4)]
    r = bench_shift(S, count)
    @printf("%-22s %10.2f %10.2f %10.1e\n", "($S,$count)", r.mn*1e6, r.md*1e6, r.err)
end

println()
@printf("%-28s %10s %10s %9s %5s\n", "truncate (m,n,S,batch,k,sg)",
    "min_us", "med_us", "GB/s", "ok")
for (tm, tn, S, count, keep) in [(256, 256, 64, 56, 40),
                                  (256, 256, 64, 240, 24),
                                  (512, 512, 128, 56, 80)]
    for nsub in (1, 2, 4, 8)
        r = bench_truncate(tm, tn, S, count, keep, nsub)
        label = "($tm,$tn,$S,$count,$keep,$nsub)"
        @printf("%-28s %10.2f %10.2f %9.1f %5s\n",
            label, r.mn * 1e6, r.md * 1e6, r.gbps, string(r.ok))
    end
end
