# h100_audit_probe.jl — every open question in one rented-GPU session.
#
#   julia --project=experiments experiments/h100_audit_probe.jl            # all phases
#   PROBE_PHASE=align julia --project=experiments experiments/h100_audit_probe.jl
#
# Phases: align ranks descriptor fold plan mixedout overhead
# Tunables: PROBE_N=8192  PROBE_BM=256  PROBE_RMAX=128  PROBE_PHASE=all
#
# Every phase is wrapped in try/catch: one failure must not waste the rental.

using CUDA, NextLA, Printf, Random

const _T   = NextLA.TLRmodule
const N    = parse(Int, get(ENV, "PROBE_N",    "8192"))
const BM   = parse(Int, get(ENV, "PROBE_BM",   "256"))
const RMAX = parse(Int, get(ENV, "PROBE_RMAX", "128"))
const PHASE = get(ENV, "PROBE_PHASE", "all")

# `align` deliberately provokes ERROR_MISALIGNED_ADDRESS, which is STICKY: it
# kills the CUDA context and every later phase. It is therefore excluded from
# "all" and must be run on its own, one variant per process.
want(p) = PHASE == "all" ? p != "align" : PHASE == p

function timeit(f, reps=10)
    f(); CUDA.synchronize()
    t = Inf
    for _ in 1:reps
        CUDA.synchronize()
        t = min(t, CUDA.@elapsed f())
    end
    t * 1e3
end

function phase(f, name)          # do-block form passes the closure first
    want(name) || return
    println("\n", "="^78, "\n### PHASE $name\n", "="^78); flush(stdout)
    try
        f()
    catch e
        println("!!! PHASE $name FAILED: ", sprint(showerror, e)[1:min(end, 400)])
    end
    CUDA.reclaim(); flush(stdout)
end

# ---------------------------------------------------------------- builders
rankgrid(q, rmax; seed=7, dist=:uniform) = begin
    rng = MersenneTwister(seed)
    if dist === :uniform
        [rand(rng, 1:rmax) for _ in 1:q, _ in 1:q]
    elseif dist === :decay          # heavy-tailed: realistic TLR off-diagonal decay
        [clamp(round(Int, rmax * exp(-2.5 * abs(i - j) / q)) + 1, 1, rmax) for i in 1:q, j in 1:q]
    else                            # single global max rank = literature baseline
        fill(rmax, q, q)
    end
end

function make_ftlr(::Type{S}, n, bm, rg; policy=:q8) where {S}
    A = NextLA.CompressedFTLRMatrix(CUDA.CUDABackend(), S, n, n, (bm, bm), rg;
                                    execution_rank_policy=policy)
    fill!(A.outer.data, S(0.01)); fill!(A.inner.data, S(0.01))
    return A
end

"""Workspace big enough for the whole GEMM (max profile), plus the analysis object."""
function analysed(::Type{S}, n, bm, rg; policy=:q8, rgB=nothing) where {S}
    A = make_ftlr(S, n, bm, rg; policy=policy)
    B = make_ftlr(S, n, bm, rgB === nothing ? rankgrid(size(rg,1), RMAX; seed=11) : rgB;
                  policy=policy)
    C = CUDA.zeros(S, n, n)
    maxb = _T.gemm_maximum_workspace_bytes(A, B)
    ws = NextLA.DenseGemmWorkspace(A, maxb)
    an = NextLA.analyze_compressed_gemm(C, A, B; workspace=ws)
    return A, B, C, ws, an
end

gflop_cc(A, B, rg) = begin      # stage1 + stage2 + stage3 MACs x2, q8 execution ranks
    q = size(rg, 1); bm = BM
    r̂ = [_T._compressed_ftlr_execution_rank(A, i, k) for i in 1:q, k in 1:q]
    ŝ = [_T._compressed_ftlr_execution_rank(B, k, j) for k in 1:q, j in 1:q]
    s1 = sum(r̂[i,k] * sum(ŝ[k,:]) * bm for i in 1:q, k in 1:q)
    s2 = sum(r̂[i,k] * ŝ[k,j] * bm for i in 1:q, k in 1:q, j in 1:q)
    s3 = sum(sum(r̂[i,:]) * bm * (q * bm) for i in 1:q)
    2 * (s1 + s2 + s3) / 1e9
end

println("device=", CUDA.name(CUDA.device()), " cap=", CUDA.capability(CUDA.device()),
        " runtime=", CUDA.runtime_version())
println("N=$N BM=$BM RMAX=$RMAX  grid=", cld(N, BM), "^2")

# ============================================================ 1. ALIGNMENT
# THE open question. All prior numbers used CUBLAS_COMPUTE_32F, which never
# touches tensor cores, so alignment could not have mattered. Prepared
# descriptors isolate kernel time from descriptor construction.
phase("align") do
    # CRITICAL: lda/ldb/ldc are held FIXED across all variants. Allocating a
    # taller store for the offset case would also change the leading dimension,
    # confounding pointer alignment with power-of-two stride aliasing (which is
    # a much larger effect and points the other way).
    PAD = 8
    function build(::Type{S}, nt, m, k, n, offA, offC) where {S}
        keep, tasks = Any[], NextLA.GroupedGemmTask[]
        for _ in 1:nt
            As = CUDA.zeros(S, m + PAD, k); fill!(As, S(0.01))
            Bs = CUDA.zeros(S, k + PAD, n); fill!(Bs, S(0.01))
            Cs = CUDA.zeros(S, m + PAD, n)
            A = view(As, (1+offA):(offA+m), :)
            B = view(Bs, 1:k, :)
            C = view(Cs, (1+offC):(offC+m), :)
            push!(keep, (As, Bs, Cs))
            push!(tasks, NextLA.GroupedGemmTask('N','N', one(S), A, B, zero(S), C))
        end
        tasks, keep
    end
    function run_variant(::Type{S}, mode, nt, m, k, n, offA, offC) where {S}
        tasks, keep = build(S, nt, m, k, n, offA, offC)
        p = NextLA.prepare_precision_gemm_grouped(tasks, mode)
        bk = NextLA.get_backend(tasks[1].C)
        t = timeit(() -> NextLA._with_grouped_host_pointer_mode(bk) do
            NextLA.precision_gemm_grouped_prepared!(p) end)
        NextLA.destroy_prepared_grouped_gemm!(p)
        keep = tasks = nothing; CUDA.reclaim()
        return t
    end
    @printf("%-22s %-4s %10s %12s %12s\n",
            "config", "qT", "all-aligned", "A misaligned", "C misaligned")
    for (label, S, mode) in (("FP16  / compute FP32", Float16, NextLA.GEMMCompute{Float32}()),
                             ("BF16  / compute FP32", Core.BFloat16, NextLA.GEMMCompute{Float32}()),
                             ("FP32  / TF32 tensor",  Float32, NextLA.TF32()),
                             ("FP32  / compute FP32", Float32, NextLA.GEMMCompute{Float32}()),
                             ("FP64  / compute FP64", Float64, NextLA.GEMMCompute{Float64}()))
        try
            nt, m, k, n = 1024, 128, 128, 256
            qT = NextLA.gemm_alignment_quantum(S)
            t0 = run_variant(S, mode, nt, m, k, n, 0, 0)
            ta = run_variant(S, mode, nt, m, k, n, 1, 0)
            tc = run_variant(S, mode, nt, m, k, n, 0, 1)
            @printf("%-22s %-4d %10.3fms %9.3fms(%.2fx) %9.3fms(%.2fx)\n",
                    label, qT, t0, ta, ta/t0, tc, tc/t0)
        catch e
            @printf("%-22s SKIPPED: %s\n", label, sprint(showerror, e)[1:min(end,60)])
        end
    end
    println("\n>> MEASURED ON SM75: FP16 FAULTS (ERROR_MISALIGNED_ADDRESS) at element")
    println(">> offsets 1,2,4 and succeeds at 0,8 — i.e. 16-byte alignment is a HARD")
    println(">> requirement on tensor-core kernels, not a perf hint. FP32/FP64/BF16 did")
    println(">> not fault on SM75, but SM75 has no BF16/TF32/FP64 tensor cores — H100")
    println(">> does, so re-test those here. A fault IS the result; expect this phase")
    println(">> to die, and run each row in its own process.")
end

# ============================================================ 2. RANK POLICY
# Guard is gone, so this is the honest bucketing curve. :maxrank is the
# literature baseline (single global rank), NOT :exact.
phase("ranks") do
    q = cld(N, BM)
    @printf("%-10s %-8s %8s %8s %10s %10s %9s\n",
            "dist", "policy", "groups", "padFLOP", "time(ms)", "GFLOP/s", "vs q8")
    for dist in (:uniform, :decay)
        base = nothing
        # :q8 first so every later row gets a ratio against it.
        for policy in (:q8, :exact, :q16, :pow2, :maxrank)
            try
                rg = policy === :maxrank ? rankgrid(q, RMAX; dist=:max) : rankgrid(q, RMAX; dist=dist)
                pol = policy === :maxrank ? :exact : policy
                A, B, C, ws, an = analysed(Float32, N, BM, rg; policy=pol)
                gf = gflop_cc(A, B, rg)
                t = timeit(() -> _T.gemm!(C, A, B; workspace=ws, alpha=1f0, beta=0f0, analysis=an))
                policy === :q8 && (base = t)
                @printf("%-10s %-8s %8s %8.2f %9.3f %10.1f %8s\n",
                        dist, policy, "-", gf, t, gf/(t/1e3),
                        base === nothing ? "-" : @sprintf("%.2fx", t/base))
                close(an); A = B = C = ws = an = nothing; CUDA.reclaim()
            catch e
                @printf("%-10s %-8s FAILED: %s\n", dist, policy,
                        sprint(showerror, e)[1:min(end,80)])
            end
        end
    end
    println("\n>> compare q8 against :maxrank (the KAUST-style baseline), not :exact.")
    println(">> padFLOP column is the executed work; time/TFLOPs is achieved efficiency.")
end

# ==================================================== 3. DESCRIPTOR REUSE
# Confirmed 70x on a synthetic stage-2. Quantify on the real C x C lowering:
# transient (rebuilds descriptors per call) vs prepared.
phase("descriptor") do
    q = cld(N, BM)
    rg = rankgrid(q, RMAX)
    A, B, C, ws, an = analysed(Float32, N, BM, rg)
    gf = gflop_cc(A, B, rg)
    t_an = timeit(() -> _T.gemm!(C, A, B; workspace=ws, alpha=1f0, beta=0f0, analysis=an))
    t_tr = timeit(() -> _T.gemm!(C, A, B; workspace=ws, alpha=1f0, beta=0f0), 3)
    @printf("  analysis/prepared : %9.3f ms (%8.1f GFLOP/s)\n", t_an, gf/(t_an/1e3))
    @printf("  transient (rebuild): %9.3f ms (%8.1f GFLOP/s)  -> %.1fx slower\n",
            t_tr, gf/(t_tr/1e3), t_tr/t_an)
    close(an)
    println("\n>> if transient >> prepared, route stages.jl/mixed_dense.jl through")
    println(">> reusable slots (machinery already exists in ext/cuda/gemm.jl).")
end

# ==================================================== 4. FOLD-SELECTION MODEL
# _compressed_ftlr_select_fold picks by MAC count and assumes both folds achieve
# the same FLOP/s. Never validated. Force each fold and compare.
phase("fold") do
    q = cld(N, BM)
    rg = rankgrid(q, RMAX)
    rgB = rankgrid(q, RMAX; seed=11)   # distinct grids: identical grids make
    A = make_ftlr(Float32, N, BM, rg)  # right/left FLOP models tie exactly
    B = make_ftlr(Float32, N, BM, rgB)
    C = CUDA.zeros(Float32, N, N)
    LA = NextLA.TLRmodule.logical_operand(A, 'N'); LB = NextLA.TLRmodule.logical_operand(B, 'N')
    plan = _T._compressed_ftlr_rank_plan(LA, LB)
    p = plan.profile
    rf = p.right_flops === nothing ? nothing : sum(p.right_flops)
    lf = p.left_flops  === nothing ? nothing : sum(p.left_flops)
    @printf("  model MACs: right=%.3e left=%.3e  -> model picks :%s (%.2fx)\n",
            something(rf, NaN), something(lf, NaN),
            (rf === nothing || lf === nothing) ? "n/a" : (rf <= lf ? "right" : "left"),
            (rf === nothing || lf === nothing) ? NaN : max(rf,lf)/min(rf,lf))
    maxb = _T.gemm_maximum_workspace_bytes(A, B)
    ws = NextLA.DenseGemmWorkspace(A, maxb)
    arena = NextLA.TLRmodule.DenseGemmArena(view(ws.storage, :), 1)
    mode = NextLA.GEMMCompute{Float32}()
    for (name, builder) in (("right", _T._execute_compressed_ftlr_foldright_run!),
                            ("left",  _T._execute_compressed_ftlr_foldleft_run!))
        try
            t = timeit(() -> for i in 1:size(rg,1)
                builder(C, LA, LB, plan, i:i, 1f0, 0f0, mode, arena)
            end, 3)
            @printf("  measured fold=:%-6s %9.3f ms\n", name, t)
        catch e
            @printf("  fold=:%-6s FAILED: %s\n", name, sprint(showerror, e)[1:min(end,70)])
        end
    end
    println("\n>> if the measured ratio disagrees with the model ratio, the MAC-count")
    println(">> heuristic in _compressed_ftlr_select_fold is choosing wrong.")
end

# ==================================================== 6. PLAN COST / O(q^3)
# _compressed_ftlr_rank_plan builds right_flops with a triple-nested
# comprehension -> O(qm*qn*qk). Claimed reducible to O(qk*qn + qm*qk).
phase("plan") do
    println("  host-side symbolic cost vs grid size (expect ~q^3 growth):")
    prev_t = prev_q = nothing
    for n in (2048, 4096, 8192, 16384, 32768)
        try
            q = cld(n, BM)
            rg = rankgrid(q, RMAX)
            A = make_ftlr(Float32, n, BM, rg); B = make_ftlr(Float32, n, BM, rg)
            LA = _T.logical_operand(A, 'N'); LB = _T.logical_operand(B, 'N')
            _T._compressed_ftlr_rank_plan(LA, LB)                    # warm
            t = @elapsed for _ in 1:3; _T._compressed_ftlr_rank_plan(LA, LB); end
            t = t / 3 * 1e3
            scaling = prev_t === nothing ? "" :
                @sprintf("  (x%.1f for q x%.1f)", t/prev_t, q/prev_q)
            @printf("  N=%-6d q=%-4d plan=%8.3f ms%s\n", n, q, t, scaling)
            prev_t, prev_q = t, q
            A = B = nothing; CUDA.reclaim()
        catch e
            @printf("  N=%-6d FAILED: %s\n", n, sprint(showerror, e)[1:min(end,70)])
        end
    end
    println("\n>> cubic growth confirms the right_flops nesting is the bottleneck.")
    println(">> Fix: w_k = sum_j col_widths[j]*rB_kj once, then right_flops[i] = sum_k rA_ik*w_k.")
end

# ==================================================== 7. MIXED-PRECISION OUT
# low_rank_terms.jl:34 asserts grouped GEMMEx rejects FP16 operands -> FP32 C.
# Same class of unverified claim as the alignment one. Test it directly.
phase("mixedout") do
    for (Sin, Sout) in ((Float16, Float32), (Core.BFloat16, Float32))
        try
            m = k = n = 256
            A = CUDA.zeros(Sin, m, k);  fill!(A, Sin(0.01))
            B = CUDA.zeros(Sin, k, n);  fill!(B, Sin(0.01))
            C = CUDA.zeros(Sout, m, n)
            t = [NextLA.GroupedGemmTask('N','N', one(Float32), A, B, zero(Float32), C)]
            NextLA.precision_gemm_grouped!(t, NextLA.GEMMCompute{Float32}())
            CUDA.synchronize()
            expect = Float32(0.01)^2 * k
            ok = isapprox(Array(C)[1,1], expect; rtol=1e-2)
            @printf("  %s operands -> %s output: ACCEPTED, correct=%s\n", Sin, Sout, ok)
        catch e
            @printf("  %s -> %s: REJECTED (%s)\n", Sin, Sout,
                    sprint(showerror, e)[1:min(end,90)])
        end
    end
    println("\n>> if ACCEPTED, the eltype(C)===T restriction in low_rank_terms.jl can be")
    println(">> lifted — which matters for FP16 accuracy (FP32 accumulation into C).")
end

# ==================================================== 8. PER-CALL OVERHEAD
# (a) fill!(tdata, 0) every call; (b) Int.(ranks(A)) == ... allocates and
# compares the whole rank grid on every numerical call.
phase("overhead") do
    q = cld(N, BM)
    rg = rankgrid(q, RMAX)
    A, B, C, ws, an = analysed(Float32, N, BM, rg)
    t_full = timeit(() -> _T.gemm!(C, A, B; workspace=ws, alpha=1f0, beta=0f0, analysis=an))
    tzero = 0.0
    for run in an.runs
        run.tdata === nothing && continue
        tzero += timeit(() -> fill!(run.tdata, 0f0), 5)
    end
    tval = @elapsed for _ in 1:20
        Int.(NextLA.ranks(A)) == an.A_ranks && Int.(NextLA.ranks(B)) == an.B_ranks &&
        Int.(NextLA.execution_ranks(A)) == an.A_execution_ranks &&
        Int.(NextLA.execution_ranks(B)) == an.B_execution_ranks
    end
    tval = tval / 20 * 1e3
    @printf("  full analysed gemm!        : %9.3f ms\n", t_full)
    @printf("  of which fill!(tdata, 0)   : %9.3f ms  (%.1f%%)\n", tzero, 100tzero/t_full)
    @printf("  host rank revalidation     : %9.3f ms  (%.1f%%)\n", tval, 100tval/t_full)
    close(an)
    println("\n>> tdata zeroing only needs to cover (k,j) gaps where rank==0;")
    println(">> revalidation should be a version counter, not 4 array materialisations.")
end

println("\nALL DONE")
