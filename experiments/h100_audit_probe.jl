# h100_audit_probe.jl — every open question in one rented-GPU session.
#
#   julia --project=experiments experiments/h100_audit_probe.jl            # all phases
#   PROBE_PHASE=align julia --project=experiments experiments/h100_audit_probe.jl
#
# Phases: align ranks descriptor fusion plan mixedout overhead
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
    println("\n>> prepare one symbolic analysis and reuse it across timed numerical calls.")
end

# ==================================================== 4. FUSION EFFICIENCY
# Fusing stage 3 does NOT change the FLOP count: a fused (bm x rho)*(rho x N)
# GEMM and qn separate (bm x rho)*(rho x bn) GEMMs are the SAME arithmetic in
# a different task grain. A pure-FLOP cost model is therefore structurally
# blind to fusion -- it can only ever see the FoldRight-vs-FoldLeft asymmetry,
# never whether fusing pays off. This phase measures the only number that can
# make fusion enter a scheduling decision at all: achieved GFLOP/s, fused vs
# unfused, for IDENTICAL total work. Uses prepared descriptors so only kernel
# time is measured.
phase("fusion") do
    # BUG FIXED: this used to hardcode Float32 regardless of PROBE_T, so a
    # PROBE_T=Float16 run silently still measured Float32/COMPUTE_32F -- never
    # touching tensor cores, exactly the confound this whole investigation
    # started by falling into once already (the very first alignment probe).
    S = eval(Meta.parse(get(ENV, "PROBE_T", "Float32")))
    bm = BM
    Nrow = parse(Int, get(ENV, "FUSION_N",  "2048"))
    nt   = parse(Int, get(ENV, "FUSION_NT", "8"))
    bn   = BM
    mode = S === Float64 ? NextLA.GEMMCompute{Float64}() : NextLA.GEMMCompute{Float32}()
    println("  dtype=$S  mode=$(typeof(mode))")
    @printf("%-6s %-6s %-5s %11s %13s %16s %8s\n",
            "rho", "Nrow", "qn", "fused(ms)", "unfused(ms)", "GFLOP/s f / u", "penalty")
    for rho in (256, 1024, 2048)
        try
            qn = max(1, cld(Nrow, bn))
            gf = 2 * nt * bm * rho * Nrow / 1e9

            # FUSED: one (bm x rho)*(rho x Nrow) task per "row".
            keepF, tasksF = Any[], NextLA.GroupedGemmTask[]
            for _ in 1:nt
                U  = CUDA.zeros(S, bm, rho);  fill!(U,  S(0.01))
                Tm = CUDA.zeros(S, rho, Nrow); fill!(Tm, S(0.01))
                Cm = CUDA.zeros(S, bm, Nrow)
                push!(keepF, (U, Tm, Cm))
                push!(tasksF, NextLA.GroupedGemmTask('N','N', one(S), U, Tm, zero(S), Cm))
            end
            pf = NextLA.prepare_precision_gemm_grouped(tasksF, mode)
            bk = NextLA.get_backend(tasksF[1].C)
            t_f = timeit(() -> NextLA._with_grouped_host_pointer_mode(bk) do
                NextLA.precision_gemm_grouped_prepared!(pf) end)
            NextLA.destroy_prepared_grouped_gemm!(pf)
            keepF = tasksF = nothing; CUDA.reclaim()

            # UNFUSED: qn separate (bm x rho)*(rho x bn) tasks per "row" --
            # identical total FLOPs; the per-row U is reused across its qn
            # column splits, exactly as FoldLeft's stage 3 does today.
            keepU, tasksU = Any[], NextLA.GroupedGemmTask[]
            for _ in 1:nt
                U = CUDA.zeros(S, bm, rho); fill!(U, S(0.01))
                for _ in 1:qn
                    Tb = CUDA.zeros(S, rho, bn); fill!(Tb, S(0.01))
                    Cb = CUDA.zeros(S, bm, bn)
                    push!(keepU, (U, Tb, Cb))
                    push!(tasksU, NextLA.GroupedGemmTask('N','N', one(S), U, Tb, zero(S), Cb))
                end
            end
            pu = NextLA.prepare_precision_gemm_grouped(tasksU, mode)
            t_u = timeit(() -> NextLA._with_grouped_host_pointer_mode(bk) do
                NextLA.precision_gemm_grouped_prepared!(pu) end)
            NextLA.destroy_prepared_grouped_gemm!(pu)

            @printf("%-6d %-6d %-5d %9.3f  %11.3f  %7.0f / %-7.0f %6.2fx\n",
                    rho, Nrow, qn, t_f, t_u, gf/(t_f/1e3), gf/(t_u/1e3), t_u/t_f)
            keepU = tasksU = nothing; CUDA.reclaim()
        catch e
            @printf("rho=%-6d FAILED: %s\n", rho, sprint(showerror, e)[1:min(end,80)])
        end
    end
    println("\n>> `penalty` = unfused_time / fused_time for IDENTICAL FLOPs. Feed this")
    println(">> number to experiments/fold_schedule_tradeoff.jl as UNFUSED_PENALTY --")
    println(">> that script needs it to convert the FLOP-only fold comparison into a")
    println(">> real time comparison; without it that comparison is blind to fusion.")
    println(">> Scale FUSION_N/FUSION_NT up to better match production qm/N on H100.")
end

# ==================================================== 5. PLAN COST / O(q^3)
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

# ==================================================== 6. MIXED-PRECISION OUT
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

# ==================================================== 7. PER-CALL OVERHEAD
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
