using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions, Dates
include("benchmark.jl")
include("flops.jl")   # calculate_gflops(flops, time_ns)

# ---------------------------------------------------------------------------
# FLOP model: C <- alpha*A*B + beta*C, square n x n. Standard GEMM count (2mnk);
# the O(n^2) alpha/beta terms are ignored, matching CUBLAS/CUTLASS convention.
# ---------------------------------------------------------------------------
flops_gemm(m, n, k) = 2.0 * m * n * k

# ---------------------------------------------------------------------------
# CUTLASS via cutlass_profiler (https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
#   - Binary: $CUTLASS_PROFILER, else `cutlass_profiler` on PATH. If neither, CUTLASS is skipped.
#   - Optional: $CUTLASS_KERNELS restricts kernels (maps to --kernels=...), e.g. "sgemm" or "tensorop".
#   - Same problem as CUBLAS.gemm!: column-major, N/N, alpha=-1, beta=1. The profiler tries every
#     matching kernel; we report the fastest successful one (verification off, we only time).
# ---------------------------------------------------------------------------
const CUTLASS_PROFILER = get(ENV, "CUTLASS_PROFILER", something(Sys.which("cutlass_profiler"), ""))
const CUTLASS_TYPES = ["CUTLASS F32" => "f32", "CUTLASS F64" => "f64"]  # name => profiler element type (A/B/C/accum)

cutlass_gemm_ms(profiler::AbstractString, n::Integer, T::AbstractString; kw...) = cutlass_gemm_ms(profiler, n, n, n, T; kw...)

function cutlass_gemm_ms(profiler::AbstractString, m::Integer, n::Integer, k::Integer, T::AbstractString; alpha = -1.0, beta = 1.0)
    mktempdir() do dir
        out = joinpath(dir, "cutlass_report")
        big = flops_gemm(m, n, k) >= flops_gemm(16384, 16384, 16384)   # == (n >= 16384) for square problems
        args = ["--operation=Gemm", "--m=$m", "--n=$n", "--k=$k",
                "--A=$T:column", "--B=$T:column", "--C=$T:column",
                "--accumulator-type=$T", "--alpha=$alpha", "--beta=$beta",
                "--warmup-iterations=$(big ? 1 : 5)", "--profiling-iterations=$(big ? 3 : 10)",
                "--verification-enabled=false", "--output=$out"]
        haskey(ENV, "CUTLASS_KERNELS") && push!(args, "--kernels=" * ENV["CUTLASS_KERNELS"])
        run(pipeline(Cmd([String(profiler); args]); stdout = devnull))

        # The profiler appends the operation kind to --output (e.g. <out>.gemm.csv); don't assume the exact name.
        csvs = filter(f -> endswith(f, ".csv"), readdir(dir; join = true))
        isempty(csvs) && error("cutlass_profiler wrote no CSV (no $T GEMM kernels built into the profiler?)")

        best_ms, best_op, header = Inf, "", String[]
        for f in csvs
            lines = readlines(f)
            isempty(lines) && continue
            header = String.(strip.(split(lines[1], ',')))
            i_rt = findfirst(==("Runtime"), header)
            i_st = findfirst(==("Status"), header)
            i_op = findfirst(==("Operation"), header)
            i_rt === nothing && continue
            for line in lines[2:end]
                cols = String.(strip.(split(line, ',')))
                length(cols) == length(header) || continue
                i_st !== nothing && lowercase(cols[i_st]) != "success" && continue
                rt = tryparse(Float64, cols[i_rt])
                (rt === nothing || rt <= 0) && continue
                if rt < best_ms
                    best_ms, best_op = rt, i_op === nothing ? "" : cols[i_op]
                end
            end
        end
        isfinite(best_ms) || error("no successful kernel in profiler CSV (columns: $(join(header, ", ")))")
        return best_ms, best_op
    end
end

# ---------------------------------------------------------------------------
# Markdown report (written next to wherever the script was launched from, i.e. pwd(),
# same place savefig() drops the plots). Timestamped so repeated runs never overwrite each other.
# ---------------------------------------------------------------------------
function md_table(io, title, impls, xs, records, field, fmt; xfield = :n, level = "###")
    println(io, level, " ", title, "\n")
    println(io, "| Implementation | ", join(xs, " | "), " |")
    println(io, "|:--|", repeat("--:|", length(xs)))
    for impl in impls
        cells = map(xs) do x
            i = findfirst(r -> r.impl == impl && getfield(r, xfield) == x, records)
            (i === nothing || isnan(getfield(records[i], field))) ? "—" : fmt(getfield(records[i], field))
        end
        println(io, "| ", impl, " | ", join(cells, " | "), " |")
    end
    println(io)
end

# One block of tables per (family, k) for the tall-and-skinny problems (x-axis = m).
function md_tallskinny(io, records)
    ts = filter(r -> r.family != "square", records)
    isempty(ts) && return
    println(io, "## Tall-and-skinny GEMM\n")
    println(io, "- **FLOP model:** 2mnk per GEMM; GFLOP/s = 2mnk / best runtime")
    println(io, "- **Timing:** ", TS_INCLUDE_SETUP ? "includes" : "excludes",
            " the FullMixedPrec(C) construction (set `TS_INCLUDE_SETUP=1` to include it, as the square benchmark above does)\n")
    for fam in unique(r.family for r in ts)
        println(io, "### ", fam == "tall" ? "Tall × small square: C(m×k) = A(m×k)·B(k×k)" :
                                           "Short-fat × tall: C(m×m) = A(m×k)·B(k×m)", "\n")
        for k in sort(unique(r.k for r in ts if r.family == fam))
            recs = filter(r -> r.family == fam && r.k == k, ts)
            ms = sort(unique(r.m for r in recs))
            impls = unique(r.impl for r in recs)
            println(io, "#### k = ", k, "\n")
            md_table(io, "Throughput (GFLOP/s) vs m", impls, ms, recs, :gflops, x -> @sprintf("%.1f", x); xfield = :m, level = "#####")
            md_table(io, "Runtime (ms, best of N) vs m", impls, ms, recs, :runtime_ms, x -> @sprintf("%.3f", x); xfield = :m, level = "#####")
            md_table(io, "Relative error vs. FP64 CUBLAS", impls, ms, recs, :rel_err, x -> @sprintf("%.2e", x); xfield = :m, level = "#####")
        end
    end
end

function write_results_markdown(records, n_values, console_log; dir = pwd())
    path = joinpath(dir, "recgemm_results_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".md")
    sq = filter(r -> r.family == "square", records)
    impls = unique(r.impl for r in sq)
    open(path, "w") do io
        println(io, "# recgemm! benchmark results\n")
        println(io, "- **Date:** ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "- **GPU:** ", CUDA.functional() ? CUDA.name(CUDA.device()) : "n/a")
        println(io, "- **Julia:** ", VERSION, " | **CUDA.jl:** ", pkgversion(CUDA))
        println(io, "- **Sizes (n x n x n):** ", join(n_values, ", "))
        println(io, "- **FLOP model:** 2n³ per GEMM; GFLOP/s = 2n³ / best runtime")
        println(io, "- **CUTLASS profiler:** ", isempty(CUTLASS_PROFILER) ? "not found (skipped)" : CUTLASS_PROFILER, "\n")

        md_table(io, "Throughput (GFLOP/s)", impls, n_values, sq, :gflops, x -> @sprintf("%.1f", x))
        md_table(io, "Runtime (ms, best of N)", impls, n_values, sq, :runtime_ms, x -> @sprintf("%.3f", x))
        md_table(io, "Relative error vs. FP64 CUBLAS", impls, n_values, sq, :rel_err, x -> @sprintf("%.2e", x))

        md_tallskinny(io, records)

        notes = filter(r -> !isempty(r.note), records)
        if !isempty(notes)
            println(io, "### Notes\n")
            for r in notes
                println(io, "- ", r.family == "square" ? "n=$(r.n)" : "$(r.family) m=$(r.m) n=$(r.n) k=$(r.k)", ", ", r.impl, ": `", r.note, "`")
            end
            println(io)
        end

        println(io, "## Raw console output\n")
        println(io, "```text")
        print(io, console_log)
        println(io, "\n```")
    end
    return path
end

# ---------------------------------------------------------------------------
# Tall-and-skinny problems (run after the square sweep inside run_recgemm_benchmark; needs the rectangular
# FullMixedPrec / recgemm! in src/).
#   family "tall"      C(m x n) = A(m x k) * B(k x n),  m >> k = n    (e.g. trailing-panel update in LU/QR)
#   family "shortfat"  C(m x m) = A(m x k) * B(k x m),  m >> k        (small inner dimension, large output)
# Every precision list needs C's column count >= 2^(length-1) to reach its base case (longest list has 7 entries),
# hence k = n >= 64 for the tall family.
# Environment knobs:
#   TS_SKIP=1           skip the tall-and-skinny section
#   TS_QUICK=1          small grid (smoke test)
#   TS_INCLUDE_SETUP=1  time FullMixedPrec(copy(C)) + recgemm! together, like the square benchmark does. Default is
#                       recgemm! only (the C copy/conversion is O(m*n) and dominates less-compute shapes at m ~ 10^6).
# ---------------------------------------------------------------------------
const TS_ENABLED       = get(ENV, "TS_SKIP", "0") != "1"
const TS_QUICK         = get(ENV, "TS_QUICK", "0") == "1"
# RECGEMM_ONLY=tallskinny  -> skip the square sweep (and its plots) and run only the tall-and-skinny sweep
const SQ_ENABLED       = get(ENV, "RECGEMM_ONLY", "") != "tallskinny"
const TS_INCLUDE_SETUP = get(ENV, "TS_INCLUDE_SETUP", "0") == "1"
const TS_KS_TALL = TS_QUICK ? [64]          : [64, 256, 1024]
const TS_MS_TALL = TS_QUICK ? [4096, 16384] : [4096, 16384, 65536, 262144, 1048576]
const TS_KS_SF   = TS_QUICK ? [64]          : [64, 256, 1024]
const TS_MS_SF   = TS_QUICK ? [1024, 4096]  : [1024, 4096, 16384, 32768]

tallskinny_problems() = vcat(
    [(family = "tall",     m = m, n = k, k = k) for k in TS_KS_TALL for m in TS_MS_TALL],
    [(family = "shortfat", m = m, n = m, k = k) for k in TS_KS_SF   for m in TS_MS_SF],
)

# Worst-case device bytes (FP64 everywhere): A, B and their FP64 copies, plus C, FP64 ground truth,
# the FullMixedPrec copy, the reconstruction and its FP64 upcast.
ts_est_bytes(p) = 8.0 * (2 * (p.m * p.k + p.k * p.n) + 5 * p.m * p.n)

# GPU-side normal(0, 0.001) in element type T (Float16 goes via Float32), so m ~ 10^6 needs no host randn.
gpu_randn(::Type{T}, dims...) where {T} =
    T === Float64 ? CUDA.randn(Float64, dims...) .* 0.001 : T.(CUDA.randn(Float32, dims...) .* 0.001f0)

# ---------------------------------------------------------------------------
# cuBLAS GemmEx baselines (CUBLAS.gemmEx! -- the same routine recgemm! calls for its F16 blocks), full m x n x k,
# C <- -A*B + C. Variants: name, element type of A/B, element type of C, CUDA.jl math mode (selects the compute type).
#   F64 / F32           full-precision GemmEx (compute 64F / 32F)
#   TF32                F32 operands on the tensor cores (FAST_MATH -> CUBLAS_COMPUTE_32F_FAST_TF32)
#   F16                 F16 in, F16 out (what CUDA.jl picks for this signature is the F16 tensor-core path)
#   F16 in, F32 out     F16 operands, F32 output/accumulate
# Names deliberately don't start with "CUBLAS" so they get accuracy curves in the plots (unlike the gemm! rows).
# Which kernels/compute types actually ran can be checked with CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stdout.
# ---------------------------------------------------------------------------
const GEMMEX_CASES = [
    ("GemmEx F64",             Float64, Float64, CUDA.DEFAULT_MATH),
    ("GemmEx F32",             Float32, Float32, CUDA.DEFAULT_MATH),
    ("GemmEx TF32",            Float32, Float32, CUDA.FAST_MATH),
    ("GemmEx F16",             Float16, Float16, CUDA.DEFAULT_MATH),
    ("GemmEx F16 in, F32 out", Float16, Float32, CUDA.DEFAULT_MATH),
]

# `rec!(impl, runtime_ms, gflops; rel_err=...)` is the caller's recorder (square: record!(n, ...); tall: see below).
function run_gemmex_case(m, n, k, emit, rec!)
    gemm_flops = flops_gemm(m, n, k)
    for (name, TAB, TC, mode) in GEMMEX_CASES
        try
            d_A = gpu_randn(TAB, m, k)
            d_B = gpu_randn(TAB, k, n)
            d_C = CUDA.zeros(TC, m, n)
            alpha, beta = TC(-1.0), TC(1.0)

            # Ground truth in FP64 from the same operands (no extra copies when they already are FP64)
            d_truth = CUDA.zeros(Float64, m, n)
            CUBLAS.gemm!('N', 'N', -1.0, TAB === Float64 ? d_A : Float64.(d_A), TAB === Float64 ? d_B : Float64.(d_B), 1.0, d_truth)

            CUDA.math_mode!(mode)
            CUBLAS.gemmEx!('N', 'N', alpha, d_A, d_B, beta, d_C)     # accuracy run (C starts at 0)
            CUDA.synchronize()
            X = Float64.(d_C)
            X .-= d_truth
            relative_error = max(norm(X) / norm(d_truth), 1e-20)
            X = d_truth = nothing

            backend = KernelAbstractions.get_backend(d_A)
            time_ns = run_manual_benchmark(backend) do
                CUBLAS.gemmEx!('N', 'N', alpha, d_A, d_B, beta, d_C)  # C accumulates across samples; cost is unchanged
            end
            runtime_ms = time_ns / 1_000_000
            gflops = calculate_gflops(gemm_flops, time_ns)
            rec!(name, runtime_ms, gflops; rel_err = relative_error)
            emit(@sprintf("  %-38s | Rel. Error: %9.2e | Runtime: %9.3f ms | %10.1f GFLOP/s\n", name, relative_error, runtime_ms, gflops))
            d_A = d_B = d_C = nothing   # drop references so the GC below can actually free them
        catch e
            emit(@sprintf("  %-38s | FAILED: %s\n", name, first(split(sprint(showerror, e), '\n'))))
        finally
            CUDA.math_mode!(CUDA.DEFAULT_MATH)
            GC.gc(); CUDA.reclaim()
        end
    end
end

# One tall/skinny problem: every recgemm! scenario, then CUBLAS F32/F64, GemmEx, then CUTLASS.
# `record!` / `emit` are the closures from run_recgemm_benchmark, so results land in the same markdown + console log.
function run_tallskinny_case(p, test_scenarios, emit, record!)
    emit("\n" * "-"^70 * "\n")
    emit("[$(p.family)] C($(p.m) x $(p.n)) = A($(p.m) x $(p.k)) * B($(p.k) x $(p.n))\n")

    GC.gc(); CUDA.reclaim()
    if ts_est_bytes(p) > 0.9 * CUDA.available_memory()
        emit(@sprintf("  [skipped] needs ~%.1f GB, only %.1f GB free\n", ts_est_bytes(p) / 2^30, CUDA.available_memory() / 2^30))
        return
    end
    gemm_flops = flops_gemm(p.m, p.n, p.k)
    rec!(args...; kw...) = record!(p.n, args...; family = p.family, m = p.m, k = p.k, kw...)

    for (name, precisions) in sort!(collect(test_scenarios); by = first)
        T_out = precisions[end]
        alpha, beta = T_out(-1.0), T_out(1.0)

        d_A = gpu_randn(T_out, p.m, p.k)
        d_B = gpu_randn(T_out, p.k, p.n)
        d_C_orig = CUDA.zeros(T_out, p.m, p.n)

        # Ground truth in FP64
        d_C_truth = CUDA.zeros(Float64, p.m, p.n)
        CUBLAS.gemm!('N', 'N', -1.0, Float64.(d_A), Float64.(d_B), 1.0, d_C_truth)

        # Accuracy
        C_mixed = FullMixedPrec(copy(d_C_orig); precisions = precisions)
        recgemm!(alpha, d_A, d_B, beta, C_mixed)
        X = Float64.(reconstruct_matrix(C_mixed))
        X .-= d_C_truth
        relative_error = max(norm(X) / norm(d_C_truth), 1e-20)
        X = nothing

        # Performance
        backend = KernelAbstractions.get_backend(d_A)
        C_perf = FullMixedPrec(copy(d_C_orig); precisions = precisions)   # reused across samples unless TS_INCLUDE_SETUP
        time_ns = run_manual_benchmark(backend) do
            if TS_INCLUDE_SETUP
                C_s = FullMixedPrec(copy(d_C_orig); precisions = precisions)
                CUDA.@sync recgemm!(alpha, d_A, d_B, beta, C_s)
            else
                CUDA.@sync recgemm!(alpha, d_A, d_B, beta, C_perf)
            end
        end
        runtime_ms = time_ns / 1_000_000
        gflops = calculate_gflops(gemm_flops, time_ns)
        rec!(name, runtime_ms, gflops; rel_err = relative_error)
        emit(@sprintf("  %-38s | Rel. Error: %9.2e | Runtime: %9.3f ms | %10.1f GFLOP/s\n",
                      name, relative_error, runtime_ms, gflops))

        d_A = d_B = d_C_orig = d_C_truth = C_mixed = C_perf = nothing
        GC.gc(); CUDA.reclaim()
    end

    emit("\n--- Benchmarking standard CUBLAS.gemm! ---\n")
    for (name, T_prec) in ("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)
        d_A = gpu_randn(T_prec, p.m, p.k)
        d_B = gpu_randn(T_prec, p.k, p.n)
        d_C = CUDA.zeros(T_prec, p.m, p.n)
        backend = KernelAbstractions.get_backend(d_A)
        time_ns = run_manual_benchmark(backend) do
            CUBLAS.gemm!('N', 'N', T_prec(-1.0), d_A, d_B, T_prec(1.0), d_C)
        end
        runtime_ms = time_ns / 1_000_000
        gflops = calculate_gflops(gemm_flops, time_ns)
        rec!(name, runtime_ms, gflops)
        emit(@sprintf("  %-38s | Runtime: %9.3f ms | %10.1f GFLOP/s\n", name, runtime_ms, gflops))
        d_A = d_B = d_C = nothing
        GC.gc(); CUDA.reclaim()
    end

    emit("\n--- Benchmarking cuBLAS GemmEx ---\n")
    run_gemmex_case(p.m, p.n, p.k, emit, rec!)

    emit("\n--- Benchmarking CUTLASS (cutlass_profiler) ---\n")
    if isempty(CUTLASS_PROFILER)
        emit("  [skipped] cutlass_profiler not found; set ENV[\"CUTLASS_PROFILER\"] or put it on PATH\n")
    else
        GC.gc(); CUDA.reclaim()
        for (name, T_str) in CUTLASS_TYPES
            try
                runtime_ms, kernel = cutlass_gemm_ms(CUTLASS_PROFILER, p.m, p.n, p.k, T_str)
                gflops = calculate_gflops(gemm_flops, runtime_ms * 1e6)
                rec!(name, runtime_ms, gflops; note = kernel)
                emit(@sprintf("  %-38s | Runtime: %9.3f ms | %10.1f GFLOP/s | %s\n", name, runtime_ms, gflops, kernel))
            catch e
                emit(@sprintf("  %-38s | FAILED: %s\n", name, first(split(sprint(showerror, e), '\n'))))
            end
        end
    end
end

# Runtime / accuracy vs m, one figure per (family, k), next to the square plots.
function plot_tallskinny(records)
    ts = filter(r -> r.family != "square", records)
    for fam in unique(r.family for r in ts), k in sort(unique(r.k for r in ts if r.family == fam))
        recs = filter(r -> r.family == fam && r.k == k, ts)
        perf = plot(title = "$fam, k=$k: runtime vs m", xlabel = "m", ylabel = "Runtime (ms)", legend = :outertopright, xaxis = :log2, yaxis = :log10)
        acc  = plot(title = "$fam, k=$k: accuracy vs m", xlabel = "m", ylabel = "-log10(Relative Error)", legend = :outertopright, xaxis = :log2)
        for impl in unique(r.impl for r in recs)
            rs = sort(filter(r -> r.impl == impl, recs); by = r -> r.m)
            ls = startswith(impl, "CUBLAS") ? :dash : startswith(impl, "GemmEx") ? :dashdot : startswith(impl, "CUTLASS") ? :dot : :solid
            plot!(perf, [r.m for r in rs], [r.runtime_ms for r in rs], label = impl, marker = :auto, linestyle = ls)
            if !startswith(impl, "CUBLAS") && !startswith(impl, "CUTLASS") && impl != "Pure F64" && impl != "GemmEx F64"
                plot!(acc, [r.m for r in rs], [-log10(max(r.rel_err, 1e-18)) for r in rs], label = impl, marker = :auto)
            end
        end
        savefig(perf, "recgemm_$(fam)_k$(k)_performance.png")
        savefig(acc,  "recgemm_$(fam)_k$(k)_accuracy.png")
    end
end

function run_recgemm_benchmark()
    # Define the matrix sizes to test
    n_values = SQ_ENABLED ? [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768] : Int[] #16, 32, 256, 512, 1024, ... 65536

    # Define the different mixed-precision scenarios
    test_scenarios = Dict(
        "Pure F16"             => [Float16, Float16, Float16],
        "Pure F32"             => [Float32, Float32, Float32],
        "Pure F64"             => [Float64, Float64, Float64],
        "[F32, F32, F64]"      => [Float32, Float32, Float64],
        "[F32, F32, F64, F64]" => [Float32, Float32, Float64, Float64],
        "[F64, F64, F32, F32]" => [Float64, Float64, Float32, Float32],
        "[F32, F64, F64]"      => [Float32, Float64, Float64],
        "[F16, F16, F32]"      => [Float16, Float16, Float32],
        "[F16, F16, F16, F32]" => [Float16, Float16, Float16, Float32],
        "[F16, F16, F16, F16, F32]" => [Float16, Float16, Float16, Float16, Float32],
        "[F16, F32, F32]"      => [Float16, Float32, Float32],
        "[F16, F32]"           => [Float16, Float32],

        # --- Compelling New Configurations ---

        # Smooth Gradients (F16 leaf -> F32 mid -> F64 root)
        # "[F16, F16, F32, F64]" => [Float16, Float16, Float32, Float64],
        # "[F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float32, Float64],
        # "[F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float32, Float64],
        # "[F16, F16, F16, F16, F16, F16, F32, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64],

        # # Extreme Jumps (F16 direct to F64)
        "[F16, F64]"           => [Float16, Float64],
        "[F16, F16, F16, F64]" => [Float16, Float16, Float16, Float64],
        "[F16, F16, F16, F16, F64]" => [Float16, Float16, Float16, Float16, Float64],
        "[F16, F16, F16, F16, F16, F16, F64]" => [Float16, Float16, Float16, Float16, Float16, Float16, Float64]
    )

    # Simplified dictionaries to store results
    accuracy_results = Dict(name => Float64[] for name in keys(test_scenarios))
    runtime_results = Dict(name => Float64[] for name in keys(test_scenarios))

    cublas_runtime_results = Dict(
        "CUBLAS F32" => Float64[],
        "CUBLAS F64" => Float64[]
    )

    # Flat record of every measurement (feeds the markdown report) + mirror of everything printed.
    records = NamedTuple[]
    log_io = IOBuffer()
    emit(s) = (print(s); print(log_io, s))
    # family = "square" (n x n x n, the sweep below) | "tall" | "shortfat" (tall/skinny section; m, k are then the real dims)
    record!(n, impl, runtime_ms, gflops; rel_err = NaN, note = "", family = "square", m = n, k = n) =
        push!(records, (family = family, m = m, n = n, k = k, impl = impl, runtime_ms = runtime_ms, gflops = gflops, rel_err = rel_err, note = note))

    emit("🚀 Starting recgemm! Benchmark...\n")

    # try/finally: even if a large size OOMs or CUTLASS blows up, the results so far still land in the .md
    try
        for n in n_values
            emit("\n" * "-"^50 * "\n")
            emit("Benchmarking C(n x n)=$n, A(n x n)=$n, B(n x n)=$n\n")
            gemm_flops = flops_gemm(n, n, n)

            for (name, precisions) in test_scenarios
                T_out = precisions[end]
                alpha, beta = -1.0, 1.0

                A_cpu = randn(T_out, n, n) .* 0.001f0
                B_cpu = randn(T_out, n, n) .* 0.001f0

                # Make strictly diagonally dominant but safe from Float16 overflow (max 65504)
                # A_cpu .+= Diagonal(fill(T_out(100.0), n))
                # B_cpu .+= Diagonal(fill(T_out(100.0), n))

                d_A = CuArray(A_cpu)
                d_B = CuArray(B_cpu)
                d_C_orig = CuArray(zeros(T_out, n, n))

                # Ground truth calculation
                d_A_fp64 = CuArray{Float64}(d_A)
                d_B_fp64 = CuArray{Float64}(d_B)
                d_C_ground_truth = CuArray(zeros(Float64, n, n))
                CUBLAS.gemm!('N', 'N', Float64(alpha), d_A_fp64, d_B_fp64, Float64(beta), d_C_ground_truth)

                # --- Simplified Logic: Call the correct function based on the test case ---
                C_for_custom = copy(d_C_orig)
                C_custom_result = if name in ["Pure F16", "Pure F32", "Pure F64"]
                    # For pure precision tests, call the standard recursive function
                    alpha = T_out(alpha)
                    beta = T_out(beta)
                    # Fallback to direct gemm for pure arrays, or wrap in FullMixedPrec with same type
                    C_mixed = FullMixedPrec(C_for_custom; precisions=precisions)
                    recgemm!(alpha, d_A, d_B, beta, C_mixed)
                    reconstruct_matrix(C_mixed)
                else
                    # For mixed precision, use the FullMixedPrec structure
                    alpha = T_out(alpha)
                    beta = T_out(beta)
                    C_mixed = FullMixedPrec(C_for_custom; precisions=precisions)
                    recgemm!(alpha, d_A, d_B, beta, C_mixed)
                    reconstruct_matrix(C_mixed)
                end

                error_norm = norm(CuArray{Float64}(C_custom_result) - d_C_ground_truth)
                solution_norm = norm(d_C_ground_truth)
                relative_error = max(error_norm / solution_norm, 1e-20)

                push!(accuracy_results[name], -log10(max(relative_error, 1e-18)))

                # Performance test
                backend = KernelAbstractions.get_backend(d_A)
                time_ns = run_manual_benchmark(backend) do
                    alpha = T_out(alpha)
                    beta = T_out(beta)
                    C_perf_mixed = FullMixedPrec(copy(d_C_orig); precisions=precisions)
                    CUDA.@sync recgemm!(alpha, d_A, d_B, beta, C_perf_mixed)
                end
                runtime_ms = time_ns / 1_000_000
                gflops = calculate_gflops(gemm_flops, time_ns)
                push!(runtime_results[name], runtime_ms)
                record!(n, name, runtime_ms, gflops; rel_err = relative_error)

                emit(@sprintf("  %-22s | Rel. Error: %9.2e | Runtime: %8.3f ms | %10.1f GFLOP/s\n",
                              name, relative_error, runtime_ms, gflops))
            end

            emit("\n--- Benchmarking standard CUBLAS.gemm! ---\n")

            for (name, T_prec) in sort!(collect(Dict("CUBLAS F32" => Float32, "CUBLAS F64" => Float64)); by = first)
                alpha, beta = T_prec(-1.0), T_prec(1.0)

                A_cpu = randn(T_prec, n, n) .* 0.001f0
                B_cpu = randn(T_prec, n, n) .* 0.001f0

                # Make strictly diagonally dominant (consistent with above)
                # A_cpu .+= Diagonal(fill(T_prec(100.0), n))
                # B_cpu .+= Diagonal(fill(T_prec(100.0), n))

                d_A_cublas = CuArray(A_cpu)
                d_B_cublas = CuArray(B_cpu)
                d_C_cublas = CuArray(zeros(T_prec, n, n))

                backend = KernelAbstractions.get_backend(d_A_cublas)
                time_ns = run_manual_benchmark(backend) do
                    CUBLAS.gemm!('N', 'N', alpha, d_A_cublas, d_B_cublas, beta, d_C_cublas)
                end
                runtime_ms = time_ns / 1_000_000
                gflops = calculate_gflops(gemm_flops, time_ns)
                push!(cublas_runtime_results[name], runtime_ms)
                record!(n, name, runtime_ms, gflops)
                emit(@sprintf("  %-22s | Runtime: %8.3f ms | %10.1f GFLOP/s\n", name, runtime_ms, gflops))
            end

            emit("\n--- Benchmarking cuBLAS GemmEx ---\n")
            run_gemmex_case(n, n, n, emit, (args...; kw...) -> record!(n, args...; kw...))

            emit("\n--- Benchmarking CUTLASS (cutlass_profiler) ---\n")

            if isempty(CUTLASS_PROFILER)
                emit("  [skipped] cutlass_profiler not found; set ENV[\"CUTLASS_PROFILER\"] or put it on PATH\n")
            else
                # The profiler is a separate process and needs its own 3 n^2 buffers: hand back Julia's cached GPU memory first.
                GC.gc(); CUDA.reclaim()
                for (name, T_str) in CUTLASS_TYPES
                    try
                        runtime_ms, kernel = cutlass_gemm_ms(CUTLASS_PROFILER, n, T_str)
                        gflops = calculate_gflops(gemm_flops, runtime_ms * 1e6)
                        record!(n, name, runtime_ms, gflops; note = kernel)
                        emit(@sprintf("  %-22s | Runtime: %8.3f ms | %10.1f GFLOP/s | %s\n", name, runtime_ms, gflops, kernel))
                    catch e
                        emit(@sprintf("  %-22s | FAILED: %s\n", name, first(split(sprint(showerror, e), '\n'))))
                    end
                end
            end
        end

        if SQ_ENABLED
        # --- Simplified Plotting Logic ---
        emit("\n" * "="^60 * "\n")
        emit("📊 Generating and saving plots...\n")

        acc_plot = plot(title="Accuracy vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="-log10(Relative Error)", legend=:outertopright, xaxis=:log2)
        perf_plot = plot(title="Performance vs. Matrix Size", xlabel="Matrix Size (n)", ylabel="Runtime (ms)", legend=:outertopright, xaxis=:log2, yaxis=:log10)

        for (name, acc_values) in accuracy_results
            if name != "Pure Float64"
                plot!(acc_plot, n_values, acc_values, label=name, marker=:auto)
            end
        end

        for (name, runtimes) in runtime_results
            plot!(perf_plot, n_values, runtimes, label=name, marker=:auto)
        end

        for (name, runtimes) in cublas_runtime_results
            plot!(perf_plot, n_values, runtimes, label=name, marker=:auto, linestyle=:dash, linewidth=2)
        end

        # CUTLASS runtimes (only sizes that succeeded)
        for (name, _) in CUTLASS_TYPES
            rs = filter(r -> r.family == "square" && r.impl == name, records)
            isempty(rs) || plot!(perf_plot, [r.n for r in rs], [r.runtime_ms for r in rs],
                                 label=name, marker=:auto, linestyle=:dot, linewidth=2)
        end

        # cuBLAS GemmEx baselines (runtime dash-dot; accuracy next to the recgemm! configs)
        for name in unique(r.impl for r in records if r.family == "square" && startswith(r.impl, "GemmEx"))
            rs = filter(r -> r.family == "square" && r.impl == name, records)
            plot!(perf_plot, [r.n for r in rs], [r.runtime_ms for r in rs], label=name, marker=:auto, linestyle=:dashdot, linewidth=2)
            name == "GemmEx F64" || plot!(acc_plot, [r.n for r in rs], [-log10(max(r.rel_err, 1e-18)) for r in rs],
                                          label=name, marker=:auto, linestyle=:dashdot, linewidth=2)
        end

        savefig(acc_plot, "recgemm_accuracy.png")
        savefig(perf_plot, "recgemm_performance.png")

        end  # SQ_ENABLED

        # --- Tall-and-skinny sweep (after the square results/plots are safely on disk) ---
        if TS_ENABLED
            emit("\n" * "="^60 * "\n")
            emit("📐 Tall-and-skinny benchmarks$(TS_QUICK ? " (TS_QUICK)" : "")...\n")
            for p in tallskinny_problems()
                run_tallskinny_case(p, test_scenarios, emit, record!)
            end
            plot_tallskinny(records)
        end

        emit("✅ Benchmark complete. Plots saved to disk.\n")
        emit("="^60 * "\n")
    finally
        md_path = write_results_markdown(records, n_values, String(take!(copy(log_io))))
        println("📝 Results written to $md_path")
    end
end

run_recgemm_benchmark()
