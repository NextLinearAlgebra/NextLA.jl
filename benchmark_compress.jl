# NextLA TLR compression benchmark -- GPU only.
#
# Run:
#   julia --project=../gpuenv benchmark_compress.jl

using LinearAlgebra, Printf, Random, Statistics

# User parameters --------------------------------------------------------------
n = [2048, 4096, 8192]
b = [128, 256, 512]
maxrank = [64, 128]
warmup_runs = 1
iterations = 3

tol = 1.0f-5
rel = true
T = Float32
seed = 42

rank_min = 1
rank_max = 48
rank_left_skew_shape = 3.0

# CUDA is intentionally required: this benchmark measures GPU compression only.
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

HAS_CUDA || error("CUDA is not available. Run with a CUDA-enabled project, for example: julia --project=../gpuenv benchmark_compress.jl")

using NextLA
using NextLA: TLRMatrix, compress!, tile_u, tile_v, dense_diag, dense_diag_corner,
              ranks, residuals, ndiag_tiles, noffdiag_tiles, tile_origin_coords,
              tile_size, alloc_workspace

gpu_sync() = CUDA.synchronize()

function reclaim_gpu!()
    gpu_sync()
    GC.gc(true)
    isdefined(CUDA, :reclaim) && CUDA.reclaim()
    gpu_sync()
    return nothing
end

function sample_left_skewed_rank(rng::AbstractRNG)
    span = rank_max - rank_min + 1
    x = rand(rng)^(1 / rank_left_skew_shape)
    return rank_min + min(floor(Int, span * x), span - 1)
end

function case_seed(base_seed::Integer, n::Int, b::Int, maxrank::Int)
    return Int(base_seed + 1_000_003n + 10_007b)
end

# Generate a dense matrix whose off-diagonal tiles have exact rank in
# rank_min:rank_max, biased toward the high end of the interval.
function generate_tiled_lowrank(n::Int, b::Int, ::Type{T}; seed::Int) where {T}
    rng = MersenneTwister(seed)
    A = zeros(T, n, n)
    mt = cld(n, b)
    true_ranks = zeros(Int, mt, mt)

    for j in 1:mt, i in 1:mt
        p0 = (i - 1) * b + 1
        q0 = (j - 1) * b + 1
        tm = min(b, n - p0 + 1)
        tn = min(b, n - q0 + 1)
        tile = view(A, p0:p0 + tm - 1, q0:q0 + tn - 1)
        if i == j
            randn!(rng, tile)
        else
            k = sample_left_skewed_rank(rng)
            mul!(tile, randn(rng, T, tm, k), randn(rng, T, tn, k)')
            true_ranks[i, j] = k
        end
    end

    return A, true_ranks
end

# Tile-by-tile relative Frobenius error, preserving the previous benchmark's
# reconstruction metric.
function rel_error(A_cpu::AbstractMatrix, A_tlr::TLRMatrix)
    D = Array(dense_diag(A_tlr))
    D_corner = Array(dense_diag_corner(A_tlr))
    err_sq = 0.0
    norm_sq = 0.0

    for ob in 1:noffdiag_tiles(A_tlr)
        lin = NextLA.TLRmodule._linear_from_offdiag(A_tlr, ob)
        i, j = NextLA.TLRmodule._inverse_tile_index(A_tlr, lin)
        p0, q0 = tile_origin_coords(A_tlr, i, j)
        tm, tn = tile_size(A_tlr, i, j)
        tile = view(A_cpu, p0:p0 + tm - 1, q0:q0 + tn - 1)
        U_ob = Array(tile_u(A_tlr, ob))
        V_ob = Array(tile_v(A_tlr, ob))
        recon = U_ob * V_ob'
        err_sq += sum(abs2, tile - recon)
        norm_sq += sum(abs2, tile)
    end

    for k in 1:ndiag_tiles(A_tlr)
        p0, q0 = tile_origin_coords(A_tlr, k, k)
        tm, tn = tile_size(A_tlr, k, k)
        tile = view(A_cpu, p0:p0 + tm - 1, q0:q0 + tn - 1)
        diag_tile = k <= size(D, 3) ? @view(D[1:tm, 1:tn, k]) : @view(D_corner[1:tm, 1:tn, 1])
        err_sq += sum(abs2, tile - diag_tile)
        norm_sq += sum(abs2, tile)
    end

    return sqrt(err_sq / norm_sq)
end

function ortho_errors(A_tlr::TLRMatrix)
    rk = Array(ranks(A_tlr))
    errs = Float64[]
    for ob in 1:noffdiag_tiles(A_tlr)
        kr = Int(rk[ob])
        kr == 0 && continue
        U_ob = Array(tile_u(A_tlr, ob))
        I_kr = Matrix{eltype(U_ob)}(I, kr, kr)
        push!(errs, norm(U_ob' * U_ob - I_kr))
    end
    return errs
end

function summary_stats(xs)
    isempty(xs) && return (NaN, NaN, NaN)
    return (minimum(xs), median(xs), maximum(xs))
end

function rank_recovery(A_tlr::TLRMatrix, true_ranks::AbstractMatrix{<:Integer})
    rk = Array(ranks(A_tlr))
    noff = noffdiag_tiles(A_tlr)
    exact = 0
    recoverable = 0
    exact_recoverable = 0

    for ob in 1:noff
        lin = NextLA.TLRmodule._linear_from_offdiag(A_tlr, ob)
        i, j = NextLA.TLRmodule._inverse_tile_index(A_tlr, lin)
        truth = Int(true_ranks[i, j])
        got = Int(rk[ob])
        exact += got == truth
        if truth <= A_tlr.maxrank
            recoverable += 1
            exact_recoverable += got == truth
        end
    end

    return exact, noff, exact_recoverable, recoverable
end

function finite_factors(A_tlr::TLRMatrix)
    arrays = (A_tlr.int_U, A_tlr.int_V, A_tlr.right_U, A_tlr.right_V,
              A_tlr.bottom_U, A_tlr.bottom_V)
    return all(A -> !any(isnan, A), arrays)
end

function run_algorithm(A_gpu, A_cpu, true_ranks, b::Int, maxrank::Int)
    A_tlr = TLRMatrix(A_gpu, b, maxrank)
    ws = alloc_workspace(A_tlr)

    for _ in 1:warmup_runs
        compress!(A_tlr, A_gpu, ws; tol=tol, rel=rel)
        gpu_sync()
    end

    best_time = Inf
    for _ in 1:iterations
        t_run = @elapsed begin
            compress!(A_tlr, A_gpu, ws; tol=tol, rel=rel)
            gpu_sync()
        end
        best_time = min(best_time, t_run)
    end

    rk = Array(ranks(A_tlr))
    resid = Array(residuals(A_tlr))
    ortho = ortho_errors(A_tlr)
    exact, total, exact_rec, total_rec = rank_recovery(A_tlr, true_ranks)
    rmin, rmed, rmax = summary_stats(rk)
    emin, emed, emax = summary_stats(resid)
    omin, omed, omax = summary_stats(ortho)

    return (;
        status="ok",
        time=best_time,
        relerr=rel_error(A_cpu, A_tlr),
        resid_min=emin,
        resid_med=emed,
        resid_max=emax,
        rank_min=rmin,
        rank_med=rmed,
        rank_max=rmax,
        exact=exact,
        total=total,
        exact_recoverable=exact_rec,
        total_recoverable=total_rec,
        ortho_min=omin,
        ortho_med=omed,
        ortho_max=omax,
        ok=finite_factors(A_tlr),
        note="",
    )
end

function failed_result(err)
    msg = sprint(showerror, err)
    return (;
        status="fail",
        time=NaN,
        relerr=NaN,
        resid_min=NaN,
        resid_med=NaN,
        resid_max=NaN,
        rank_min=NaN,
        rank_med=NaN,
        rank_max=NaN,
        exact=0,
        total=0,
        exact_recoverable=0,
        total_recoverable=0,
        ortho_min=NaN,
        ortho_med=NaN,
        ortho_max=NaN,
        ok=false,
        note=first(msg, min(length(msg), 60)),
    )
end

function print_header()
    println("NextLA TLR compress! GPU benchmark")
    println("  n = $n")
    println("  b = $b")
    println("  maxrank = $maxrank")
    println("  warmup_runs = $warmup_runs")
    println("  iterations = $iterations")
    println("  tol = $tol")
    println("  rel = $rel")
    println("  algorithm = cholqr2")
    println("  rank_distribution = left-skewed power(shape=$rank_left_skew_shape) on $rank_min:$rank_max")
    println("  GPU = ", CUDA.name(CUDA.device()))
    println()

    @printf("%6s %5s %7s %10s %10s %10s %10s %9s %7s %7s %7s %13s %13s %10s %10s %10s %-5s %s\n",
            "n", "b", "maxrank", "best_s", "rel_err",
            "res_min", "res_med", "res_max", "rk_min", "rk_med", "rk_max",
            "recov", "recov<=mr", "u_min", "u_med", "u_max", "ok", "note")
    println(repeat("-", 190))
end

function print_row(n::Int, b::Int, maxrank::Int, r)
    recov = r.total == 0 ? "-" : string(r.exact, "/", r.total)
    recov_cap = r.total_recoverable == 0 ? "-" : string(r.exact_recoverable, "/", r.total_recoverable)
    @printf("%6d %5d %7d %10.4f %10.2e %10.2e %10.2e %9.2e %7.0f %7.1f %7.0f %13s %13s %10.2e %10.2e %10.2e %-5s %s\n",
            n, b, maxrank, r.time, r.relerr,
            r.resid_min, r.resid_med, r.resid_max,
            r.rank_min, r.rank_med, r.rank_max,
            recov, recov_cap,
            r.ortho_min, r.ortho_med, r.ortho_max,
            string(r.ok), r.note)
end

function run_case(n::Int, b::Int, maxrank::Int)
    reclaim_gpu!()
    A_cpu, true_ranks = generate_tiled_lowrank(n, b, T; seed=case_seed(seed, n, b, maxrank))
    A_gpu = CuArray(A_cpu)
    gpu_sync()

    result = try
        run_algorithm(A_gpu, A_cpu, true_ranks, b, maxrank)
    catch err
        failed_result(err)
    finally
        reclaim_gpu!()
    end
    print_row(n, b, maxrank, result)
    flush(stdout)

    A_gpu = nothing
    A_cpu = nothing
    true_ranks = nothing
    reclaim_gpu!()
    return nothing
end

function main()
    print_header()
    for n_i in n, b_i in b, maxrank_i in maxrank
        run_case(n_i, b_i, maxrank_i)
    end
end

main()
