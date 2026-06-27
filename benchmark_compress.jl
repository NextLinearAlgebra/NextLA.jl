# NextLA TLR compression benchmark — CPU and GPU
#
# Run (CPU only):
#   julia --project=. benchmark_compress.jl
#
# Run (CPU + GPU):
#   julia --project=../gpuenv benchmark_compress.jl
#
# The script detects CUDA availability automatically.

using LinearAlgebra, Printf, Random, Statistics

# ── load CUDA if available ────────────────────────────────────────────────────
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

using NextLA
using NextLA: TLRMatrix, compress!, tile_u, tile_v, dense_diag, ranks,
              ndiag_tiles, noffdiag_tiles, tile_origin_coords, tile_size, alloc_workspace

# ── generate a matrix whose off-diagonal tiles have rank in [r_lo, r_hi] ─────
function generate_tiled_lowrank(n::Int, b::Int;
                                 r_lo::Int=10, r_hi::Int=20,
                                 T::Type=Float32, seed::Int=42)
    rng        = MersenneTwister(seed)
    A          = zeros(T, n, n)
    mt         = cld(n, b)
    true_ranks = zeros(Int, mt, mt)

    for j in 1:mt, i in 1:mt
        p0 = (i-1)*b + 1;  q0 = (j-1)*b + 1
        tm = min(b, n-p0+1); tn = min(b, n-q0+1)
        if i == j
            A[p0:p0+tm-1, q0:q0+tn-1] .= randn(rng, T, tm, tn)
        else
            k = rand(rng, r_lo:r_hi)
            mul!(view(A, p0:p0+tm-1, q0:q0+tn-1),
                 randn(rng, T, tm, k),
                 randn(rng, T, tn, k)')
            true_ranks[i, j] = k
        end
    end
    A, true_ranks
end

# ── tile-by-tile relative Frobenius error ─────────────────────────────────────
function rel_error(A_cpu::AbstractMatrix, A_tlr::TLRMatrix)
    D  = Array(dense_diag(A_tlr))
    err_sq = norm_sq = 0.0
    for ob in 1:noffdiag_tiles(A_tlr)
        lin = NextLA.TLRmodule._linear_from_offdiag(A_tlr, ob)
        i, j = NextLA.TLRmodule._inverse_tile_index(A_tlr, lin)
        p0, q0 = tile_origin_coords(A_tlr, i, j)
        tm, tn = tile_size(A_tlr, i, j)
        tile  = A_cpu[p0:p0+tm-1, q0:q0+tn-1]
        U_ob  = Array(tile_u(A_tlr, ob))
        V_ob  = Array(tile_v(A_tlr, ob))
        recon = U_ob * V_ob'
        err_sq  += sum(abs2, tile - recon)
        norm_sq += sum(abs2, tile)
    end
    for k in 1:ndiag_tiles(A_tlr)
        p0, q0 = tile_origin_coords(A_tlr, k, k)
        tm, tn = tile_size(A_tlr, k, k)
        tile = A_cpu[p0:p0+tm-1, q0:q0+tn-1]
        err_sq  += sum(abs2, tile - D[1:tm,1:tn,k])
        norm_sq += sum(abs2, tile)
    end
    sqrt(err_sq / norm_sq)
end

# ── synchronise helper (no-op on CPU, CUDA.synchronize() on GPU) ─────────────
gpu_sync() = HAS_CUDA ? CUDA.synchronize() : nothing

# ── run one benchmark case on a given device ──────────────────────────────────
function run_case(A_cpu, true_rk, device_label, B, MAXRANK, TOL;
                  to_device = identity)
    n = size(A_cpu, 1)
    @printf("  [%s]  n=%d  b=%d  maxrank=%d  tol=%.1f\n",
            device_label, n, B, MAXRANK, TOL)

    A = to_device(A_cpu)
    gpu_sync()

    # warmup (JIT)
    Aw = TLRMatrix(A, B, MAXRANK)
    wsw = alloc_workspace(Aw)
    compress!(Aw, A, wsw);          gpu_sync()
    compress!(Aw, A, wsw; tol=TOL); gpu_sync()

    # ── Algorithm 1: cholqr2, fixed rank ─────────────────────────────────────
    T1 = TLRMatrix(A, B, MAXRANK); gpu_sync()
    ws1 = alloc_workspace(T1)
    t1 = @elapsed begin; compress!(T1, A, ws1); gpu_sync(); end
    e1 = rel_error(A_cpu, T1)
    ok1 = !any(isnan, Array(T1.int_U))

    # ── Algorithm 2: cholqr + NS + adaptive truncation ───────────────────────
    T2 = TLRMatrix(A, B, MAXRANK); gpu_sync()
    ws2 = alloc_workspace(T2)
    t2 = @elapsed begin; compress!(T2, A, ws2; tol=TOL); gpu_sync(); end
    e2  = rel_error(A_cpu, T2)
    rk2 = Array(ranks(T2))
    ok2 = !any(isnan, Array(T2.int_U))

    noff  = noffdiag_tiles(T2)
    exact = sum(1:noff) do ob
        lin  = NextLA.TLRmodule._linear_from_offdiag(T2, ob)
        i, j = NextLA.TLRmodule._inverse_tile_index(T2, lin)
        rk2[ob] == true_rk[i, j]
    end

    @printf("    Alg1 cholqr2   %8.4f s  rel_err=%.2e  ok=%-5s  rank=%d (fixed)\n",
            t1, e1, ok1, MAXRANK)
    print_ortho_summary("Alg1", T1)

    @printf("    Alg2 cholqr+NS %8.4f s  rel_err=%.2e  ok=%-5s  ranks∈[%d,%d]  recovery=%d/%d\n",
            t2, e2, ok2, minimum(rk2), maximum(rk2), exact, noff)
    print_ortho_summary("Alg2", T2)

    @printf("    Alg2 speedup vs Alg1: %.2fx\n", t1/t2)
end

# ── per-tile orthogonality errors for U and V ────────────────────────────────
function ortho_errors(A_tlr::TLRMatrix)
    rk      = Array(ranks(A_tlr))
    noff    = noffdiag_tiles(A_tlr)
    u_errs  = Float64[]
    for ob in 1:noff
        kr = Int(rk[ob])
        kr == 0 && continue
        Uob = Array(tile_u(A_tlr, ob))
        Ikr = Matrix{eltype(Uob)}(I, kr, kr)
        push!(u_errs, norm(Uob' * Uob - Ikr))
    end
    return u_errs
end

function print_ortho_summary(label::AbstractString, A_tlr::TLRMatrix)
    u_errs = ortho_errors(A_tlr)

    @printf("      %-4s U ortho ||UᵀU-I||_F: min=%.2e  median=%.2e  max=%.2e\n",
            label, minimum(u_errs), median(u_errs), maximum(u_errs))
end

# ── main ──────────────────────────────────────────────────────────────────────
const B       = 128
const MAXRANK = 32
const TOL     = 1.0f0   # Frobenius budget for removed columns; null columns
                         # have V-norms ≈ 0 so any TOL > 0 removes them

println("NextLA TLR compress! benchmark")
println("  tile size b=$B  maxrank=$MAXRANK  tol=$TOL")
println("  off-diagonal tile ranks: 10..20 (random)")
if HAS_CUDA
    println("  GPU: ", CUDA.name(CUDA.device()))
else
    println("  GPU: not available (run with ../gpuenv to enable CUDA)")
end
println()

println("view alloc check: ",
        @allocated(view(zeros(Float32,256,256), 1:128, 1:128)),
        " B (vs ", @allocated(copy(zeros(Float32,128,128))), " B for data copy)")
println()

for n in [2046, 4094, 8190]
    @printf("═══ n = %d ════════════════════════════════════════════\n", n)

    t_gen = @elapsed A_cpu, true_rk = generate_tiled_lowrank(n, B)
    @printf("  matrix generated in %.3f s  (%.0f MB)\n\n", t_gen, n^2*4/1e6)

    if n<=8192
        run_case(A_cpu, true_rk, "CPU", B, MAXRANK, TOL)
    end

    if HAS_CUDA
        println()
        run_case(A_cpu, true_rk, "GPU", B, MAXRANK, TOL; to_device=CuArray)
    end

    println()
end
