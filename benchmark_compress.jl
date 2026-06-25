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
using NextLA: TLRMatrix, compress!, left_factors, right_factors, dense_diag, ranks,
              ndiag_tiles, noffdiag_tiles, tile_origin_coords, tile_sizes,
              offdiag_linear_index, inverse_tile_index

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
    layout = A_tlr.layout
    # pull factors to CPU (no-op if already CPU)
    U  = Array(left_factors(A_tlr))
    V  = Array(right_factors(A_tlr))
    D  = Array(dense_diag(A_tlr))
    rk = Array(ranks(A_tlr))

    err_sq = norm_sq = 0.0
    for ob in 1:noffdiag_tiles(layout)
        lin = offdiag_linear_index(layout, ob)
        i, j = inverse_tile_index(layout, lin)
        p0, q0 = tile_origin_coords(layout, i, j)
        tm, tn = tile_sizes(layout, i, j);  kr = rk[ob]
        tile  = A_cpu[p0:p0+tm-1, q0:q0+tn-1]
        recon = view(U,1:tm,1:kr,ob) * view(V,1:tn,1:kr,ob)'
        err_sq  += sum(abs2, tile - recon)
        norm_sq += sum(abs2, tile)
    end
    for k in 1:ndiag_tiles(layout)
        p0, q0 = tile_origin_coords(layout, k, k)
        tm, tn = tile_sizes(layout, k, k)
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
    compress!(Aw, A);          gpu_sync()
    compress!(Aw, A; tol=TOL); gpu_sync()

    # ── Algorithm 1: cholqr2, fixed rank ─────────────────────────────────────
    T1 = TLRMatrix(A, B, MAXRANK); gpu_sync()
    t1 = @elapsed begin; compress!(T1, A); gpu_sync(); end
    e1 = rel_error(A_cpu, T1)
    ok1 = !any(isnan, Array(left_factors(T1)))

    # ── Algorithm 2: cholqr + NS + adaptive truncation ───────────────────────
    T2 = TLRMatrix(A, B, MAXRANK); gpu_sync()
    t2 = @elapsed begin; compress!(T2, A; tol=TOL); gpu_sync(); end
    e2  = rel_error(A_cpu, T2)
    rk2 = Array(ranks(T2))
    ok2 = !any(isnan, Array(left_factors(T2)))

    layout = T2.layout
    noff = noffdiag_tiles(layout)
    exact = sum(1:noff) do ob
        lin = offdiag_linear_index(layout, ob)
        i, j = inverse_tile_index(layout, lin)
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
    layout = A_tlr.layout

    # Pull factors/ranks to CPU. No-op if already CPU.
    U  = Array(left_factors(A_tlr))
    V  = Array(right_factors(A_tlr))
    rk = Array(ranks(A_tlr))

    noff = noffdiag_tiles(layout)

    u_errs = Float64[]
    v_errs = Float64[]

    for ob in 1:noff
        lin = offdiag_linear_index(layout, ob)
        i, j = inverse_tile_index(layout, lin)

        tm, tn = tile_sizes(layout, i, j)
        kr = Int(rk[ob])

        kr == 0 && continue

        Uob = @view U[1:tm, 1:kr, ob]

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

for n in [2048, 4096, 8192]
    @printf("═══ n = %d ════════════════════════════════════════════\n", n)

    t_gen = @elapsed A_cpu, true_rk = generate_tiled_lowrank(n, B)
    @printf("  matrix generated in %.3f s  (%.0f MB)\n\n", t_gen, n^2*4/1e6)

    run_case(A_cpu, true_rk, "CPU", B, MAXRANK, TOL)

    if HAS_CUDA
        println()
        run_case(A_cpu, true_rk, "GPU", B, MAXRANK, TOL; to_device=CuArray)
    end

    println()
end
