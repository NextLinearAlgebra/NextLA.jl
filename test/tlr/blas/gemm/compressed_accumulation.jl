# gemm of two CompressedFTLRMatrix operands into a compressed result,
# src/tlr/blas/gemm/compressed_accumulation.
# Helpers: test/setup/tlr.jl.

function _compressed_output_fixture(::Type{T}, ArrayType;
                                    n=48, b=16, rA=3, rB=4, seed=1200) where {T}
    backend = KernelAbstractions.get_backend(ArrayType(zeros(T, 1)))
    q = cld(n, b)

    A = TileLowRank.CompressedFTLRMatrix(backend, T, n, n, b, fill(rA, q, q))
    B = TileLowRank.CompressedFTLRMatrix(backend, T, n, n, b, fill(rB, q, q))
    fill_random_tlr!(A, ArrayType; seed=seed+1)
    fill_random_tlr!(B, ArrayType; seed=seed+2)
    return A, B
end

function _check_compressed_output_gemm(ArrayType, synchronize;
                                       transA='N', transB='N',
                                       rA=3, rB=4, seed=1200,
                                       rank_multiple=0)
    T = Float64
    A, B = _compressed_output_fixture(T, ArrayType; rA, rB, seed)
    op(X, t) = uppercase(t) == 'T' ? transpose(X) : X
    reference = 1.2 .* (op(reconstruct_tlr(A), transA) *
                        op(reconstruct_tlr(B), transB))
    C = NextLA.gemm(
        A, B; maxrank=16, rank_multiple, alpha=1.2, transA, transB,
        tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4)
    synchronize(C.outer.data)
    got = reconstruct_tlr(C)
    @test norm(got-reference) / max(norm(reference), eps(T)) <= 2e-6
    @test all(TileLowRank.ranks(C) .<= 16)
    @test TileLowRank.rank_multiple(C) == rank_multiple
    return C, A, B
end

@testset "allocation-returning compressed-output GEMM [CPU]" begin
    Random.seed!(3037)
    for (transA, transB) in (('N','N'), ('N','T'), ('T','N'), ('T','T'))
        _check_compressed_output_gemm(
            Array, _ -> nothing; transA, transB,
            rA=transA == 'N' ? 2 : 5,
            rB=transB == 'N' ? 4 : 3,
            seed=1300 + Int(transA) + Int(transB))
    end
    C, A, B = _check_compressed_output_gemm(
        Array, _ -> nothing; rank_multiple=8, seed=1410)
    @test TileLowRank.maximum_storage_rank(C) % 8 == 0

    # Finalized compressed output cannot be resized in place.
    @test_throws MethodError NextLA.gemm!(C, A, B; tol=1e-7)
    @test_throws ArgumentError NextLA.gemm(A, B; maxrank=-1)
    @test_throws ArgumentError NextLA.gemm(A, B; maxrank=17)
    @test_throws ArgumentError NextLA.gemm(A, B; maxrank=16, r_required=0)
end

@testset "compressed-output GEMM regular-grid boundary [CPU]" begin
    Random.seed!(3038)
    ranks = ones(Int, 3, 3)
    A = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, ranks)
    B = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), Float64, 10, 10, 4, ranks)
    @test_throws ArgumentError NextLA.gemm(A, B; maxrank=4)
end

@testset "compressed-output GEMM rolling workspace bounds [CPU]" begin
    Random.seed!(3039)
    T = Float64
    n, b, rA, rB, maxrank, block = 48, 16, 2, 2, 12, 4
    A, B = _compressed_output_fixture(T, Array; n, b, rA, rB, seed=1500)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)

    # `maxrank` must clear the coupling's true rank (qk * min(rA, rB) = 3*2 =
    # 6 here) or the comparison below is checking a lossy truncation, not
    # exact recovery, against too tight a tolerance.

    # the same staging shape `gemm` builds internally, used only to size a
    # capacity=1 workspace: every member then gets its own rolling-admission
    # wave (cold start, then repeated backfill) instead of the default single
    # wave sized for every member at once.
    qm, qn = TileLowRank.grid_size(A)[1], TileLowRank.grid_size(B)[2]
    staging = TileLowRank.CompressedFTLRMatrix(
        KernelAbstractions.CPU(), T, n, n, (b, b), fill(maxrank, qm, qn);
        outer_order=TileLowRank.TileRowMajor, inner_order=TileLowRank.TileColMajor)
    spec = _TLRM.tlr_gemm_workspace_spec(staging, A, B; transA='N', transB='N', block)
    tight = _TLRM.tlr_gemm_workspace_bytes(spec, 1)

    C_rolling = NextLA.gemm(
        A, B; maxrank, tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block,
        workspace=tight)
    C_single = NextLA.gemm(
        A, B; maxrank, tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block)

    @test norm(reconstruct_tlr(C_rolling) - reference) / norm(reference) <= 2e-6
    @test reconstruct_tlr(C_rolling) ≈ reconstruct_tlr(C_single) rtol=1e-8
    @test_throws ArgumentError NextLA.gemm(A, B; maxrank, workspace=tight - 1)
end

# Product tiles of rank 16 (qk * rA = 4 * 8, capped by the tile size) with
# maxrank 8: every tile stops at the cap, and residuals(C) must report what the
# cap leaves out, not only the truncation (which is zero at the cap). The
# estimate's squared value has relative standard deviation at most √(2/w) with
# w = 8 Gaussian vectors (src/tlr/blas/gemm/compressed_accumulation/
# rolling_schedule.jl), so a factor of 2 on the norm is several deviations wide.
function _check_capped_residuals(ArrayType, synchronize)
    n, b, maxrank = 64, 16, 8
    A, B = _compressed_output_fixture(Float64, ArrayType; n, b, rA=8, rB=8, seed=1600)
    C = NextLA.gemm(A, B; maxrank, tol=1e-10, rel=true)
    synchronize(C.outer.data)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)
    got = reconstruct_tlr(C)
    @test all(==(maxrank), TileLowRank.ranks(C))
    qm, qn = TileLowRank.grid_size(C)
    for j in 1:qn, i in 1:qm
        rows, cols = ((i - 1) * b + 1):(i * b), ((j - 1) * b + 1):(j * b)
        actual = norm(got[rows, cols] - reference[rows, cols])
        reported = TileLowRank.residuals(C)[expected_storage_slot(C, i, j)]
        @test actual / 2 <= reported <= 2 * actual
    end
end

@testset "compressed-output GEMM residuals at the rank cap [CPU]" begin
    Random.seed!(3044)
    _check_capped_residuals(Array, _ -> nothing)
end

@testset "compressed-output GEMM residuals at the rank cap [$backend_name]" for (backend_name, ArrayType, synchronize) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3045)
    _check_capped_residuals(ArrayType, synchronize)
end

@testset "compressed-output GEMM [$backend_name]" for (backend_name, ArrayType, synchronize) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3040)
    A, B = _compressed_output_fixture(Float32, ArrayType; n=32, b=16)
    C = NextLA.gemm(
        A, B; maxrank=16, rank_multiple=8,
        tol=1f-4, rel=true, eps_rel=1f-4, r_required=3, block=4)
    synchronize(C.outer.data)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)
    @test norm(reconstruct_tlr(C)-reference) / norm(reference) <= 5f-3
end
