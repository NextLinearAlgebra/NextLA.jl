@testset "row-basis end-to-end driver" begin
    rng = MersenneTwister(908)
    T = Float64
    A = NextLA.TLRMatrix(zeros(T, 12, 12), (4, 4), 2;
                           tile_order=NextLA.TileRowMajor())
    B = NextLA.TLRMatrix(zeros(T, 12, 12), (4, 4), 2;
                           tile_order=NextLA.TileColMajor())
    C = NextLA.TLRMatrix(zeros(T, 12, 12), (4, 4), 4)
    fill_random_tlr!(A, Array; seed=Int(rand(rng, 1:10^6)))
    fill_random_tlr!(B, Array; seed=Int(rand(rng, 1:10^6)))
    _TLRM._row_basis_gemm!(C, A, B; tol=0.0)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)
    @test reconstruct_tlr(C) ≈ reference atol=2e-9 rtol=2e-9
    for slot in axes(C.int_U, 3)
        rank = Int(C.ranks[slot])
        rank == 0 && continue
        @test norm(C.int_U[:, 1:rank, slot]' * C.int_U[:, 1:rank, slot] - I, Inf) <= 2e-10
    end

    C0 = reconstruct_tlr(C)
    _TLRM.gemm!(C, A, B; alpha=1.25, beta=-0.4, tol=0.0)
    @test reconstruct_tlr(C) ≈ 1.25 * reference - 0.4 * C0 atol=3e-9 rtol=3e-9
end

@testset "row-basis zero product preserves beta C" begin
    T = Float64
    A = NextLA.TLRMatrix(zeros(T, 8, 8), (4, 4), 0; tile_order=NextLA.TileRowMajor())
    B = NextLA.TLRMatrix(zeros(T, 8, 8), (4, 4), 2; tile_order=NextLA.TileColMajor())
    C = NextLA.TLRMatrix(zeros(T, 8, 8), (4, 4), 4)
    fill_random_tlr!(B, Array; seed=14)
    fill_random_tlr!(C, Array; seed=15)
    C0 = reconstruct_tlr(C)
    _TLRM.gemm!(C, A, B; alpha=1.0, beta=-0.6, tol=0.0)
    @test reconstruct_tlr(C) ≈ -0.6 * C0 atol=2e-10 rtol=2e-10
end

@testset "row-basis CPU layout fallbacks" begin
    T = Float64
    for (oa, ob) in ((NextLA.TileRowMajor(), NextLA.TileRowMajor()),
                     (NextLA.TileColMajor(), NextLA.TileRowMajor()),
                     (NextLA.TileColMajor(), NextLA.TileColMajor()))
        A = NextLA.TLRMatrix(zeros(T, 8, 8), (4, 4), 2; tile_order=oa)
        B = NextLA.TLRMatrix(zeros(T, 8, 8), (4, 4), 2; tile_order=ob)
        C = NextLA.TLRMatrix(zeros(T, 8, 8), (4, 4), 4)
        fill_random_tlr!(A, Array; seed=11)
        fill_random_tlr!(B, Array; seed=12)
        _TLRM.gemm!(C, A, B; tol=0.0)
        @test reconstruct_tlr(C) ≈ reconstruct_tlr(A) * reconstruct_tlr(B) atol=2e-9 rtol=2e-9
    end
end

@testset "row-basis beta across backends" begin
    T = Float64
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            A = NextLA.TLRMatrix(ArrayType(zeros(T, 12, 12)), (4, 4), 2; tile_order=NextLA.TileRowMajor())
            B = NextLA.TLRMatrix(ArrayType(zeros(T, 12, 12)), (4, 4), 2; tile_order=NextLA.TileColMajor())
            C = NextLA.TLRMatrix(ArrayType(zeros(T, 12, 12)), (4, 4), 4)
            fill_random_tlr!(A, ArrayType; seed=31); fill_random_tlr!(B, ArrayType; seed=32)
            fill_random_tlr!(C, ArrayType; seed=33); synchronize(C.int_U)
            ref = reconstruct_tlr(A) * reconstruct_tlr(B)
            C0 = reconstruct_tlr(C)
            NextLA.TLRmodule.gemm!(C, A, B; alpha=1.25, beta=-0.4, tol=0.0)
            synchronize(C.int_U)
            @test reconstruct_tlr(C) ≈ 1.25 * ref - 0.4 * C0 atol=3e-9 rtol=3e-9
        end
    end
end

@testset "row-basis rectangular (bm≠k≠bn) tiles" begin
    T = Float64
    # A tiles 4×3 (row=4, contraction=3), B tiles 3×5 (contraction=3, col=5),
    # C tiles 4×5. Exercises the distinct (bm, k, bn) dimensions end to end.
    A = NextLA.TLRMatrix(zeros(T, 8, 6), (4, 3), 2; tile_order=NextLA.TileRowMajor())
    B = NextLA.TLRMatrix(zeros(T, 6, 10), (3, 5), 2; tile_order=NextLA.TileColMajor())
    C = NextLA.TLRMatrix(zeros(T, 8, 10), (4, 5), 4)
    fill_random_tlr!(A, Array; seed=21); fill_random_tlr!(B, Array; seed=22)
    reference = reconstruct_tlr(A) * reconstruct_tlr(B)
    _TLRM._row_basis_gemm!(C, A, B; tol=0.0)
    @test reconstruct_tlr(C) ≈ reference atol=2e-9 rtol=2e-9

    C0 = reconstruct_tlr(C)
    _TLRM.gemm!(C, A, B; alpha=0.9, beta=0.5, tol=0.0)
    @test reconstruct_tlr(C) ≈ 0.9 * reference + 0.5 * C0 atol=3e-9 rtol=3e-9
end

# Direct numerical check of the batched Stage 2 (both contraction branches),
# replacing the deleted per-tile `accumulate_row_coefficients!` unit test.
@testset "batched row coefficients" begin
    rng = MersenneTwister(906)
    T = Float64
    kd, bn, K, qn, rA, rB = 5, 6, 3, 4, 2, 3
    B = NextLA.TLRMatrix(zeros(T, K * kd, qn * bn), (kd, bn), rB;
                           tile_order=NextLA.TileColMajor())
    fill_random_tlr!(B, Array; seed=61)
    BpU = _TLRM.interior_operand(_TLRM.FullGrid(), B.int_U, B.order, K, qn)
    BpV = _TLRM.interior_operand(_TLRM.FullGrid(), B.int_V, B.order, K, qn)
    for t in (1, 4)   # t <= rA and t > rA branches (rA = 2)
        Vrow = randn(rng, T, kd, rA, K)
        P = randn(rng, T, t, rA, K)
        Vm = zeros(T, bn, t, qn)
        _TLRM._accumulate_row_block!(Vm, Vrow, P, BpU, BpV, qn, T(1.3),
                                     _TLRM.default_gemm_compute_mode(T))
        for j in 1:qn
            ref = zeros(T, bn, t)
            for k in 1:K
                W = Array(_TLRM.tilefactor(BpU, k, j))
                Z = Array(_TLRM.tilefactor(BpV, k, j))
                ref .+= Z * (P[:, :, k] * (Vrow[:, :, k]' * W))'
            end
            @test Vm[:, :, j] ≈ 1.3 .* ref rtol=1e-10 atol=1e-10
        end
    end
end
