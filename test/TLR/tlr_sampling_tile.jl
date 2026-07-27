# RangeFind on one tile: the batched ARA core (ara_build_basis!/ara_truncate!)
# driven through the implicit sampler (apply_right!/apply_left!). No exact
# residual is formed on this path (docs/TODO.md, worklog item 4).

@testset "RangeFind on one tile" begin
    T = Float64
    rng = MersenneTwister(9101)

    @testset "recovers a controlled-rank tile" begin
        # q_k = 1, r_A small: X_ij = alpha * A_i1 * B_1j has rank <= r_A
        # generically, well below the output capacity -- ARA must converge
        # early and truncation must land on the true rank, not the cap.
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qm=2, qn=2, qk=1,
                                           bm=20, bk=8, bn=24, rA=3, rB=8)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        i, j = 1, 2
        alpha = T(1.4)
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, alpha)
        @test rank(Xref) <= 3

        maxrank = 15
        U = zeros(T, 20, maxrank, 1)
        V = zeros(T, 24, maxrank, 1)
        ranks = zeros(Int32, 1)
        err_sq = zeros(Float64, 1)
        info = _TLRM.range_find_tile!(U, V, ranks, err_sq, ops, i, j;
                                      alpha, eps_rel=1e-7, r_required=4,
                                      tol=1e-6, rel=true, block=4)
        r = Int(ranks[1])
        @test r == rank(Xref)
        @test r < maxrank
        Ur, Vr = U[:, 1:r, 1], V[:, 1:r, 1]
        @test norm(Ur' * Ur - I, Inf) <= 1e-11
        @test norm(Xref - Ur * Vr') / norm(Xref) <= 1e-6
        @test sqrt(err_sq[1]) <= 1e-6 * norm(Xref) * (1 + 1e-6)
        # Columns beyond the achieved rank are zero -- the zero-pad invariant.
        @test all(iszero, U[:, (r+1):maxrank, 1])
        @test all(iszero, V[:, (r+1):maxrank, 1])
        @test info.passes >= 1
    end

    @testset "beta != 0 folds C's tile into the found range" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qm=2, qn=2, qk=1,
                                           bm=16, bk=6, bn=18, rA=2, rB=6)
        C_tlr = NextLA.TLRMatrix(zeros(T, 2 * 16, 2 * 18), (16, 18), 4)
        fill_random_tlr!(C_tlr, Array; seed=931)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        Cdense = reconstruct_tlr(C_tlr)
        LA, LB, LC = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr),
                     _TLRM.logical_operand(C_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        alpha, beta = T(0.9), T(-0.5)
        i, j = 2, 1
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, alpha;
                               Cdense, C_tlr, beta)

        maxrank = 16
        U = zeros(T, 16, maxrank, 1)
        V = zeros(T, 18, maxrank, 1)
        ranks = zeros(Int32, 1)
        err_sq = zeros(Float64, 1)
        _TLRM.range_find_tile!(U, V, ranks, err_sq, ops, i, j;
                              alpha, beta, C=LC, eps_rel=1e-7, r_required=4,
                              tol=1e-6, rel=true, block=4)
        r = Int(ranks[1])
        Ur, Vr = U[:, 1:r, 1], V[:, 1:r, 1]
        @test norm(Ur' * Ur - I, Inf) <= 1e-11
        @test norm(Xref - Ur * Vr') / norm(Xref) <= 1e-6
    end

    @testset "a full-rank tile saturates the output cap" begin
        # Random, unstructured factors at both operands' full capacity: X_ij
        # is generically full rank, so the ARA loop must run to maxrank and
        # ara_truncate! must report the cap as the achieved rank.
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qm=1, qn=1, qk=1,
                                           bm=12, bk=12, bn=12, rA=12, rB=12)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        i, j = 1, 1
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, T(1))
        @test rank(Xref) == 12

        maxrank = 8   # capacity below the tile's true rank
        U = zeros(T, 12, maxrank, 1)
        V = zeros(T, 12, maxrank, 1)
        ranks = zeros(Int32, 1)
        err_sq = zeros(Float64, 1)
        _TLRM.range_find_tile!(U, V, ranks, err_sq, ops, i, j;
                              alpha=T(1), eps_rel=1e-7, r_required=4,
                              tol=1e-12, rel=true, block=4)
        @test ranks[1] == maxrank            # saturation: the signal to act on
        U1, V1 = U[:, :, 1], V[:, :, 1]
        @test norm(U1' * U1 - I, Inf) <= 1e-10   # still orthonormal at the cap
        # err_sq is *not* a trustworthy error bound here: with no `energy`
        # passed (computing ‖X_ij‖² exactly needs the O(q_k²) cross terms
        # worklog item 4 rejects on the hot path), ara_truncate! measures the
        # error only within the 8-dim basis it was given, not against the true
        # 12-dim range. That basis is generically a near-perfect fit for
        # itself, so err_sq underreports by orders of magnitude -- `ranks[1]
        # == maxrank` is the signal a caller must act on, not this number.
        true_err = norm(Xref - U1 * V1') / norm(Xref)
        @test true_err > 1e-3
        @test sqrt(err_sq[1]) < 1e-6 * norm(Xref)
    end

    @testset "eps_rel below the Cholesky-QR floor is rejected" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qm=1, qn=1, qk=1,
                                           bm=10, bk=4, bn=10, rA=3, rB=3)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        U = zeros(T, 10, 5, 1); V = zeros(T, 10, 5, 1)
        ranks = zeros(Int32, 1); err_sq = zeros(Float64, 1)
        @test_throws ArgumentError _TLRM.range_find_tile!(
            U, V, ranks, err_sq, ops, 1, 1;
            alpha=T(1), eps_rel=1e-12, tol=1e-8)
    end

    @testset "maxrank = 0 gives an all-zero tile with no error" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qm=1, qn=1, qk=1,
                                           bm=8, bk=3, bn=8, rA=2, rB=2)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        U = fill(T(NaN), 8, 0, 1); V = fill(T(NaN), 8, 0, 1)
        ranks = fill(Int32(-1), 1); err_sq = fill(-1.0, 1)
        info = _TLRM.range_find_tile!(U, V, ranks, err_sq, ops, 1, 1;
                                      alpha=T(1), eps_rel=1e-6, tol=1e-6)
        @test ranks[1] == 0
        @test err_sq[1] == 0.0
        @test info.passes == 0
    end
end

@testset "RangeFind on one tile on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            T = Float64
            rng = MersenneTwister(9102)
            A_tlr, B_tlr = _tile_apply_fixture(T, ArrayType; qm=2, qn=2, qk=1,
                                               bm=20, bk=8, bn=24, rA=3, rB=8)
            Adense = reconstruct_tlr(A_tlr)
            Bdense = reconstruct_tlr(B_tlr)
            LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
            ops = _TLRM.logical_operands(LA, LB)
            i, j = 1, 2
            alpha = T(1.4)
            Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, alpha)

            maxrank = 15
            U = ArrayType(zeros(T, 20, maxrank, 1))
            V = ArrayType(zeros(T, 24, maxrank, 1))
            ranks = ArrayType(zeros(Int32, 1))
            err_sq = ArrayType(zeros(Float64, 1))
            _TLRM.range_find_tile!(U, V, ranks, err_sq, ops, i, j;
                                   alpha, eps_rel=1e-7, r_required=4,
                                   tol=1e-6, rel=true, block=4)
            synchronize(U)
            r = Int(Array(ranks)[1])
            @test r == rank(Xref)
            Uh, Vh = Array(U), Array(V)
            Ur, Vr = Uh[:, 1:r, 1], Vh[:, 1:r, 1]
            @test norm(Ur' * Ur - I, Inf) <= 1e-9
            @test norm(Xref - Ur * Vr') / norm(Xref) <= 1e-6
        end
    end
end
