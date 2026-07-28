# The factor-list sampler for one output tile -- ApplyRight/ApplyLeft
# against a dense reference, without ever materializing the output tile.

# Dense reference for the update of tile (i,j): alpha * (A's tile-row i) *
# (B's tile-col j) [+ beta * C's tile (i,j)], as a b_m x b_n matrix, taken
# directly from the whole-matrix dense reconstruction.
function _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i::Int, j::Int, alpha;
                         Cdense=nothing, C_tlr=nothing, beta=false)
    bm = NextLA.nominal_tile_size(A_tlr, 1)
    bn = NextLA.nominal_tile_size(B_tlr, 2)
    rows = ((i - 1) * bm + 1):(i * bm)
    cols = ((j - 1) * bn + 1):(j * bn)
    X = alpha * Adense[rows, :] * Bdense[:, cols]
    if !iszero(beta)
        cbm = NextLA.nominal_tile_size(C_tlr, 1)
        cbn = NextLA.nominal_tile_size(C_tlr, 2)
        crows = ((i - 1) * cbm + 1):(i * cbm)
        ccols = ((j - 1) * cbn + 1):(j * cbn)
        X = X + beta * Cdense[crows, ccols]
    end
    return X
end

function _tile_apply_fixture(::Type{T}, ArrayType; qm=3, qn=3, qk=4,
                             bm=4, bk=3, bn=5, rA=3, rB=2) where {T}
    RM = NextLA.TileRowMajor(); CM = NextLA.TileColMajor()
    A_tlr = NextLA.PaddedFTLRMatrix(ArrayType(zeros(T, qm * bm, qk * bk)), (bm, bk), rA;
                            tile_order=RM)
    B_tlr = NextLA.PaddedFTLRMatrix(ArrayType(zeros(T, qk * bk, qn * bn)), (bk, bn), rB;
                            tile_order=CM)
    fill_random_tlr!(A_tlr, ArrayType; seed=811)
    fill_random_tlr!(B_tlr, ArrayType; seed=812)
    return A_tlr, B_tlr
end

@testset "tile factor-list sampler" begin
    T = Float64
    rng = MersenneTwister(9001)

    @testset "ApplyRight/ApplyLeft match the dense reference, beta=0" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        alpha = T(1.7)

        for (i, j) in ((1, 1), (2, 3), (3, 2))
            cpl, bt = _TLRM.tile_factor_list(ops, i, j; alpha, beta=false)
            @test bt === nothing
            Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, alpha)

            s = 6
            Omega = randn(rng, T, size(Xref, 2), s)
            Y = zeros(T, size(Xref, 1), s)
            _TLRM.apply_right!(Y, cpl, bt, Omega)
            @test Y ≈ Xref * Omega rtol = 1e-10

            s2 = 4
            Q = randn(rng, T, size(Xref, 1), s2)
            Z = zeros(T, size(Xref, 2), s2)
            _TLRM.apply_left!(Z, cpl, bt, Q)
            @test Z ≈ Xref' * Q rtol = 1e-10

            # Self-consistency, independent of the dense reference:
            # Yᵀ Q = Ωᵀ Xᵀ Q = Ωᵀ Z.
            @test Y' * Q ≈ Omega' * Z rtol = 1e-10
        end
    end

    @testset "beta != 0 folds in C's own tile" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        C_tlr = NextLA.PaddedFTLRMatrix(zeros(T, 3 * 4, 3 * 5), (4, 5), 5)
        fill_random_tlr!(C_tlr, Array; seed=813)
        Cdense = reconstruct_tlr(C_tlr)
        LA, LB, LC = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr),
                     _TLRM.logical_operand(C_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        alpha, beta = T(0.6), T(-1.3)

        i, j = 2, 2
        cpl, bt = _TLRM.tile_factor_list(ops, i, j; alpha, beta, C=LC)
        @test bt !== nothing
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, alpha;
                               Cdense, C_tlr, beta)

        s = 5
        Omega = randn(rng, T, size(Xref, 2), s)
        Y = zeros(T, size(Xref, 1), s)
        _TLRM.apply_right!(Y, cpl, bt, Omega)
        @test Y ≈ Xref * Omega rtol = 1e-10

        Q = randn(rng, T, size(Xref, 1), s)
        Z = zeros(T, size(Xref, 2), s)
        _TLRM.apply_left!(Z, cpl, bt, Q)
        @test Z ≈ Xref' * Q rtol = 1e-10

        # beta=0 must require no C: the common case shouldn't force a C operand.
        @test_throws ArgumentError _TLRM.tile_factor_list(
            ops, i, j; alpha, beta=T(1))
    end

    @testset "zero-rank operand gives a correct zero (or pure-beta) result" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; rA=0)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        i, j = 1, 2
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, T(1))
        @test all(iszero, Xref)   # rA=0 => A is identically zero

        cpl, bt = _TLRM.tile_factor_list(ops, i, j; alpha=T(1), beta=false)
        Omega = randn(rng, T, size(Xref, 2), 3)
        Y = fill(T(NaN), size(Xref, 1), 3)   # would expose a missed branch
        _TLRM.apply_right!(Y, cpl, bt, Omega)
        @test all(iszero, Y)

        Q = randn(rng, T, size(Xref, 1), 3)
        Z = fill(T(NaN), size(Xref, 2), 3)
        _TLRM.apply_left!(Z, cpl, bt, Q)
        @test all(iszero, Z)
    end

    @testset "rectangular, non-square tiles" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qm=2, qn=2, qk=3,
                                           bm=6, bk=2, bn=3, rA=2, rB=4)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        i, j = 2, 1
        cpl, bt = _TLRM.tile_factor_list(ops, i, j; alpha=T(1), beta=false)
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, T(1))
        @test size(Xref) == (6, 3)

        Omega = randn(rng, T, 3, 4)
        Y = zeros(T, 6, 4)
        _TLRM.apply_right!(Y, cpl, bt, Omega)
        @test Y ≈ Xref * Omega rtol = 1e-10

        Q = randn(rng, T, 6, 4)
        Z = zeros(T, 3, 4)
        _TLRM.apply_left!(Z, cpl, bt, Q)
        @test Z ≈ Xref' * Q rtol = 1e-10
    end

    @testset "single contraction tile (q_k = 1)" begin
        A_tlr, B_tlr = _tile_apply_fixture(T, Array; qk=1)
        Adense, Bdense = reconstruct_tlr(A_tlr), reconstruct_tlr(B_tlr)
        LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
        ops = _TLRM.logical_operands(LA, LB)
        i, j = 1, 1
        cpl, bt = _TLRM.tile_factor_list(ops, i, j; alpha=T(1), beta=false)
        Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, T(1))

        Omega = randn(rng, T, size(Xref, 2), 2)
        Y = zeros(T, size(Xref, 1), 2)
        _TLRM.apply_right!(Y, cpl, bt, Omega)
        @test Y ≈ Xref * Omega rtol = 1e-10
    end
end

@testset "tile factor-list sampler on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            T = Float64
            rng = MersenneTwister(9002)
            A_tlr, B_tlr = _tile_apply_fixture(T, ArrayType)
            Adense = reconstruct_tlr(A_tlr)   # host reference, from device factors
            Bdense = reconstruct_tlr(B_tlr)
            LA, LB = _TLRM.logical_operand(A_tlr), _TLRM.logical_operand(B_tlr)
            ops = _TLRM.logical_operands(LA, LB)
            alpha = T(1.3)
            i, j = 2, 2
            cpl, bt = _TLRM.tile_factor_list(ops, i, j; alpha, beta=false)
            Xref = _tile_dense_ref(Adense, Bdense, A_tlr, B_tlr, i, j, alpha)

            s = 5
            Omega = ArrayType(randn(rng, T, size(Xref, 2), s))
            Y = ArrayType(zeros(T, size(Xref, 1), s))
            _TLRM.apply_right!(Y, cpl, bt, Omega)
            synchronize(Y)
            @test Array(Y) ≈ Xref * Array(Omega) rtol = 1e-9

            Q = ArrayType(randn(rng, T, size(Xref, 1), s))
            Z = ArrayType(zeros(T, size(Xref, 2), s))
            _TLRM.apply_left!(Z, cpl, bt, Q)
            synchronize(Z)
            @test Array(Z) ≈ Xref' * Array(Q) rtol = 1e-9
        end
    end
end
