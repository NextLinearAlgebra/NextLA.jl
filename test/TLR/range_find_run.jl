# R3: fixed-column RangeFind with a shared H pass and swap-compacted active
# members. Effective ranks are deliberately ordered low-to-high so early
# convergence creates holes rather than an already-retired suffix.

function _column_run_fixture(::Type{T}, ArrayType) where {T}
    qm, qk, qn = 3, 1, 2
    bm, bk, bn = 24, 14, 26
    rA = rB = 12
    A, B = _tile_apply_fixture(T, ArrayType; qm, qn, qk, bm, bk, bn, rA, rB)
    LA = _TLRM.logical_operand(A)

    # The container capacity stays uniform, but zero tails give the three
    # members distinct numerical ranks and therefore distinct pass counts.
    for (i, r) in enumerate((2, 7, 11))
        Ua, Va = _TLRM.logical_tile_factors(LA, i, 1)
        r < rA && begin
            fill!(view(Ua, :, (r + 1):rA), zero(T))
            fill!(view(Va, :, (r + 1):rA), zero(T))
        end
    end
    return A, B
end

function _check_column_run(::Type{T}, ArrayType, synchronize; with_beta=false) where {T}
    A, B = _column_run_fixture(T, ArrayType)
    Ad, Bd = reconstruct_tlr(A), reconstruct_tlr(B)
    LA, LB = _TLRM.logical_operand(A), _TLRM.logical_operand(B)
    ops = _TLRM.logical_operands(LA, LB)
    rows, j = 1:3, 2
    alpha = T(1.25)
    beta = with_beta ? T(-0.35) : zero(T)
    C = if with_beta
        x = NextLA.TLRMatrix(ArrayType(zeros(T, 3 * 24, 2 * 26)), (24, 26), 3;
                             tile_order=NextLA.TileColMajor())
        fill_random_tlr!(x, ArrayType; seed=9142)
        x
    else
        nothing
    end
    LC = C === nothing ? nothing : _TLRM.logical_operand(C)
    Cd = C === nothing ? nothing : reconstruct_tlr(C)
    maxrank = 16

    U = ArrayType(zeros(T, 24, maxrank, length(rows)))
    V = ArrayType(zeros(T, 26, maxrank, length(rows)))
    ranks = ArrayType(zeros(Int32, length(rows)))
    err = ArrayType(zeros(Float64, length(rows)))
    info = _TLRM.range_find_column_run!(
        U, V, ranks, err, ops, rows, j;
        alpha, beta, C=LC, eps_rel=1e-7, r_required=3,
        tol=1e-6, rel=true, block=4,
    )
    synchronize(U)

    Uh, Vh = Array(U), Array(V)
    rh = Int.(Array(ranks))
    @test info.active_counts[1] == length(rows)
    @test issorted(info.active_counts; rev=true)
    @test info.active_counts[end] < info.active_counts[1]
    @test sort(info.member_ids) == collect(1:length(rows))
    @test info.member_ids != collect(1:length(rows)) # a real swap occurred

    for (slot, i) in enumerate(rows)
        X = _tile_dense_ref(Ad, Bd, A, B, i, j, alpha;
                            Cdense=Cd, C_tlr=C, beta)
        r = rh[slot]
        @test r == rank(X)
        @test norm(X - Uh[:, 1:r, slot] * Vh[:, 1:r, slot]') / norm(X) <= 1e-6
        @test norm(Uh[:, 1:r, slot]' * Uh[:, 1:r, slot] - I, Inf) <= 1e-9
        @test all(iszero, Uh[:, (r + 1):maxrank, slot])
        @test all(iszero, Vh[:, (r + 1):maxrank, slot])
    end
end

@testset "RangeFind packed column run (R3)" begin
    _check_column_run(Float64, Array, _ -> nothing)
    _check_column_run(Float64, Array, _ -> nothing; with_beta=true)
end

@testset "RangeFind packed column run on GPU (R3)" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        @testset "$backend_name" begin
            _check_column_run(Float64, ArrayType, synchronize)
            _check_column_run(Float64, ArrayType, synchronize; with_beta=true)
        end
    end
end
