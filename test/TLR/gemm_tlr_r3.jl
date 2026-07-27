function _canonical_tlr_fixture(::Type{T}, ArrayType;
                                rA=3, rB=4, rC=16, seed=1200) where {T}
    n, b = 48, 16
    A = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, rA)
    B = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, rB)
    C = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, rC)
    fill_random_tlr!(A, ArrayType; seed=seed + 1)
    fill_random_tlr!(B, ArrayType; seed=seed + 2)
    return A, B, C
end

function _check_canonical_tlr_gemm(ArrayType, synchronize;
                                   transA='N', transB='N',
                                   rA=3, rB=4, beta=0.0, seed=1200)
    T = Float64
    A, B, C = _canonical_tlr_fixture(T, ArrayType; rA, rB, seed)
    if !iszero(beta)
        fill_random_tlr!(C, ArrayType; seed=seed + 3)
    end
    Ad, Bd, C0 = reconstruct_tlr(A), reconstruct_tlr(B), reconstruct_tlr(C)
    op(X, t) = uppercase(t) == 'T' ? transpose(X) : X
    alpha = T(1.2)
    ref = alpha * op(Ad, transA) * op(Bd, transB) + T(beta) * C0

    NextLA.TLRmodule.gemm!(
        C, A, B; alpha, beta=T(beta), transA, transB,
        tol=1e-7, rel=true, eps_rel=1e-7, r_required=3, block=4,
    )
    synchronize(C.int_U)
    got = reconstruct_tlr(C)
    @test norm(got - ref) / max(norm(ref), eps(T)) <= 2e-6
    @test all(NextLA.ranks(C) .<= NextLA.maxrank(C))
    return C, A, B
end

@testset "canonical row-major TLR-result gemm! (R3)" begin
    A, B, C = _canonical_tlr_fixture(Float64, Array)
    @test _TLRM.tile_order(A) isa NextLA.TileRowMajor
    @test _TLRM.tile_order(B) isa NextLA.TileRowMajor
    @test _TLRM.tile_order(C) isa NextLA.TileRowMajor

    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='N')
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='N',
                              beta=-0.3, seed=1210)
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='T',
                              rA=2, rB=6, seed=1220) # rank cost chooses right
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='N', transB='T',
                              rA=6, rB=2, seed=1230) # rank cost chooses left
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='T',
                              seed=1240)
    _check_canonical_tlr_gemm(Array, _ -> nothing; transA='T', transB='T',
                              beta=0.25, seed=1245)

    LA = _TLRM.logical_operand(A, 'N')
    LBT = _TLRM.logical_operand(B, 'T')
    @test _TLRM.choose_tlr_sampling_side(LA, LBT, 16, 4, 2, 6) == :right
    @test _TLRM.choose_tlr_sampling_side(LA, LBT, 16, 4, 6, 2) == :left

    @test_throws ArgumentError _TLRM.gemm!(
        C, A, B; transA='T', transB='N', tol=1e-6)

    Acol = NextLA.TLRMatrix(zeros(Float64, 48, 48), 16, 3;
                            tile_order=NextLA.TileColMajor())
    @test_throws ArgumentError _TLRM.gemm!(C, Acol, B; tol=1e-6)
end

@testset "canonical row-major TLR-result gemm! on CUDA (R3)" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='N', seed=1250)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='N', transB='T',
                rA=2, rB=6, seed=1260)
            _check_canonical_tlr_gemm(
                ArrayType, synchronize; transA='T', transB='T', seed=1270)
        end
    end
end
