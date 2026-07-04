function fill_random_tlr!(A_tlr::NextLA.TLRMatrix, ArrayType::Type; seed::Integer)
    rng = MersenneTwister(seed)
    T = eltype(A_tlr)
    A_tlr.D .= ArrayType(randn(rng, T, size(A_tlr.D)))
    A_tlr.D_corner .= ArrayType(randn(rng, T, size(A_tlr.D_corner)))
    A_tlr.int_U .= ArrayType(randn(rng, T, size(A_tlr.int_U)))
    A_tlr.int_V .= ArrayType(randn(rng, T, size(A_tlr.int_V)))
    A_tlr.right_U .= ArrayType(randn(rng, T, size(A_tlr.right_U)))
    A_tlr.right_V .= ArrayType(randn(rng, T, size(A_tlr.right_V)))
    A_tlr.bottom_U .= ArrayType(randn(rng, T, size(A_tlr.bottom_U)))
    A_tlr.bottom_V .= ArrayType(randn(rng, T, size(A_tlr.bottom_V)))
    A_tlr.ranks .= A_tlr.maxrank
    return A_tlr
end

function assert_tlr_gemm_matches_dense(ArrayType::Type, T::Type, n::Int, b::Int, r::Int,
                                       orderA, orderB, synchronize; budget::Int,
                                       alpha=T(1.3), beta=T(-0.4), atol=1e-10, rtol=1e-10)
    A_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, r; tile_order=orderA)
    B_tlr = NextLA.TLRMatrix(ArrayType(zeros(T, n, n)), b, r; tile_order=orderB)
    fill_random_tlr!(A_tlr, ArrayType; seed=101)
    fill_random_tlr!(B_tlr, ArrayType; seed=202)

    rng = MersenneTwister(303)
    C0_cpu = randn(rng, T, n, n)
    C = ArrayType(C0_cpu)
    NextLA.TLRmodule.gemm!(C, A_tlr, B_tlr; alpha=alpha, beta=beta, max_workspace=budget)
    synchronize(C)

    A_dense = reconstruct_tlr(A_tlr)
    B_dense = reconstruct_tlr(B_tlr)
    C_ref = alpha * A_dense * B_dense + beta * C0_cpu
    @test isapprox(Array(C), C_ref; atol=atol, rtol=rtol)
end

@testset "TLR gemm! to dense on CPU" begin
    orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
    for orderA in orders, orderB in orders
        @testset "$(orderA) * $(orderB), budget=1" begin
            assert_tlr_gemm_matches_dense(Array, Float64, 12, 4, 2, orderA, orderB, _ -> nothing;
                                          budget=1)
        end
        @testset "$(orderA) * $(orderB), large budget" begin
            assert_tlr_gemm_matches_dense(Array, Float64, 16, 4, 3, orderA, orderB, _ -> nothing;
                                          budget=128 * 1024 * 1024)
        end
    end

    @testset "zero rank and one tile" begin
        assert_tlr_gemm_matches_dense(Array, Float64, 8, 4, 0,
                                      NextLA.TileRowMajor(), NextLA.TileColMajor(), _ -> nothing;
                                      budget=1)
        assert_tlr_gemm_matches_dense(Array, Float64, 4, 4, 2,
                                      NextLA.TileColMajor(), NextLA.TileRowMajor(), _ -> nothing;
                                      budget=1)
    end

end

@testset "TLR gemm! to dense on CUDA" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            orders = (NextLA.TileRowMajor(), NextLA.TileColMajor())
            for orderA in orders, orderB in orders
                assert_tlr_gemm_matches_dense(ArrayType, Float32, 12, 4, 2, orderA, orderB, synchronize;
                                              budget=1, alpha=Float32(1.2), beta=Float32(0.25),
                                              atol=5f-3, rtol=5f-3)
            end
        end
    end
end
