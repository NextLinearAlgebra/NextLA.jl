# Numerical primitives shared by active TLR algorithms.

@testset "TLR numerical primitives on CPU" begin
    @test _TLRM.tlr_orthogonalization_type(Float16) == Float32
    @test _TLRM.tlr_orthogonalization_type(Float32) == Float64
    @test _TLRM.tlr_orthogonalization_type(Float64) == Float64

    rng = MersenneTwister(511)
    X = randn(rng, Float32, 7, 5, 4)
    norm_sq = zeros(Float64, size(X, 3))
    @inferred _TLRM.batch_frobenius_norms_sq!(norm_sq, X)
    @test norm_sq ≈ [sum(abs2, X[:, :, k]) for k in axes(X, 3)]
end

@testset "TLR numerical primitives on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name == "CUDA" || continue
        @testset "$backend_name" begin
            rng = MersenneTwister(512)
            X = ArrayType(randn(rng, Float32, 7, 5, 4))
            norm_sq = ArrayType(zeros(Float64, size(X, 3)))
            _TLRM.batch_frobenius_norms_sq!(norm_sq, X)
            synchronize(norm_sq)
            @test Array(norm_sq) ≈
                  [sum(abs2, Float64.(Array(X[:, :, k]))) for k in axes(X, 3)] rtol=1e-10
        end
    end
end
