# Batched tile norms and the orthogonalization type, src/tlr/core/compression/norms.jl
# and src/tlr/core/compression/blas_utils.jl.
# Helpers: test/setup/tlr.jl.

@testset "TLR numerical primitives [CPU]" begin
    Random.seed!(3006)
    @test _TLRM.tlr_orthogonalization_type(Float16) == Float32
    @test _TLRM.tlr_orthogonalization_type(Float32) == Float64
    @test _TLRM.tlr_orthogonalization_type(Float64) == Float64

    rng = MersenneTwister(511)
    X = randn(rng, Float32, 7, 5, 4)
    norm_sq = zeros(Float64, size(X, 3))
    @inferred _TLRM.batch_frobenius_norms_sq!(norm_sq, X)
    @test norm_sq ≈ [sum(abs2, X[:, :, k]) for k in axes(X, 3)]
end

@testset "TLR numerical primitives [$backend_name]" for (backend_name, ArrayType, synchronize) in filter(b -> b[1] == "CUDA", TLR_BACKENDS)
    Random.seed!(3007)
    rng = MersenneTwister(512)
    X = ArrayType(randn(rng, Float32, 7, 5, 4))
    norm_sq = ArrayType(zeros(Float64, size(X, 3)))
    _TLRM.batch_frobenius_norms_sq!(norm_sq, X)
    synchronize(norm_sq)
    @test Array(norm_sq) ≈
          [sum(abs2, Float64.(Array(X[:, :, k]))) for k in axes(X, 3)] rtol=1e-10
end
