using Test
using LinearAlgebra
using Random
using KernelAbstractions: CPU, get_backend
using NextLA
using NextLA: DeviceParams, compute_params, scqr3!, scqr3_gram!, scqr3_fukaya_shift

include(joinpath(@__DIR__, "gpu_backends.jl"))

# LAPACK-style scaled orthogonality (cf. `test/geqrt.jl`, `test/geqr2.jl`):
#   opnorm(Q'Q - I, 1) / (m * eps(real(T))) < 10
# Thin panel `A` is `m×b` with `b` columns to be orthogonal: check `b×b` Gram to `I`.
function scqr3_orth_metric(A::AbstractMatrix)
    Ac = Array(A)
    T = eltype(Ac)
    m, b = size(Ac)
    Id = Matrix{T}(I, b, b)
    return opnorm(Ac' * Ac - Id, 1) / (max(m, b) * eps(real(T)))
end
# Use at most one GPU backend in addition to CPU: loading several vendor runtimes
# (e.g. CUDA then oneAPI/ROCm) in one Julia process has triggered SIGSEGV on some hosts.
function scqr3_test_backends()
    out = Tuple{String, Type, Function}[]
    push!(out, ("CPU", Array, _ -> nothing))
    gpus = available_gpu_backends()
    if !isempty(gpus)
        push!(out, gpus[1])
    end
    return out
end
@testset "scqr3_fukaya_shift reference (Fukaya et al. 2018, eq. (67))" begin
    # s = 11 * (m*n + n*(n+1)) * eps(real(T)) * tr with n = b = 3, m = 5, tr = 1
    m, b, tr = 5, 3, 1.0
    n = b
    sref = 11 * (m * n + n * (n + 1)) * eps(Float64) * tr
    @test scqr3_fukaya_shift(Float64, m, b, tr) ≈ sref
    tr32 = Float32(tr)
    sref32 = Float32(11) * (Float32(m * n) + Float32(n * (n + 1))) * eps(Float32) * tr32
    @test scqr3_fukaya_shift(Float32, m, b, tr32) ≈ sref32
end

@testset "scqr3_gram! TILE_DIM → different tiles vs mul! (CPU)" begin
    rng = MersenneTwister(99)
    T = Float64
    m, b = 30, 10
    A = randn(rng, T, m, b)
    Gref = zeros(T, b, b)
    mul!(Gref, A', A)
    # Abstract snap: 9 → 8 (tie 8/10); 18 → 16 (tie 16/20). Pick on CPU uses full feasible set.
    @test NextLA._scqr3_snap_gram_tile(9) == 8
    @test NextLA._scqr3_snap_gram_tile(18) == 16
    @test NextLA._scqr3_pick_gram_tile(CPU(), T, 9) == 8
    @test NextLA._scqr3_pick_gram_tile(CPU(), T, 18) == 16
    p8 = DeviceParams(4, 10_000, b, 1, 4, 2, 2, 1, 9, 1, 100, T(1.0))
    p16 = DeviceParams(4, 10_000, b, 1, 4, 2, 2, 1, 18, 1, 100, T(1.0))
    G1 = zeros(T, b, b)
    G2 = zeros(T, b, b)
    scqr3_gram!(G1, A, m, b; params = p8)
    scqr3_gram!(G2, A, m, b; params = p16)
    @test G1 ≈ Gref rtol = 1e-12 atol = 1e-12
    @test G2 ≈ Gref rtol = 1e-12 atol = 1e-12
end

@testset "scqr3_gram! vs mul!(A', A) (CPU)" begin
    rng = MersenneTwister(42)
    for T in (Float32, Float64, ComplexF64)
        for (m, b) in ((40, 12), (35, 17), (16, 16))
            A = randn(rng, T, m, b)
            Gref = zeros(T, b, b)
            G = zeros(T, b, b)
            mul!(Gref, A', A)
            scqr3_gram!(G, A, m, b)
            rtol = T <: Complex ? 80 * sqrt(eps(real(T))) : (T == Float32 ? 8e-5 : 1e-12)
            atol = T <: Complex ? 80 * eps(real(T)) : (T == Float32 ? 8e-6 : 1e-12)
            @test G ≈ Gref rtol = rtol atol = atol
        end
    end
end

for (backend_name, ArrayType, synchronize) in scqr3_test_backends()
    backend_name == "CPU" && continue
    @testset "scqr3_gram! vs mul!(A', A) [$backend_name]" begin
        rng = MersenneTwister(43)
        T = ComplexF64
        m, b = 48, 11
        A = ArrayType(randn(rng, T, m, b))
        G = ArrayType(zeros(T, b, b))
        scqr3_gram!(G, A, m, b)
        synchronize(A)
        synchronize(G)
        Ar = Array(A)
        @test Array(G) ≈ Ar' * Ar rtol = 120 * sqrt(eps(Float64)) atol = 120 * eps(Float64)
    end
    @testset "scqr3_gram! TILE_DIM vs mul! [$backend_name]" begin
        rng = MersenneTwister(44)
        T = ComplexF64
        m, b = 40, 9
        A = ArrayType(randn(rng, T, m, b))
        Ar = Array(A)
        Gref = Ar' * Ar
        p16 = DeviceParams(4, 10_000, b, 1, 4, 2, 2, 1, 14, 1, 100, one(T))
        p64 = DeviceParams(4, 10_000, b, 1, 4, 2, 2, 1, 50, 1, 100, one(T))
        be = get_backend(A)
        @test NextLA._scqr3_pick_gram_tile(be, T, 15) == 16
        @test NextLA._scqr3_pick_gram_tile(be, T, 50) == 32
        G1 = ArrayType(zeros(T, b, b))
        G2 = ArrayType(zeros(T, b, b))
        scqr3_gram!(G1, A, m, b; params = p16)
        scqr3_gram!(G2, A, m, b; params = p64)
        synchronize(G1)
        synchronize(G2)
        rtol = 120 * sqrt(eps(Float64))
        atol = 120 * eps(Float64)
        @test Array(G1) ≈ Gref rtol = rtol atol = atol
        @test Array(G2) ≈ Gref rtol = rtol atol = atol
    end
end

@testset "scqr3_cholesky! argument checks (CPU)" begin
    G = zeros(3, 3)
    info = [0]
    @test_throws ArgumentError NextLA.scqr3_cholesky!(G, info, 4)
    @test_throws ArgumentError NextLA.scqr3_cholesky!(G, Int[], 2)
    @test_throws ArgumentError NextLA.scqr3_cholesky!(G, info, -1)
    NextLA.scqr3_cholesky!(G, info, 0)
    @test info[1] == 0
end

for (backend_name, ArrayType, synchronize) in scqr3_test_backends()
    @testset "scqr3_cholesky! [$backend_name]" begin
        rng = MersenneTwister(17)
        for T in (Float32, Float64)
            b = 9
            n = 16
            X = ArrayType(randn(rng, T, n, b))
            Gcpu = Matrix(X' * X)
            Rref = cholesky(Hermitian(Gcpu, :U)).U
            G = ArrayType(copy(Gcpu))
            info = ArrayType(zeros(Int, 1))
            NextLA.scqr3_cholesky!(G, info, b)
            synchronize(G)
            synchronize(info)
            @test Array(info)[1] == 0
            Gu = triu(Array(G)[1:b, 1:b])
            @test Gu ≈ Rref rtol = 10 * sqrt(eps(T)) atol = 10 * eps(T)
        end

        T = ComplexF64
        b = 6
        n = 10
        Xc = ArrayType(complex.(randn(rng, n, b), randn(rng, n, b)))
        Gcpu = Matrix(Xc' * Xc)
        Rref = cholesky(Hermitian(Gcpu, :U)).U
        G = ArrayType(copy(Gcpu))
        info = ArrayType(zeros(Int, 1))
        NextLA.scqr3_cholesky!(G, info, b)
        synchronize(G)
        synchronize(info)
        @test Array(info)[1] == 0
        Gu = triu(Array(G)[1:b, 1:b])
        @test Gu ≈ Rref rtol = 10 * sqrt(eps(real(T))) atol = 10 * eps(real(T))

        # indefinite: fail on first diagonal / Schur complement
        b2 = 3
        Hcpu = Matrix(Diagonal(T[-1.0, 2.0, 3.0]))
        H = ArrayType(copy(Hcpu))
        info2 = ArrayType(zeros(Int, 1))
        NextLA.scqr3_cholesky!(H, info2, b2)
        synchronize(H)
        synchronize(info2)
        @test Array(info2)[1] == 1

        # shift restores PD (host folds s into diagonal before kernel)
        s = 2.0
        H2cpu = copy(Hcpu)
        for i in 1:b2
            H2cpu[i, i] += s
        end
        H2 = ArrayType(H2cpu)
        info3 = ArrayType(zeros(Int, 1))
        NextLA.scqr3_cholesky!(H2, info3, b2)
        synchronize(H2)
        synchronize(info3)
        @test Array(info3)[1] == 0
        R2 = cholesky(Hermitian(H2cpu, :U)).U
        @test triu(Array(H2)[1:b2, 1:b2]) ≈ R2
    end
end

@testset "scqr3! three-iter loop (CPU)" begin
    for T in (Float64, Float32)
        m = 80
        b = 12
        rng = MersenneTwister(3)
        A = randn(rng, T, m, b)
        R = zeros(T, b, b)
        G = zeros(T, b, b)
        info = [0]
        params = compute_params(CPU(), T, max(m, b); b = b, c = 1)
        scqr3!(m, b, A, R, G, info; params = params)
        @test scqr3_orth_metric(A) < 10
        A2 = randn(rng, T, m, b)
        R2 = zeros(T, b, b)
        scqr3!(A2, R2)
        @test scqr3_orth_metric(A2) < 10
        dp = DeviceParams(8, 1000, b, 2, 4, 2, 2, 2, 16, 2, 50, T(1.0))
        A3 = randn(rng, T, m, b)
        R3 = zeros(T, b, b)
        G3 = zeros(T, b, b)
        info3 = [0]
        @test_throws ArgumentError scqr3!(m, b, A3, R3, G3, info3; params = dp)
        part = zeros(T, b, b, dp.Px * dp.Pz)
        scqr3!(m, b, A3, R3, G3, info3; params = dp, partials = part)
        @test scqr3_orth_metric(A3) < 10
    end
end

for (backend_name, ArrayType, synchronize) in scqr3_test_backends()
    @testset "scqr3! [$backend_name]" begin
        T = Float64
        m = 48
        # b = 11 ⇒ nextpow(2, b) = 16 for device-side trace pack + workgroup_reduce padding
        b = 11
        rng = MersenneTwister(11)
        A = ArrayType(randn(rng, T, m, b))
        R = ArrayType(zeros(T, b, b))
        be = get_backend(A)
        # N must be large enough vs probed P so requested b is not clamped (b_max = N ÷ P_x).
        N_layout = max(256, m, 16 * b)
        params = compute_params(be, T, N_layout; b = b, c = 1)
        @test params.b == b
        G = ArrayType(zeros(T, b, b))
        info = ArrayType(zeros(Int, 1))
        scqr3!(m, b, A, R, G, info; params = params)
        synchronize(A)
        @test scqr3_orth_metric(A) < 10
    end
end
