# unified_rectrxm! with a mixed-precision container, src/mxp/blas/trsm/rectrxm_mixed.jl:
# triangular solve (func 'S') and multiply (func 'M') with A a TriMixedPrec,
# SymmMixedPrec or FullMixedPrec. A solve is checked by its backward error and a
# multiply by its forward error, both against the container's own stored values in
# Float64, so they measure the operation and not how the container rounded A; the
# bound is that of test/setup/mxp.jl for the narrowest precision involved.

# A triangle with an O(1) diagonal and an off-diagonal part of norm O(1), so the
# solves with and without a unit diagonal are well conditioned.
function _mxp_trxm_matrix(T, n, uplo)
    S = mxp_tri(uplo, mxp_rand(T, n, n) ./ T(n))
    S[LinearAlgebra.diagind(S)] .= one(T) .+ rand(T, n)
    return S
end

function _mxp_trxm_container(kind, T, n, uplo, P)
    A = _mxp_trxm_matrix(T, n, uplo)
    kind === TriMixedPrec && return TriMixedPrec(A, uplo; precisions=P)
    kind === SymmMixedPrec && return SymmMixedPrec(A + transpose(A) - Diagonal(A), uplo; precisions=P)
    return FullMixedPrec(A + transpose(mxp_tri(uplo, A)) - Diagonal(A); precisions=P)   # both triangles
end

# The triangle the routine reads, in Float64, from what the container stores.
_mxp_stored(Amp, uplo) = mxp_tri(uplo, Float64.(Array(reconstruct_matrix(Amp))))

function _mxp_trxm_err(side, uplo, trans, diag, func, alpha, Ad, X, B)
    if func == 'S'
        return mxp_trsm_backward_error(side, uplo, trans, diag, Ad, X, alpha .* Float64.(B))
    end
    return mxp_relerr(X, alpha .* mxp_trmm_reference(side, uplo, trans == 'C' ? 'T' : trans, diag, Ad, B))
end

@testset "RECTRXM mixed [CPU]" begin
    Random.seed!(2051)

    @testset "$(nameof(kind)) uplo=$uplo side=$side func=$func trans=$trans diag=$diag precisions=$P" for
            kind in (TriMixedPrec, SymmMixedPrec, FullMixedPrec), uplo in ('L', 'U'), side in ('L', 'R'),
            func in ('S', 'M'), trans in ('N', 'T', 'C'), diag in ('N', 'U'),
            P in (DataType[Float32, Float32], DataType[Float16, Float32], DataType[Float16, Float32, Float64])
        n, m = 37, 5
        T = P[end]
        Amp = _mxp_trxm_container(kind, T, n, uplo, P)
        B = side == 'L' ? mxp_rand(T, n, m) : mxp_rand(T, m, n)
        X = copy(B)
        @test unified_rectrxm!(side, uplo, trans, diag, T(2), func, Amp, X) === X
        @test _mxp_trxm_err(side, uplo, trans, diag, func, 2.0, _mxp_stored(Amp, uplo), X, B) <=
              mxp_tol(mxp_narrowest(P), n)
    end

    @testset "n=$n, precisions=$P" for n in (1, 2, 16, 64, 300), P in (DataType[NextLA.BFloat16, Float32],
                                                                      DataType[NextLA.BFloat16, Float16, Float32, Float64])
        T = P[end]
        for side in ('L', 'R'), func in ('S', 'M')
            Amp = _mxp_trxm_container(TriMixedPrec, T, n, 'L', P)
            B = side == 'L' ? mxp_rand(T, n, 3) : mxp_rand(T, 3, n)
            X = copy(B)
            unified_rectrxm!(side, 'L', 'N', 'N', one(T), func, Amp, X)
            @test _mxp_trxm_err(side, 'L', 'N', 'N', func, 1.0, _mxp_stored(Amp, 'L'), X, B) <=
                  mxp_tol(mxp_narrowest(P), n)
        end
    end

    # B of another type than the container: wider (the narrow block's product is
    # added through a partial sum), narrower (the wider block multiplies B
    # directly), and a transposed narrow block that is copied before mixing types.
    @testset "B $TB with precisions=$P, trans=$trans" for (TB, P) in
            ((Float64, DataType[Float16, Float32]), (Float64, DataType[Float32, Float32]),
             (Float32, DataType[Float64, Float64]), (Float32, DataType[Float16, Float64])),
            trans in ('N', 'T')
        n = 37
        Amp = _mxp_trxm_container(TriMixedPrec, P[end], n, 'L', P)
        for side in ('L', 'R'), func in ('S', 'M')
            B = side == 'L' ? mxp_rand(TB, n, 4) : mxp_rand(TB, 4, n)
            X = copy(B)
            unified_rectrxm!(side, 'L', trans, 'N', one(TB), func, Amp, X)
            @test eltype(X) === TB
            @test _mxp_trxm_err(side, 'L', trans, 'N', func, 1.0, _mxp_stored(Amp, 'L'), X, B) <=
                  mxp_tol(mxp_narrowest(vcat(P, TB)), n)
        end
    end

    @testset "scaled off-diagonal blocks, func=$func" for func in ('S', 'M')
        n, m = 16, 4
        A = tril(rand(Float32, n, n)) * 1f5 + Diagonal(1f6 * ones(Float32, n))
        Amp = TriMixedPrec(A, 'L'; precisions=DataType[Float16, Float32])
        @test Amp.OffDiag_scale != 1
        B = mxp_rand(Float32, n, m)
        X = copy(B)
        unified_rectrxm!('L', 'L', 'N', 'N', 1f0, func, Amp, X)
        @test _mxp_trxm_err('L', 'L', 'N', 'N', func, 1.0, _mxp_stored(Amp, 'L'), X, B) <= mxp_tol(Float16, n)
    end

    @testset "a scaled leaf, func=$func" for func in ('S', 'M')
        n, m = 16, 4
        A = tril(rand(n, n)) * 1.0e39 + 2.0e39I
        Amp = TriMixedPrec(A, 'L'; precisions=DataType[Float32])
        @test Amp.Base_scale != 1
        B = mxp_rand(Float64, n, m)
        X = copy(B)
        unified_rectrxm!('L', 'L', 'N', 'N', 1.0, func, Amp, X)
        Ad = tril(Float64.(Amp.Base) .* Amp.Base_scale)
        @test _mxp_trxm_err('L', 'L', 'N', 'N', func, 1.0, Ad, X, B) <= mxp_tol(Float32, n)
    end

    @testset "a multiply that overflows gives Inf, not floatmax" begin
        Amp = TriMixedPrec(Matrix(1.0e39I, 4, 4), 'L'; precisions=DataType[Float32])
        @test Amp.Base_scale != 1
        B = fill(1f34, 4, 2)
        unified_rectrxm!('L', 'L', 'N', 'N', 1, 'M', Amp, B)
        @test all(isinf, B)
    end

    # A non-finite source is converted as it is, not quantized (which would throw).
    @testset "a non-finite B through a narrow off-diagonal block" begin
        Amp = _mxp_trxm_container(TriMixedPrec, Float32, 8, 'L', DataType[Float16, Float32])
        B = mxp_rand(Float32, 8, 2)
        B[2, 1] = Inf32
        unified_rectrxm!('L', 'L', 'N', 'N', 1f0, 'M', Amp, B)
        @test !isfinite(B[2, 1]) && all(isfinite, B[:, 2])
    end

    # The off-diagonal product rounds B's part to the block's Float16 through a
    # scaled copy; a plain conversion made values past floatmax(Float16) Inf.
    @testset "an off-diagonal product keeps B values past floatmax(Float16)" begin
        n = 16
        Amp = _mxp_trxm_container(TriMixedPrec, Float32, n, 'L', DataType[Float16, Float32])
        B = 1.0f6 .* mxp_rand(Float32, n, 3)
        X = copy(B)
        unified_rectrxm!('L', 'L', 'N', 'N', 1f0, 'S', Amp, X)
        @test all(isfinite, X)
        @test _mxp_trxm_err('L', 'L', 'N', 'N', 'S', 1.0, _mxp_stored(Amp, 'L'), X, B) <= mxp_tol(Float16, n)
    end

    # The dense driver's TRMM base case is only valid up to TILE_DIM = 16; a larger
    # threshold used to give silently wrong answers, so unified_rec clamps it.
    @testset "multiply respects the dense TRMM tile limit" begin
        n, m = 64, 8
        Amp = _mxp_trxm_container(TriMixedPrec, Float32, n, 'L', DataType[Float32, Float32])
        B = mxp_rand(Float32, n, m)
        X = copy(B)
        MixedPrecision._rec_mixed!('M', 'L', 'L', 'N', Amp, X, 256)
        @test _mxp_trxm_err('L', 'L', 'N', 'N', 'M', 1.0, _mxp_stored(Amp, 'L'), X, B) <= mxp_tol(Float32, n)
    end

    @testset "the form without diag takes 'N', $(nameof(kind))" for kind in (TriMixedPrec, FullMixedPrec)
        Amp = _mxp_trxm_container(kind, Float32, 16, 'L', DataType[Float32, Float32])
        B = mxp_rand(Float32, 16, 3)
        B1, B2 = copy(B), copy(B)
        unified_rectrxm!('L', 'L', 'N', 1f0, 'S', Amp, B1)
        unified_rectrxm!('L', 'L', 'N', 'N', 1f0, 'S', Amp, B2)
        @test B1 == B2
    end

    @testset "uplo must be A.uplo for a triangular or symmetric container" begin
        Tri = _mxp_trxm_container(TriMixedPrec, Float32, 16, 'L', DataType[Float32, Float32])
        Sym = _mxp_trxm_container(SymmMixedPrec, Float32, 16, 'L', DataType[Float32, Float32])
        B = mxp_rand(Float32, 16, 3)
        @test_throws ArgumentError unified_rectrxm!('L', 'U', 'N', 'N', 1, 'S', Tri, copy(B))
        @test_throws ArgumentError unified_rectrxm!('R', 'U', 'T', 'N', 1, 'M', Tri, copy(B'))
        @test_throws ArgumentError unified_rectrxm!('L', 'U', 'N', 'N', 1, 'S', Sym, copy(B))
    end

    @testset "the flags are checked" begin
        Amp = _mxp_trxm_container(TriMixedPrec, Float32, 8, 'L', DataType[Float32])
        B = mxp_rand(Float32, 8, 2)
        @test_throws ArgumentError unified_rectrxm!('X', 'L', 'N', 'N', 1f0, 'S', Amp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'X', 'N', 'N', 1f0, 'S', Amp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'X', 'N', 1f0, 'S', Amp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'N', 'X', 1f0, 'S', Amp, B)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'N', 'N', 1f0, 'X', Amp, B)
    end
end

# On a GPU: the solve runs on the device with each kind of container; the stored
# values are read through a host copy of the container and checked on the CPU.
@testset "RECTRXM mixed [$name]" for (name, AT, _) in filter(b -> b[1] != "CPU", MXP_BACKENDS)
    Random.seed!(2052)
    n = 37

    @testset "FullMixedPrec uplo=$uplo side=$side func=$func trans=$trans diag=$diag precisions=$P" for
            uplo in ('L', 'U'), side in ('L', 'R'), func in ('S', 'M'), trans in ('N', 'T'), diag in ('N', 'U'),
            P in (DataType[Float32, Float32], DataType[Float16, Float32, Float64])
        T = P[end]
        A = _mxp_trxm_matrix(T, n, uplo)
        Amp = FullMixedPrec(AT(A + transpose(A) - Diagonal(A)); precisions=P)
        B = side == 'L' ? mxp_rand(T, n, 4) : mxp_rand(T, 4, n)
        X = AT(copy(B))
        unified_rectrxm!(side, uplo, trans, diag, T(2), func, Amp, X)
        @test X isa AT
        @test _mxp_trxm_err(side, uplo, trans, diag, func, 2.0, _mxp_stored(Amp, uplo), X, B) <=
              mxp_tol(mxp_narrowest(P), n)
    end

    @testset "$(nameof(kind)) uplo=$uplo side=$side func=$func trans=$trans" for
            kind in (TriMixedPrec, SymmMixedPrec), uplo in ('L', 'U'), side in ('L', 'R'), func in ('S', 'M'),
            trans in ('N', 'T')
        P = DataType[Float16, Float32, Float64]
        A = _mxp_trxm_matrix(Float64, n, uplo)
        A = kind === SymmMixedPrec ? A + transpose(A) - Diagonal(A) : A
        B = side == 'L' ? mxp_rand(Float64, n, 4) : mxp_rand(Float64, 4, n)
        Amp = kind(AT(A), uplo; precisions=P)
        X = AT(copy(B))
        unified_rectrxm!(side, uplo, trans, 'N', 2.0, func, Amp, X)
        @test X isa AT
        @test _mxp_trxm_err(side, uplo, trans, 'N', func, 2.0, _mxp_stored(mxp_host(Amp), uplo), X, B) <=
              mxp_tol(Float16, n)
    end
end
