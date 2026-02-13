"""Build the pentagonal matrix that pemv! actually reads.

Columnwise storage + [Conj]Trans:
  The bottom-left l×l block A[m-l+1:m, 1:l] is treated as upper triangular.
  Zero the strictly-lower triangle of that block.

Rowwise storage + NoTrans:
  The top-right l×l block A[1:l, n-l+1:n] is treated as lower triangular.
  Zero the strictly-upper triangle of that block.
"""
function make_pentagonal(A, m, n, l, storev)
    Ap = copy(A)
    if l <= 1      # l==1 is treated as l==0 (no triangle)
        return Ap
    end
    if storev == 'C'
        for j in 1:l, i in (m - l + j + 1):m
            Ap[i, j] = zero(eltype(A))
        end
    else  # 'R'
        for j in 1:l
            col = n - l + j
            for i in 1:(j - 1)
                Ap[i, col] = zero(eltype(A))
            end
        end
    end
    return Ap
end

@testset "PEMV" begin

    # ── Column storage, conjugate transpose ────────────────────────────────
    @testset "Column storage, conjugate transpose" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, l = 40, 30, 10
            alpha = T(2.5 + 1.5im)
            beta  = T(1.2 - 0.8im)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            Y_orig = copy(Y)
            work = zeros(T, n)

            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)

            Ap = make_pentagonal(A, m, n, l, 'C')
            Y_ref = alpha * Ap' * X + beta * Y_orig
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── Column storage, transpose (real) ───────────────────────────────────
    @testset "Column storage, transpose (real)" begin
        @testset "$T" for T in (Float32, Float64)
            rtol = test_rtol(T)
            m, n, l = 40, 30, 10
            alpha = T(2.0)
            beta  = T(0.5)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            Y_orig = copy(Y)
            work = zeros(T, n)

            NextLA.pemv!('T', 'C', m, n, l, alpha, A, X, beta, Y, work)

            Ap = make_pentagonal(A, m, n, l, 'C')
            Y_ref = alpha * transpose(Ap) * X + beta * Y_orig
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── Row storage, no transpose ──────────────────────────────────────────
    @testset "Row storage, no transpose" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, l = 30, 40, 10
            alpha = T(1.8 + 2.2im)
            beta  = T(0.5 + 1.0im)

            A = rand(T, m, n)
            X = rand(T, n)
            Y = rand(T, m)
            Y_orig = copy(Y)
            work = zeros(T, m)

            NextLA.pemv!('N', 'R', m, n, l, alpha, A, X, beta, Y, work)

            Ap = make_pentagonal(A, m, n, l, 'R')
            Y_ref = alpha * Ap * X + beta * Y_orig
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── Wrapper (auto-workspace) ───────────────────────────────────────────
    @testset "Wrapper (auto-workspace)" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n = 40, 30
            alpha = T(2.0 + 1.0im)
            beta  = T(1.5 - 0.5im)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            Y_orig = copy(Y)

            # The wrapper defaults l = min(m,n) = 30
            NextLA.pemv!('C', 'C', alpha, A, X, beta, Y)

            l_def = min(m, n)
            Ap = make_pentagonal(A, m, n, l_def, 'C')
            Y_ref = alpha * Ap' * X + beta * Y_orig
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── Zero alpha ─────────────────────────────────────────────────────────
    @testset "Zero alpha" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, l = 20, 15, 5
            alpha = zero(T)
            beta  = T(2.0 + 1.5im)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            Y_orig = copy(Y)
            work = zeros(T, n)

            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)

            @test Y ≈ beta * Y_orig rtol=rtol
        end
    end

    # ── Zero beta ──────────────────────────────────────────────────────────
    @testset "Zero beta" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n, l = 20, 15, 5
            alpha = T(2.0 + 1.5im)
            beta  = zero(T)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            work = zeros(T, n)

            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)

            Ap = make_pentagonal(A, m, n, l, 'C')
            Y_ref = alpha * Ap' * X
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── Different sizes ────────────────────────────────────────────────────
    @testset "Different sizes" begin
        T = ComplexF64
        rtol = test_rtol(T)
        for (m, n, l) in [(10, 8, 5), (25, 20, 12), (50, 40, 25)]
            alpha = rand(T)
            beta  = rand(T)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            Y_orig = copy(Y)
            work = zeros(T, n)

            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)

            Ap = make_pentagonal(A, m, n, l, 'C')
            Y_ref = alpha * Ap' * X + beta * Y_orig
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── l == 0 (no triangle, pure gemv) ────────────────────────────────────
    @testset "l == 0 (pure gemv)" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n = 20, 15
            alpha = T(1.5 + 0.5im)
            beta  = T(0.8 - 0.3im)

            A = rand(T, m, n)
            X = rand(T, m)
            Y = rand(T, n)
            Y_orig = copy(Y)
            work = zeros(T, n)

            NextLA.pemv!('C', 'C', m, n, 0, alpha, A, X, beta, Y, work)

            Y_ref = alpha * A' * X + beta * Y_orig
            @test Y ≈ Y_ref rtol=rtol
        end
    end

    # ── Quick return for zero dims ─────────────────────────────────────────
    @testset "Quick return for zero dims" begin
        T = ComplexF64
        alpha = T(1.0)
        beta  = T(1.0)
        A = rand(T, 1, 5)
        X = rand(T, 5)
        Y = rand(T, 1)
        work = zeros(T, 5)
        Y_orig = copy(Y)

        # m = 0 → early return
        NextLA.pemv!('C', 'C', 0, 5, 0, alpha, A, X, beta, Y, work)

        # n = 0 → early return
        A2 = rand(T, 5, 1)
        X2 = T[]
        Y2 = rand(T, 5)
        work2 = zeros(T, 5)
        NextLA.pemv!('C', 'C', 5, 0, 0, alpha, A2, X2, beta, Y2, work2)
    end

    # ── BLAS consistency (l == 0 ⟹ full gemv) ─────────────────────────────
    @testset "BLAS consistency (l == 0)" begin
        @testset "$T" for T in (ComplexF32, ComplexF64)
            rtol = test_rtol(T)
            m, n = 20, 15
            alpha = T(2.0 + 1.0im)
            beta  = T(1.5 - 0.8im)

            A = rand(T, m, n)
            X = rand(T, m)
            Y1 = rand(T, n)
            Y2 = copy(Y1)
            work = zeros(T, n)

            # l == 0 → pemv should match plain gemv
            NextLA.pemv!('C', 'C', m, n, 0, alpha, A, X, beta, Y1, work)
            BLAS.gemv!('C', alpha, A, X, beta, Y2)

            @test Y1 ≈ Y2 rtol=rtol
        end
    end

    # ── Error handling ─────────────────────────────────────────────────────
    @testset "Error handling" begin
        T = ComplexF64
        m, n, l = 10, 8, 5
        alpha = one(T); beta = one(T)
        A = zeros(T, m, n)
        X = zeros(T, n)
        Y = zeros(T, m)
        work = zeros(T, m)

        @test_throws ArgumentError NextLA.pemv!('X', 'C', m, n, l, alpha, A, X, beta, Y, work)
        @test_throws ArgumentError NextLA.pemv!('C', 'X', m, n, l, alpha, A, X, beta, Y, work)
        @test_throws ArgumentError NextLA.pemv!('N', 'C', m, n, l, alpha, A, X, beta, Y, work)
        @test_throws ArgumentError NextLA.pemv!('C', 'R', m, n, l, alpha, A, X, beta, Y, work)
        @test_throws ArgumentError NextLA.pemv!('C', 'C', -1, n, l, alpha, A, X, beta, Y, work)
        @test_throws ArgumentError NextLA.pemv!('C', 'C', m, -1, l, alpha, A, X, beta, Y, work)
        @test_throws ArgumentError NextLA.pemv!('C', 'C', m, n, min(m,n)+1, alpha, A, X, beta, Y, work)
    end

    # ── Deterministic ──────────────────────────────────────────────────────
    @testset "Deterministic" begin
        Random.seed!(123)
        T = ComplexF64
        m, n, l = 20, 15, 8
        alpha = T(1.5 + 0.5im)
        beta  = T(0.8 - 0.3im)
        A = rand(T, m, n)
        X = rand(T, m)
        Y1 = rand(T, n); Y2 = copy(Y1)
        work1 = zeros(T, n); work2 = zeros(T, n)

        NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y1, work1)
        NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y2, work2)
        @test Y1 == Y2
    end
end
