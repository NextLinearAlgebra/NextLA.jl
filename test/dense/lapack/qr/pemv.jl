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

for (backend_name, ArrayType, synchronize) in available_backends()
    @testset "PEMV [$backend_name]" begin
        @testset "Column storage, conjugate transpose" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n, l = 40, 30, 10
                alpha = T(2.5 + 1.5im)
                beta  = T(1.2 - 0.8im)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                Y_orig_cpu = Array(copy(Y))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
                synchronize(Y)

                Ap = make_pentagonal(Array(A), m, n, l, 'C')
                Y_ref = alpha * Ap' * Array(X) + beta * Y_orig_cpu
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "Column storage, transpose (real)" begin
            @testset "$T" for T in (Float32, Float64)
                rtol = test_rtol(T)
                m, n, l = 40, 30, 10
                alpha = T(2.0)
                beta  = T(0.5)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                Y_orig_cpu = Array(copy(Y))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('T', 'C', m, n, l, alpha, A, X, beta, Y, work)
                synchronize(Y)

                Ap = make_pentagonal(Array(A), m, n, l, 'C')
                Y_ref = alpha * transpose(Ap) * Array(X) + beta * Y_orig_cpu
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "Row storage, no transpose" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n, l = 30, 40, 10
                alpha = T(1.8 + 2.2im)
                beta  = T(0.5 + 1.0im)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, n)))
                Y = ArrayType(copy(rand(T, m)))
                Y_orig_cpu = Array(copy(Y))
                work = ArrayType(zeros(T, m))

                NextLA.pemv!('N', 'R', m, n, l, alpha, A, X, beta, Y, work)
                synchronize(Y)

                Ap = make_pentagonal(Array(A), m, n, l, 'R')
                Y_ref = alpha * Ap * Array(X) + beta * Y_orig_cpu
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "Wrapper (auto-workspace)" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n = 40, 30
                alpha = T(2.0 + 1.0im)
                beta  = T(1.5 - 0.5im)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                Y_orig_cpu = Array(copy(Y))

                NextLA.pemv!('C', 'C', alpha, A, X, beta, Y)
                synchronize(Y)

                l_def = min(m, n)
                Ap = make_pentagonal(Array(A), m, n, l_def, 'C')
                Y_ref = alpha * Ap' * Array(X) + beta * Y_orig_cpu
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "Zero alpha" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n, l = 20, 15, 5
                alpha = zero(T)
                beta  = T(2.0 + 1.5im)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                Y_orig_cpu = Array(copy(Y))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
                synchronize(Y)

                @test Array(Y) ≈ beta * Y_orig_cpu rtol=rtol
            end
        end

        @testset "Zero beta" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n, l = 20, 15, 5
                alpha = T(2.0 + 1.5im)
                beta  = zero(T)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
                synchronize(Y)

                Ap = make_pentagonal(Array(A), m, n, l, 'C')
                Y_ref = alpha * Ap' * Array(X)
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "Different sizes" begin
            T = ComplexF64
            rtol = test_rtol(T)
            for (m, n, l) in [(10, 8, 5), (25, 20, 12), (50, 40, 25)]
                alpha = rand(T)
                beta  = rand(T)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                Y_orig_cpu = Array(copy(Y))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y, work)
                synchronize(Y)

                Ap = make_pentagonal(Array(A), m, n, l, 'C')
                Y_ref = alpha * Ap' * Array(X) + beta * Y_orig_cpu
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "l == 0 (pure gemv)" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n = 20, 15
                alpha = T(1.5 + 0.5im)
                beta  = T(0.8 - 0.3im)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y = ArrayType(copy(rand(T, n)))
                Y_orig_cpu = Array(copy(Y))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('C', 'C', m, n, 0, alpha, A, X, beta, Y, work)
                synchronize(Y)

                Y_ref = alpha * Array(A)' * Array(X) + beta * Y_orig_cpu
                @test Array(Y) ≈ Y_ref rtol=rtol
            end
        end

        @testset "Quick return for zero dims" begin
            T = ComplexF64
            alpha = T(1.0)
            beta  = T(1.0)
            A = ArrayType(copy(rand(T, 1, 5)))
            X = ArrayType(copy(rand(T, 5)))
            Y = ArrayType(copy(rand(T, 1)))
            work = ArrayType(zeros(T, 5))
            Y_orig = Array(copy(Y))

            NextLA.pemv!('C', 'C', 0, 5, 0, alpha, A, X, beta, Y, work)
            synchronize(Y)

            A2 = ArrayType(copy(rand(T, 5, 1)))
            X2 = ArrayType(zeros(T, 0))
            Y2 = ArrayType(copy(rand(T, 5)))
            work2 = ArrayType(zeros(T, 5))
            NextLA.pemv!('C', 'C', 5, 0, 0, alpha, A2, X2, beta, Y2, work2)
        end

        @testset "BLAS consistency (l == 0)" begin
            @testset "$T" for T in (ComplexF32, ComplexF64)
                rtol = test_rtol(T)
                m, n = 20, 15
                alpha = T(2.0 + 1.0im)
                beta  = T(1.5 - 0.8im)

                A = ArrayType(copy(rand(T, m, n)))
                X = ArrayType(copy(rand(T, m)))
                Y1 = ArrayType(copy(rand(T, n)))
                Y2_cpu = copy(Array(Y1))
                work = ArrayType(zeros(T, n))

                NextLA.pemv!('C', 'C', m, n, 0, alpha, A, X, beta, Y1, work)
                synchronize(Y1)
                BLAS.gemv!('C', alpha, Array(A), Array(X), beta, Y2_cpu)

                @test Array(Y1) ≈ Y2_cpu rtol=rtol
            end
        end

        @testset "Deterministic" begin
            Random.seed!(123)
            T = ComplexF64
            m, n, l = 20, 15, 8
            alpha = T(1.5 + 0.5im)
            beta  = T(0.8 - 0.3im)
            A = ArrayType(copy(rand(T, m, n)))
            X = ArrayType(copy(rand(T, m)))
            Y1 = ArrayType(copy(rand(T, n)))
            Y2 = ArrayType(copy(Y1))
            work1 = ArrayType(zeros(T, n))
            work2 = ArrayType(zeros(T, n))

            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y1, work1)
            NextLA.pemv!('C', 'C', m, n, l, alpha, A, X, beta, Y2, work2)
            synchronize(Y1)
            synchronize(Y2)
            @test Array(Y1) == Array(Y2)
        end
    end
end

@testset "PEMV Error handling" begin
    T = ComplexF64
    m, n, l = 10, 8, 5
    alpha = one(T)
    beta = one(T)
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
