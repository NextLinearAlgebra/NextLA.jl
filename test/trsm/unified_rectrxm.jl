@testset "unified_rectrxm! GPU" begin
    backends = available_gpu_backends()
    if isempty(backends)
        @test_skip "No GPU backends available"
    end
    for (backend_name, ArrayType, synchronize) in backends
        @testset "[$backend_name]" begin
            tol = 1e-14

            @testset "func=$func side=$side uplo=$uplo trans=$trans n=$n m=$m" for
                    n     in [16, 32, 128, 256],
                    m     in [1, 8, 64],
                    side  in ['L', 'R'],
                    uplo  in ['L', 'U'],
                    trans in ['N', 'T', 'C'],
                    func  in ['S', 'M']

                alpha = 1.0

                A = if uplo == 'L'
                    Matrix(LowerTriangular(rand(n, n) .+ 1))
                else
                    Matrix(UpperTriangular(rand(n, n) .+ 1))
                end
                A += Diagonal(10 * ones(n))

                B = side == 'L' ? rand(n, m) .+ 1 : rand(m, n) .+ 1

                Ac, Bc = copy(A), copy(B)
                A_gpu = ArrayType(A)
                B_gpu = ArrayType(B)

                unified_rectrxm!(side, uplo, trans, alpha, func, A_gpu, B_gpu)
                synchronize(B_gpu)

                if func == 'S'
                    LinearAlgebra.BLAS.trsm!(side, uplo, trans, 'N', alpha, Ac, Bc)
                else
                    LinearAlgebra.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac, Bc)
                end

                rel_err = norm(Array(B_gpu) - Bc) / norm(Bc)
                @test rel_err < tol
            end
        end
    end
end

# The GPU testset above is the only coverage unified_rectrxm! had, and on CUDA
# it now dispatches into ext/cuda/rectrxm.jl. Without the block below, the
# recursive KernelAbstractions path in src/trsm/rectrxm.jl — which is what CPU,
# oneAPI and Metal use — would be exercised by nothing at all.
@testset "unified_rectrxm! CPU (recursive KA path)" begin
    tol = 1e-12

    @testset "func=$func side=$side uplo=$uplo trans=$trans n=$n m=$m" for
            n     in [16, 32, 128, 256],
            m     in [1, 8, 64],
            side  in ['L', 'R'],
            uplo  in ['L', 'U'],
            trans in ['N', 'T', 'C'],
            func  in ['S', 'M']

        alpha = 1.0

        A = if uplo == 'L'
            Matrix(LowerTriangular(rand(n, n) .+ 1))
        else
            Matrix(UpperTriangular(rand(n, n) .+ 1))
        end
        A += Diagonal(10 * ones(n))

        B = side == 'L' ? rand(n, m) .+ 1 : rand(m, n) .+ 1

        Ac, Bc = copy(A), copy(B)
        A_cpu, B_cpu = copy(A), copy(B)

        unified_rectrxm!(side, uplo, trans, alpha, func, A_cpu, B_cpu)

        if func == 'S'
            LinearAlgebra.BLAS.trsm!(side, uplo, trans, 'N', alpha, Ac, Bc)
        else
            LinearAlgebra.BLAS.trmm!(side, uplo, trans, 'N', alpha, Ac, Bc)
        end

        @test norm(B_cpu - Bc) / norm(Bc) < tol
    end
end

# A silently non-dispatching extension looks exactly like a passing suite --
# that is how the oneAPI extension went unnoticed. Assert where each call lands.
@testset "unified_rectrxm! dispatch" begin
    _slashes(p) = replace(p, "\\" => "/")
    n = 8
    A = Matrix(LowerTriangular(rand(n, n) .+ 1)) + Diagonal(10 * ones(n))
    B = rand(n, 2) .+ 1
    # The seven-argument form is generic and only delegates, so it always lands
    # in src/. The eight-argument form is the one the extension specialises.
    @test endswith(_slashes(_method_file(unified_rectrxm!, 'L', 'L', 'N', 1.0, 'S', A, B)),
                   "src/trsm/rectrxm.jl")
    @test endswith(_slashes(_method_file(unified_rectrxm!, 'L', 'L', 'N', 'N', 1.0, 'S', A, B)),
                   "src/trsm/rectrxm.jl")

    # Backends with a vendor fast path; the rest stay on the portable kernels.
    fast = Dict("CUDA" => "ext/cuda/rectrxm.jl", "AMDGPU" => "ext/amdgpu/rectrxm.jl")
    for (name, AT, _) in available_gpu_backends()
        haskey(fast, name) || continue
        Ag, Bg = AT(A), AT(B)
        @test endswith(_slashes(_method_file(unified_rectrxm!, 'L', 'L', 'N', 'N', 1.0, 'S', Ag, Bg)),
                       fast[name])
    end
end

# diag='U' treats the stored diagonal as all ones without reading it, the BLAS
# convention. The seven-argument form delegates here with 'N', so the sweeps
# above already cover that half.
#
# The CPU kernels and the vendor extensions (CUDA, AMDGPU) implement this
# independently -- the kernels skip the diagonal read, the vendor library is
# told 'U' -- so the last testset checks they agree. Each passing alone would not catch the extension quietly
# ignoring the flag.
@testset "unified_rectrxm! unit diagonal" begin
    tol = 1e-12

    @testset "func=$func side=$side uplo=$uplo trans=$trans diag=$dg" for
            func  in ['S', 'M'],
            side  in ['L', 'R'],
            uplo  in ['L', 'U'],
            trans in ['N', 'T'],
            dg    in ['N', 'U']

        n, m = 32, 8
        A = uplo == 'L' ? Matrix(LowerTriangular(rand(n, n) .+ 1)) :
                          Matrix(UpperTriangular(rand(n, n) .+ 1))
        A += Diagonal(10 * ones(n))
        B = side == 'L' ? rand(n, m) .+ 1 : rand(m, n) .+ 1

        ref = copy(B)
        if func == 'S'
            LinearAlgebra.BLAS.trsm!(side, uplo, trans, dg, 1.0, A, ref)
        else
            LinearAlgebra.BLAS.trmm!(side, uplo, trans, dg, 1.0, A, ref)
        end

        B_cpu = copy(B)
        unified_rectrxm!(side, uplo, trans, dg, 1.0, func, copy(A), B_cpu)
        @test norm(B_cpu - ref) / norm(ref) < tol

        for (name, AT, sync) in available_gpu_backends()
            name in ("CUDA", "AMDGPU") || continue
            Ag, Bg = AT(copy(A)), AT(copy(B))
            unified_rectrxm!(side, uplo, trans, dg, 1.0, func, Ag, Bg)
            sync(Bg)
            @test norm(Array(Bg) - ref) / norm(ref) < tol
            # the point of the testset: both paths, same answer
            @test Array(Bg) ≈ B_cpu rtol=1e-10
        end
    end

    # A Float16 base case accumulates in Float32 and rounds once on store. The
    # test is not a tolerance: it is that the answer equals the one obtained by
    # doing the whole solve in Float32 and rounding at the end, which is what
    # the promotion claims to deliver. A kernel accumulating in Float16 misses
    # it by more than an order of magnitude at n = 128.
    @testset "Float16 accumulates wider than it stores" begin
        for n in (16, 64, 128), (side, uplo, func) in
                [('L', 'L', 'S'), ('L', 'U', 'S'), ('R', 'L', 'M')]
            A0 = Matrix(LowerTriangular(rand(Float64, n, n) .+ 1.0))
            uplo == 'U' && (A0 = Matrix(UpperTriangular(rand(Float64, n, n) .+ 1.0)))
            B0 = rand(Float64, n, n)
            A16, B16 = Float16.(A0), Float16.(B0)

            got = copy(B16)
            unified_rectrxm!(side, uplo, 'N', 'N', Float16(1), func, copy(A16), got)

            # The same operation carried out in Float32 and rounded once at the
            # end. They agree exactly only where the call is a single base case:
            # above the threshold the recursion's GEMM updates round to Float16
            # at every step, so the promotion buys accuracy within a block and
            # not across the recursion. That is the branch's bargain too --
            # dispatch_trsm! wrapped the base-case call and nothing above it.
            want32 = Float32.(B16)
            unified_rectrxm!(side, uplo, 'N', 'N', Float32(1), func,
                             Float32.(A16), want32)
            if n <= (func == 'S' ? 256 : 16)
                @test got == Float16.(want32)
            end

            # and it is genuinely better than accumulating in Float16: compare
            # both against the exact solve of the rounded inputs
            exact = func == 'S' ?
                (side == 'L' ? Float64.(A16) \ Float64.(B16) :
                               Float64.(B16) / Float64.(A16)) :
                (side == 'L' ? Float64.(A16) * Float64.(B16) :
                               Float64.(B16) * Float64.(A16))
            @test norm(Float64.(got) - exact) / norm(exact) < 1e-3
        end
    end

    @testset "rejects a bad diag" begin
        A = Matrix(LowerTriangular(rand(8, 8) .+ 1)) + Diagonal(10 * ones(8))
        B = rand(8, 2)
        @test_throws ArgumentError unified_rectrxm!('L', 'L', 'N', 'X', 1.0, 'S', A, B)
    end
end

# trsm!/trmm! are the BLAS-order entry points over unified_rectrxm!. They are
# checked on every backend against LinearAlgebra.BLAS, over the whole flag
# space and all four BLAS element types, with alpha != 1 so the scaling is
# exercised.
# On CUDA and AMDGPU the answer must also come from the vendor fast path: a
# wrapper that quietly fell back to the portable kernels would still pass the
# numerical checks.
@testset "trsm! / trmm!" begin
    _slashes(p) = replace(p, "\\" => "/")
    @test :trsm! in names(NextLA) && :trmm! in names(NextLA)
    @test :unified_rectrxm! in names(NextLA)
    @test NextLA.trsm! !== LinearAlgebra.BLAS.trsm!
    @test NextLA.trmm! !== LinearAlgebra.BLAS.trmm!

    fast = Dict("CUDA" => "ext/cuda/rectrxm.jl", "AMDGPU" => "ext/amdgpu/rectrxm.jl")
    targets = [("CPU", Array, _ -> nothing); available_gpu_backends()]
    for (name, AT, sync) in targets
        @testset "[$name] $fn T=$T side=$side uplo=$uplo trans=$trans diag=$dg" for
                T     in (Float32, Float64, ComplexF32, ComplexF64),
                fn    in (trsm!, trmm!),
                side  in ('L', 'R'),
                uplo  in ('L', 'U'),
                trans in ('N', 'T', 'C'),
                dg    in ('N', 'U')

            n, m = 48, 5
            # small off-diagonal part: well conditioned with or without the
            # stored diagonal, so Float32 can be held to a tight tolerance
            L = T(0.1) .* rand(T, n, n)
            A = (uplo == 'L' ? Matrix(LowerTriangular(L)) : Matrix(UpperTriangular(L))) +
                Diagonal(fill(T(2), n))
            B = side == 'L' ? rand(T, n, m) .+ 1 : rand(T, m, n) .+ 1
            alpha = T(1.5)

            ref = copy(B)
            blas = fn === trsm! ? LinearAlgebra.BLAS.trsm! : LinearAlgebra.BLAS.trmm!
            blas(side, uplo, trans, dg, alpha, A, ref)

            Ag, Bg = AT(A), AT(B)
            out = fn(side, uplo, trans, dg, alpha, Ag, Bg)
            sync(Bg)
            @test out === Bg                        # in place, like BLAS
            tol = real(T) === Float32 ? 1e-5 : 1e-12
            @test norm(Array(Bg) - ref) / norm(ref) < tol
        end

        if haskey(fast, name)
            @testset "[$name] reaches the vendor library" begin
                for T in (Float32, Float64, ComplexF32, ComplexF64)
                    Ag = AT(Matrix(LowerTriangular(rand(T, 8, 8))) + Diagonal(fill(T(2), 8)))
                    Bg = AT(rand(T, 8, 3))
                    # the wrapper is generic; what it calls is the extension method
                    @test endswith(_slashes(_method_file(unified_rectrxm!, 'L', 'L', 'N', 'N',
                                                         T(1), 'S', Ag, Bg)), fast[name])
                end
            end
        end
    end
end

# A transposed Float16 operand reaches the base case as a view of a Transpose
# once the recursion splits it (TRMM above n = 16, TRSM above 256), or at once
# when the caller passes a view. The base case widens it to Float32; done
# directly, that broadcast has no GPU method and falls back to scalar indexing.
# Checked on every backend, both sides of both thresholds, whole arrays and
# offset views, against the Float64 answer for the rounded inputs.
@testset "Float16 with transposes and views" begin
    targets = [("CPU", Array, _ -> nothing); available_gpu_backends()]
    for (name, AT, sync) in targets
        @testset "[$name] $fn n=$n side=$side uplo=$uplo trans=$trans view=$useview" for
                fn      in (trsm!, trmm!),
                n       in (17, 300),
                side    in ('L', 'R'),
                uplo    in ('L', 'U'),
                trans   in ('N', 'T', 'C'),
                useview in (false, true)

            m = 5
            L = rand(n, n) ./ n
            A = Float16.((uplo == 'L' ? Matrix(LowerTriangular(L)) : Matrix(UpperTriangular(L))) +
                         Diagonal(fill(2.0, n)))
            B = Float16.(side == 'L' ? rand(n, m) .+ 1 : rand(m, n) .+ 1)

            ref = Float64.(B)
            blas = fn === trsm! ? LinearAlgebra.BLAS.trsm! : LinearAlgebra.BLAS.trmm!
            blas(side, uplo, trans, 'N', 1.0, Float64.(A), ref)

            if useview
                # offset blocks of larger arrays, so neither operand is a whole array
                Abig = AT(zeros(Float16, n + 3, n + 2))
                Bbig = AT(zeros(Float16, size(B, 1) + 2, size(B, 2) + 2))
                Ag = view(Abig, 3:n+2, 2:n+1)
                Bg = view(Bbig, 2:size(B, 1)+1, 2:size(B, 2)+1)
                copyto!(Ag, AT(A)); copyto!(Bg, AT(B))
            else
                Ag, Bg = AT(A), AT(B)
            end
            fn(side, uplo, trans, 'N', Float16(1), Ag, Bg)
            sync(Bg)
            @test norm(Float64.(Array(Bg)) - ref) / norm(ref) < 1e-2
        end
    end
end

# Complex. trsm!/trmm! above cover ComplexF32/ComplexF64 at n = 48 through each
# backend's own route; this goes past the TRSM threshold (n = 300) so the
# recursion's GEMM updates run in complex arithmetic, on whole arrays and on
# offset views, and on a GPU it also forces the portable method with invoke --
# the path AMD views and every backend without a vendor BLAS take. 'C' is the
# case that differs from 'T' only for complex.
@testset "complex: recursion, views and the portable path" begin
    GEN = Tuple{Char,Char,Char,Char,Number,Char,AbstractMatrix,AbstractMatrix}
    targets = [("CPU", Array, _ -> nothing, (false,));
               [(nm, AT, sy, (false, true)) for (nm, AT, sy) in available_gpu_backends()]]
    for (name, AT, sync, portables) in targets
        @testset "[$name] portable=$portable $T $fn side=$side uplo=$uplo trans=$trans view=$useview" for
                portable in portables,
                T        in (ComplexF32, ComplexF64),
                fn       in (trsm!, trmm!),
                side     in ('L', 'R'),
                uplo     in ('L', 'U'),
                trans    in ('N', 'T', 'C'),
                useview  in (false, true)

            n, m = 300, 5
            L = rand(T, n, n) ./ n
            A = (uplo == 'L' ? Matrix(LowerTriangular(L)) : Matrix(UpperTriangular(L))) +
                Diagonal(fill(T(2), n))
            B = side == 'L' ? rand(T, n, m) : rand(T, m, n)
            alpha = T(1.5 - 0.5im)

            ref = copy(B)
            blas = fn === trsm! ? LinearAlgebra.BLAS.trsm! : LinearAlgebra.BLAS.trmm!
            blas(side, uplo, trans, 'N', alpha, A, ref)

            if useview
                Abig = AT(zeros(T, n + 3, n + 2))
                Bbig = AT(zeros(T, size(B, 1) + 2, size(B, 2) + 2))
                Ag = view(Abig, 3:n+2, 2:n+1)
                Bg = view(Bbig, 2:size(B, 1)+1, 2:size(B, 2)+1)
                copyto!(Ag, AT(A)); copyto!(Bg, AT(B))
            else
                Ag, Bg = AT(A), AT(B)
            end
            if portable
                invoke(unified_rectrxm!, GEN, side, uplo, trans, 'N', alpha,
                       fn === trsm! ? 'S' : 'M', Ag, Bg)
            else
                fn(side, uplo, trans, 'N', alpha, Ag, Bg)
            end
            sync(Bg)
            tol = real(T) === Float32 ? 1e-4 : 1e-10
            @test norm(Array(Bg) - ref) / norm(ref) < tol
        end
    end
end

# trmm! with a separate output: C gets alpha * op(A) * B (or B * op(A)) and B is
# left alone; C = B is the in-place form. Checked on every backend against
# LinearAlgebra.BLAS.trmm! over side, uplo and transa, real and complex.
@testset "trmm! with a separate output" begin
    targets = [("CPU", Array, _ -> nothing); available_gpu_backends()]
    for (name, AT, sync) in targets
        @testset "[$name] $T side=$side uplo=$uplo trans=$trans" for
                T     in (Float32, Float64, ComplexF32, ComplexF64),
                side  in ('L', 'R'),
                uplo  in ('L', 'U'),
                trans in ('N', 'T', 'C')

            n, m = 24, 5
            L = T(0.1) .* rand(T, n, n)
            A = (uplo == 'L' ? Matrix(LowerTriangular(L)) : Matrix(UpperTriangular(L))) +
                Diagonal(fill(T(2), n))
            B = side == 'L' ? rand(T, n, m) : rand(T, m, n)
            alpha = T(1.5)
            ref = copy(B)
            LinearAlgebra.BLAS.trmm!(side, uplo, trans, 'N', alpha, A, ref)

            Ag, Bg = AT(A), AT(B)
            Cg = AT(zeros(T, size(B)))
            out = trmm!(side, uplo, trans, 'N', alpha, Ag, Bg, Cg)
            sync(Cg)
            tol = real(T) === Float32 ? 1e-5 : 1e-12
            @test out === Cg
            @test norm(Array(Cg) - ref) / norm(ref) < tol
            @test Array(Bg) == B                      # B untouched

            Bi = AT(B)                                 # C = B: the in-place form
            trmm!(side, uplo, trans, 'N', alpha, Ag, Bi, Bi)
            sync(Bi)
            @test norm(Array(Bi) - ref) / norm(ref) < tol
        end
    end
    @testset "rejects a C of the wrong size" begin
        A = Matrix(LowerTriangular(rand(4, 4))) + 2I
        @test_throws DimensionMismatch trmm!('L', 'L', 'N', 'N', 1.0, A, rand(4, 2), zeros(3, 2))
    end
end
