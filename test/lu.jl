@testset "lu! with varing pivot" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for pivot in [NoPivot(), RowNonZero(), CompletePivoting()]
            for m in [10, 100, 1000]
                for n in [m, div(m,10)*9, div(m,10)*11]
                    # Use smaller matrices for Float32 to avoid numerical issues
                    
                    # Better conditioning for rectangular matrices
                    params = parameter_creation("GE", 3, 100, 100)
                    A = matrix_generation(T, m, n; 
                        mode=params.mode, 
                        cndnum=params.cndnum,
                        anorm=params.anorm,
                        kl=params.kl,
                        ku=params.ku)

                    B = deepcopy(A)
                    DLA_A = NextLAMatrix{T}(A)
                    F = LinearAlgebra.lu!(DLA_A, pivot)
                    m, n = size(F.factors)
                    L = tril(F.factors[1:m, 1:min(m,n)])
                    for i in 1:min(m,n); L[i,i] = 1 end
                    U = triu(F.factors[1:min(m,n), 1:n])
                    p = LinearAlgebra.ipiv2perm(F.ipiv,m)
                    q = LinearAlgebra.ipiv2perm(F.jpiv, n)
                   
                    # Calculate relative error
                    reconstructed = L * U
                    original = B[p, q]
                    @test L * U ≈ B[p, q]
                    @test norm(reconstructed) ≈ norm(original)
                    @test norm(reconstructed - original) / norm(original) ≈ 0.0 atol=1e-6 
                end
            end
        end
    end
end

@testset "lu generic_lufact!" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for pivot in [NoPivot(), RowNonZero(),  RowMaximum(), CompletePivoting()]
            for m in [10, 100, 1000]
                for n in [m, div(m,10)*9, div(m,10)*11]
                    params = parameter_creation("GE", 3, 100, 100)
                    A = matrix_generation(T, m, n; 
                        mode=params.mode, 
                        cndnum=params.cndnum,
                        anorm=params.anorm,
                        kl=params.kl,
                        ku=params.ku)
                    B = deepcopy(A)
                    DLA_A = NextLAMatrix{T}(A)
                    F =LinearAlgebra.generic_lufact!(DLA_A, pivot)
                    m, n = size(F.factors)
                    L = tril(F.factors[1:m, 1:min(m,n)])
                    for i in 1:min(m,n); L[i,i] = 1 end
                    U = triu(F.factors[1:min(m,n), 1:n])
                    p = LinearAlgebra.ipiv2perm(F.ipiv,m)
                    q = LinearAlgebra.ipiv2perm(F.jpiv, n)
                    @test L * U ≈ B[p, q]
                    @test norm(L * U) ≈ norm(B[p, q])
                end
            end
        end
    end
end

@testset "lu! default RowMaximum()" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for m in [10, 100, 1000]
            for n in [m, div(m,10)*9, div(m,10)*11]
                params = parameter_creation("GE", 3, 100, 100)
                    A = matrix_generation(T, m, n; 
                        mode=params.mode, 
                        cndnum=params.cndnum,
                        anorm=params.anorm,
                        kl=params.kl,
                        ku=params.ku)
                B = deepcopy(A)
                DLA_A = NextLAMatrix{T}(A)
                F =LinearAlgebra.lu!(DLA_A)
                m, n = size(F.factors)
                L = tril(F.factors[1:m, 1:min(m,n)])
                for i in 1:min(m,n); L[i,i] = 1 end
                U = triu(F.factors[1:min(m,n), 1:n])
                p = LinearAlgebra.ipiv2perm(F.ipiv,m)
                @test L * U ≈ B[p, :]
                @test norm(L * U) ≈ norm(B[p, :])
            end
        end
    end
end

# getrf2! reports through its return value, on every path.
#
# Each early exit used to `return` bare, so the caller's
# `A, ipiv, info = getrf2!(...)` raised MethodError: no method matching
# iterate(::Nothing) -- a 0-by-0 input was enough. The two singular paths also
# wrote `info[] = 1` to an Int, which is a MethodError of its own, and the two
# recursive calls dropped the info they computed, so a singular block never
# reached the caller.
@testset "getrf2! returns (A, ipiv, info) on every path" begin
    for T in (Float64, ComplexF64)
        # quick return: the path that crashed the caller
        A = zeros(T, 0, 0)
        Af, ipiv, info = NextLA.getrf2!(A, Int[], 0)
        @test Af === A
        @test info == 0

        # m == 1 with a zero pivot: singular, reported through info
        A = zeros(T, 1, 1)
        _, _, info = NextLA.getrf2!(A, Vector{Int}(undef, 1), 0)
        @test info == 1

        # n == 1, all zero: the other single-column exit
        A = zeros(T, 4, 1)
        _, _, info = NextLA.getrf2!(A, Vector{Int}(undef, 1), 0)
        @test info == 1

        # a singular block inside the recursion: info is the failing column,
        # offset by the split, not lost
        A = T[1 0; 0 0]
        _, _, info = NextLA.getrf2!(A, Vector{Int}(undef, 2), 0)
        @test info == 2

        # and a non-singular matrix still factors, with info == 0
        A0 = T[4 1; 1 3]
        A = copy(A0)
        Af, ipiv, info = NextLA.getrf2!(A, Vector{Int}(undef, 2), 0)
        @test info == 0
        L = UnitLowerTriangular(Af)
        U = UpperTriangular(Af)
        @test L * U ≈ A0[LinearAlgebra.ipiv2perm(ipiv, 2), :]

        # the caller reports it as LAPACK does rather than crashing
        F = LinearAlgebra.lu!(NextLAMatrix{T}(T[1 0; 0 0]), RowMaximum(); check=false)
        @test F.info == 2
        @test_throws LinearAlgebra.SingularException LinearAlgebra.lu!(
            NextLAMatrix{T}(T[1 0; 0 0]), RowMaximum())
    end
end
