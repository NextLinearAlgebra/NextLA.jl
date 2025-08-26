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