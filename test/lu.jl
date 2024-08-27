@testset "lu! with varing pivot" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for pivot in [NoPivot(), RowNonZero(), CompletePivoting()]
            for m in [10, 100, 1000]
                for n in [m, div(m,10)*9, div(m,10)*11]
                    A = rand(T, m, n)
                    B = copy(A)
                    DLA_A = DLAMatrix{T}(A)
                    F =LinearAlgebra.lu!(DLA_A, pivot)
                    m, n = size(F.factors)
                    L = tril(F.factors[1:m, 1:min(m,n)])
                    for i in 1:min(m,n); L[i,i] = 1 end
                    U = triu(F.factors[1:min(m,n), 1:n])
                    p = LinearAlgebra.ipiv2perm(F.ipiv,m)
                    q = LinearAlgebra.ipiv2perm(F.jpiv, n)
                    L * U ≈ B[p, q]
                    norm(L * U) ≈ norm(B[p, q])
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
                    A = rand(T, m, n)
                    B = copy(A)
                    DLA_A = DLAMatrix{T}(A)
                    F =LinearAlgebra.generic_lufact!(DLA_A, pivot)
                    m, n = size(F.factors)
                    L = tril(F.factors[1:m, 1:min(m,n)])
                    for i in 1:min(m,n); L[i,i] = 1 end
                    U = triu(F.factors[1:min(m,n), 1:n])
                    p = LinearAlgebra.ipiv2perm(F.ipiv,m)
                    q = LinearAlgebra.ipiv2perm(F.jpiv, n)
                    L * U ≈ B[p, q]
                    norm(L * U) ≈ norm(B[p, q])
                end
            end
        end
    end
end


@testset "lu! default RowMaximum()" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for m in [10, 100, 1000]
            for n in [m, div(m,10)*9, div(m,10)*11]
                A = rand(T, m, n)
                B = copy(A)
                DLA_A = DLAMatrix{T}(A)
                F =LinearAlgebra.lu!(DLA_A)
                m, n = size(F.factors)
                L = tril(F.factors[1:m, 1:min(m,n)])
                for i in 1:min(m,n); L[i,i] = 1 end
                U = triu(F.factors[1:min(m,n), 1:n])
                p = LinearAlgebra.ipiv2perm(F.ipiv,m)
                L * U ≈ B[p, :]
                norm(L * U) ≈ norm(B[p, :])
            end
        end
    end
end