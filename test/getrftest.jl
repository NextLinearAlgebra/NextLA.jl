using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools
using Test


function gen_getrf2_test_rand(::Type{T}, m, n) where {T<: Number}
    A = rand(T,m,n)
    ipiv = ones(Int, min(m,n))
    info = Ref{Int}(0)

    for i in 1:min(m,n)
        ipiv[i] = Int(i)
    end

    B = deepcopy(A)
    bpiv = deepcopy(ipiv)
    C = deepcopy(A)

    #display(A)

    getrf2!(A, ipiv, info)
    lapack_getrf2!(T, B, bpiv)

    #display(A)
    #println("-----")
    #display(B)

    #println(norm(B - A) / norm(B))
    #println(norm(ipiv - bpiv))
    return norm(A - B) / norm(B)
end

#gen_getrf2_test_rand(ComplexF64,2197,1698)

@testset "datatype = $T" for (T,tol) in [(Float64, 1e-14), (Float32, 1e-6)]
    for m in [1000, 2197, 4096, 9214]
        for n in [1000, 2197, 4096, 9482]
            @test gen_getrf2_test_rand(T,m,n) ≈ 0 atol=tol
        end
    end
end
