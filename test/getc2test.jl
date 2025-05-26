using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools


function gen_getc2_test_rand(::Type{T}, n) where T
    A = rand(T,n,n)

    ipiv = ones(Int, n)
    jpiv = ones(Int, n)
    info = Ref{Int}(0)

    for i in 1:n
        ipiv[i] = Int(i)
        jpiv[i] = Int(i)
    end

    B = deepcopy(A)
    bipiv = deepcopy(ipiv)
    bjpiv = deepcopy(jpiv)

    getc2!(A,ipiv,jpiv,info)
    lapack_getc2!(B,bipiv,bjpiv)

    println("lapack vs myfunc ", norm(B - A) / norm(B))
    println("maximum difference between i pivots ", maximum(abs.(ipiv - bipiv)))
    println("maximum difference between j pivots ", maximum(abs.(jpiv - bjpiv)))

    #return max(maximum(abs.(ipiv - bipiv)), maximum(abs.(jpiv - bjpiv)))
    return norm(B - A) / norm(B)
end

"""
# count how many times pivot differs
cnt = 0

for i in 1:100
    t1 = gen_getc2_test_rand(ComplexF64, 500)
    t2 = gen_getc2_test_rand(ComplexF32, 500)

    if t1 > 0 
        cnt += 1
    end

    if t2 > 0
        cnt += 1
    end
end

@show cnt
"""

# general testing 
gen_getc2_test_rand(ComplexF64, 500)
gen_getc2_test_rand(Float64, 500)
gen_getc2_test_rand(ComplexF32, 500)
gen_getc2_test_rand(Float32, 500)

gen_getc2_test_rand(ComplexF64, 1000)
gen_getc2_test_rand(Float64, 1000)
gen_getc2_test_rand(ComplexF32, 1000)
gen_getc2_test_rand(Float32, 1000)

gen_getc2_test_rand(ComplexF64, 2000)
gen_getc2_test_rand(Float64, 2000)
gen_getc2_test_rand(ComplexF32, 2000)
gen_getc2_test_rand(Float32, 2000)

"""
# test that has 0.0 rounding issues
n = 5
A = rand(Float64, n,n)
ipiv = ones(Int, n)
jpiv = ones(Int, n)

for i in 1:n
    ipiv[i] = i
    jpiv[i] = i

    for j in 1:n
        A[i,j] = i + j + 0.0
    end
end

B = deepcopy(A)
bipiv = deepcopy(ipiv)
bjpiv= deepcopy(jpiv)

info = Ref{Int}(0)

getc2!(A,ipiv,jpiv,info)
lapack_getc2!(B,bipiv,bjpiv)
println("lapack vs myfunc ", norm(B - A) / norm(B))

display(A)
display(B)

display(ipiv)
display(bipiv)
display(jpiv)
display(bjpiv)
"""

"""
# testing if smlnum is same 
n = 1
A = rand(ComplexF32, n,n)
ipiv = ones(Int, n)
jpiv = ones(Int, n)

for i in 1:n
    ipiv[i] = i
    jpiv[i] = i
end

A[1,1] = zero(ComplexF32)

B = deepcopy(A)
bipiv = deepcopy(ipiv)
bjpiv= deepcopy(jpiv)

info = Ref{Int}(0)

getc2!(A,ipiv,jpiv,info)
lapack_getc2!(B,bipiv,bjpiv)

println("lapack vs myfunc ", norm(B - A) / norm(B))

display(A)
display(B)
display(ipiv)
display(bipiv)
display(jpiv)
display(bjpiv)
"""
