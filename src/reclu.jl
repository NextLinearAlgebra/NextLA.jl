include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("trsm.jl")
include("trmm.jl")
include("matmul.jl")
include("rectrxm.jl")
include("recsyrk.jl")
include("cholesky.jl")


function lu_recursive_impl!(A, b::Int)
    m, n = size(A)

    if min(m, n) <= b
        CUSOLVER.getrf!(A) # <-- this didnt work either
        # CUSOLVER.lu!(A)
        return
    end

    n1 = n ÷ 2
    A11 = @view A[1:n1,     1:n1]
    A12 = @view A[1:n1,   n1+1:n]
    A21 = @view A[n1+1:m, 1:n1]
    A22 = @view A[n1+1:m, n1+1:n]

    backend = KernelAbstractions.get_backend(A)
    lu_recursive_impl!(A11, b)
    KernelAbstractions.synchronize(backend)
    CUBLAS.trsm!('L', 'L', 'N', 'U', one(eltype(A)), A11, A12)
    KernelAbstractions.synchronize(backend)
    CUBLAS.trsm!('R', 'U', 'N', 'N', one(eltype(A)), A11, A21)
    KernelAbstractions.synchronize(backend)
    CUBLAS.gemm!('N', 'N', -one(eltype(A)), A21, A12, one(eltype(A)), A22)
    KernelAbstractions.synchronize(backend)
    lu_recursive_impl!(A22, b)
    KernelAbstractions.synchronize(backend)

    return
end



function lu_recursive_nopiv_diagfirst!(A::StridedMatrix{T}, b::Int) where {T<:Real}
    m, n = size(A)
    if min(m, n) ≤ b
        lu!(A)  # leaf: no-pivot LU
        return
    end

    # Column split: A = [A1 | A2], then 2×2 blocks
    n1 = n ÷ 2
    A11 = @view A[1:n1,     1:n1]
    A12 = @view A[1:n1,   n1+1:n]
    A21 = @view A[n1+1:m, 1:n1]
    A22 = @view A[n1+1:m, n1+1:n]

    # 1) Factor diagonal block A11 = L11*U11 (no pivot)
    lu_recursive_nopiv_diagfirst!(A11, b)

    # 2) U12 = L11^{-1} * A12  (Left, Lower, NoTrans, Unit)
    BLAS.trsm!('L','L','N','U', one(T), A11, A12)

    # 3) L21 = A21 * U11^{-1}  (Right, Upper, NoTrans, NonUnit)
    BLAS.trsm!('R','U','N','N', one(T), A11, A21)

    # 4) Schur update: A22 -= L21 * U12
    BLAS.gemm!('N','N', -one(T), A21, A12, one(T), A22)

    # 5) Recurse on trailing block
    lu_recursive_nopiv_diagfirst!(A22, b)
    return
end


function diag_dominant(n; boost=2.0*n)
    A = randn(Float64, n, n)
    @inbounds for i in 1:n
        A[i,i] += boost
    end
    return A
end

println("CPU Recursive LU Test")
n, b = 1024, 128
A0 = diag_dominant(n)
A  = copy(A0)
lu_recursive_nopiv_diagfirst!(A, b)
L = UnitLowerTriangular(A); U = UpperTriangular(A)
println("rel residual (cpu) = ", norm(L*U - A0)/norm(A0))

println("GPU Recursive LU Test")
n = 1024
b = 128

println("Creating a $(n)x$(n) matrix on the CPU")
A0_cpu = diag_dominant(n)

println("Moving matrix data to the GPU")
A_gpu = CuArray(A0_cpu)
A0_gpu = copy(A_gpu)

println("Running LU")
CUDA.@sync lu_recursive_impl!(A_gpu, b)

# display(A_gpu)
# display(L)
# display(U)
# display(A0_gpu)

println("Verifying the result")
L_gpu = UnitLowerTriangular(A_gpu)
U_gpu = UpperTriangular(A_gpu)
# display(L_gpu)
# display(U_gpu)
# display(Array(L_gpu)*Array(U_gpu))

A_result = Array(L_gpu) * Array(U_gpu)
residual_norm = norm(A_result - Array(A0_gpu))
initial_norm = norm(A0_gpu)
rel_err = residual_norm / initial_norm

@printf("\nRelative residual ≈ %.2e\n", rel_err)


