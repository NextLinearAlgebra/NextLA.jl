using LinearAlgebra
using LinearAlgebra.BLAS

function lu_recursive_nopiv!(A::StridedMatrix{T}, b::Int) where {T<:Real}
    m, n = size(A)
    if min(m, n) ≤ b
        # Leaf: local LU with pivoting only on diagonal 
        lu!(A)  # stores U in upper, L (unit) in strict lower
        return
    end

    # Split by columns: A = [A1 | A2]
    n1 = n ÷ 2
    A1 = @view A[:, 1:n1]        # panel (m × n1)
    A2 = @view A[:, n1+1:n]      # trailing (m × n2)

    # 1) Factor the left panel recursively: [A11; A21] ← L-panel + U11
    lu_recursive_nopiv!(A1, b)

    # Extract block views (all are contiguous range views)
    A11 = @view A[1:n1,     1:n1]      # contains L11 (lower, unit) + U11 (upper)
    A12 = @view A[1:n1,   n1+1:n]      # will become U12
    A21 = @view A[n1+1:m, 1:n1]        # holds L21 (already computed)
    A22 = @view A[n1+1:m, n1+1:n]

    # 2) U12 = L11^{-1} * A12  (Left, Lower, NoTrans, Unit)
    BLAS.trsm!('L','L','N','U', one(T), A11, A12)

    # 3) Schur update: A22 -= L21 * U12
    BLAS.gemm!('N','N', -one(T), A21, A12, one(T), A22)

    # 4) Recurse on trailing submatrix
    lu_recursive_nopiv!(A22, b)
    return
end

# Build a matrix that is safe for no-pivot LU (e.g., diagonally dominant)
n = 20480
A = randn(n,n)
A_diag_boost = 2.0 * n                        # make diagonally dominant
@inbounds @simd for i in 1:n
    A[i,i] += A_diag_boost
end
A0 = copy(A)

b = 128                                       # leaf size (tune per hardware)
lu_recursive_nopiv!(A, b)

L = UnitLowerTriangular(A)
U = UpperTriangular(A)
relerr = norm(L*U - A0) / norm(A0)
println("relative residual ≈ ", relerr)