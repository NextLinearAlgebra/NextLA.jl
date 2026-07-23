using LinearAlgebra
include("wrappers.jl")
include("matmul.jl")
include("rectrxm.jl")
include("fullmixedprec.jl")

export lu_nopiv_recursive!

function lu_basecase_nopiv!(A::AbstractMatrix)
    getrf!(A)
    return A
end

function lu_nopiv_recursive!(A::AbstractMatrix, block_size::Int=256)
    n, m = size(A)
    @assert n == m "LU requires a square matrix"

    if n <= block_size
        # lu_basecase_nopiv!(A)
        CUSOLVER.getrf!(A)
        return A
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    A11 = @view A[1:mid,     1:mid]
    A12 = @view A[1:mid,     mid+1:n]
    A21 = @view A[mid+1:n,   1:mid]
    A22 = @view A[mid+1:n,   mid+1:n]

    lu_nopiv_recursive!(A11, block_size)

    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A11, A12)

    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A11, A21)

    recgemm!(-1.0f0, A21, A12, 1.0f0, A22)

    lu_nopiv_recursive!(A22, block_size)

    return A
end

function lu_nopiv_recursive!(A::FullMixedPrec, block_size::Int=4096)
    if A.BaseCase !== nothing
        lu_nopiv_recursive!(A.BaseCase, block_size)
        return A
    end

    lu_nopiv_recursive!(A.A11, block_size)

    unified_rectrxm!('L', 'L', 'N', 'U', 1.0f0, 'S', A.A11, A.A12)

    unified_rectrxm!('R', 'U', 'N', 'N', 1.0f0, 'S', A.A11, A.A21)

    recgemm!(-1.0f0, A.A21, A.A12, 1.0f0, A.A22)

    lu_nopiv_recursive!(A.A22, block_size)

    return A
end