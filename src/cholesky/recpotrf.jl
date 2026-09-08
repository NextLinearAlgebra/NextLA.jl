using LinearAlgebra
using StochasticRounding
include("symmmixedprec.jl")
include("recmixedprectri.jl")
include("trsm.jl")
include("trmm.jl")
include("matmul.jl")
include("rectrxm.jl")
include("recsyrk.jl")
include("potrf.jl")
include("wrappers.jl")

"""
    potrf_recursive!(A, block_size)

Performs an in-place, nested recursive Cholesky factorization on the matrix `A`.
The recursion dynamically splits the matrix until the sub-block size is less than or 
equal to `block_size`, at which point it falls back to standard hardware BLAS/LAPACK routines.
"""
function potrf_recursive!(A, block_size)
    n = size(A, 1)

    if n <= block_size
        potrf!(A)
        return
    end

    n1 = 2^floor(Int, log2(n)) ÷ 2  
    
    A11 = @view A[1:n1, 1:n1]
    A21 = @view A[n1+1:end, 1:n1]
    A22 = @view A[n1+1:end, n1+1:end]

    potrf_recursive!(A11, block_size)

    if (eltype(A11) == Float16)
        unified_rectrxm!('R', 'L', 'T', 'N', 1.0, 'S', A11, A21)
    else
        trsm!('R', 'L', 'T', 'N', 1.0, A11, A21)
    end

    if (eltype(A21) == Float16)
        recsyrk!(-1.0, A21, 1.0, A22)
    else
        syrk!('L', 'N', -1.0, A21, 1.0, A22)
    end
    
    potrf_recursive!(A22, block_size)
end

"""
    reconstruct_matrix(A::SymmMixedPrec{T_Base})

Reconstructs a full dense matrix from the symmetric mixed-precision recursive block 
structure `A`. Used primarily for validation and returning to standard dense formats.
"""
function reconstruct_matrix(A::SymmMixedPrec{T_Base}) where {T_Base}
    if A.BaseCase !== nothing
        return copy(A.BaseCase)
    end
    
    C11 = reconstruct_matrix(A.A11)
    C22 = reconstruct_matrix(A.A22)
    C21 = A.OffDiag
    n1, m1 = size(C11)
    n2, m2 = size(C22)
    n = n1 + n2

    C_full = CuArray{T_Base}(undef, n, n)
    C_full[1:n1, 1:m1] .= C11
    C_full[n1+1:n, 1:m1] .= C21
    C_full[n1+1:n, m1+1:n] .= C22
    C_full[1:n1, m1+1:n] .= transpose(C21)

    return C_full
end

"""
    potrf_recursive!(A::SymmMixedPrec)

Performs an in-place, nested recursive Cholesky factorization on a symmetric mixed-precision 
matrix structure `A`. The recursion handles off-diagonal updates and falls back to standard 
hardware routines at the base case.
"""
function potrf_recursive!(A::SymmMixedPrec)
    if A.BaseCase !== nothing
        potrf_recursive!(A.BaseCase, 4096)
        return
    end

    potrf_recursive!(A.A11) 

    unified_rectrxm!('R', 'L', 'T', 'N', 1.0, 'S', TriMixedPrec(A.A11), A.OffDiag)

    recsyrk!(-1.0, A.OffDiag, 1.0, A.A22)

    potrf_recursive!(A.A22)
end