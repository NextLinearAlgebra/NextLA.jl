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
