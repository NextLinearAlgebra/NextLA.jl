export recgemm!
include("wrappers.jl")

# User-facing flags: 'N' = normal, 'Y' = transpose
_blas_trans(flag::Char) =
    flag in ('N', 'n') ? 'N' :
    flag in ('Y', 'y') ? 'T' :
    throw(ArgumentError("transpose flag must be 'N' or 'Y', got $flag"))

function _gemm_blocks(A::AbstractMatrix, trans::Char, mid::Int, n::Int)
    A11 = @view A[1:mid,     1:mid]
    A12 = @view A[1:mid,     mid+1:n]
    A21 = @view A[mid+1:n,   1:mid]
    A22 = @view A[mid+1:n,   mid+1:n]

    # returned block remains physically untransposed
    # trans value is passed to GEMM to apply the operation
    return trans == 'N' ? (A11, A12, A21, A22) :
                          (A11, A21, A12, A22)
end

function _gemm_dispatch!(transA::Char, transB::Char, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    TA, TB, TC = eltype(A), eltype(B), eltype(C)

    if TA == TB == TC && TC in (Float32, Float64)
        gemm!(transA, transB, TC(alpha), A, B, TC(beta), C)
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        gemmEx!('N', 'N', alpha, A, B, beta, C)
    else
        A_final = (TA == TC) ? A : TC.(A)
        B_final = (TB == TC) ? B : TC.(B)
        if TC in (Float32, Float64)
            gemm!(transA, transB, TC(alpha), A_final, B_final, TC(beta), C)
        else
            gemmEx!(transA, transB, alpha, A_final, B_final, beta, C)
        end
    end
end

function recgemm!(transA::Char, transB::Char, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    _gemm_dispatch!(_blas_trans(transA), _blas_trans(transB), alpha, A, B, beta, C)
end

function recgemm!(transA::Char, transB::Char, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec; parallel::Bool=(size(C, 1) > 512))
    
    blasA = _blas_trans(transA)
    blasB = _blas_trans(transB)
    
    if C.BaseCase !== nothing
        _gemm_dispatch!(blasA, blasB, alpha, A, B, beta, C.BaseCase)
        return
    end

    n = size(C, 1)
    mid = size(C.A11, 1)

    A11 = view(A, 1:mid, 1:mid)
    A12 = view(A, 1:mid, mid+1:n)
    A21 = view(A, mid+1:n, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)

    B11 = view(B, 1:mid, 1:mid)
    B12 = view(B, 1:mid, mid+1:n)
    B21 = view(B, mid+1:n, 1:mid)
    B22 = view(B, mid+1:n, mid+1:n)

    if parallel
        @sync begin
            @async begin
                recgemm!(blasA, blasB, alpha, A11, B11, beta, C.A11; parallel=false)
                recgemm!(blasA, blasB, alpha, A12, B21, 1.0, C.A11; parallel=false)
            end
            @async begin
                _gemm_dispatch!(blasA, blasB, alpha, A11, B12, beta, C.A12)
                _gemm_dispatch!(blasA, blasB, alpha, A12, B22, 1.0, C.A12)
            end
            @async begin
                _gemm_dispatch!(blasA, blasB, alpha, A21, B11, beta, C.A21)
                _gemm_dispatch!(blasA, blasB, alpha, A22, B21, 1.0, C.A21)
            end
            @async begin
                recgemm!(blasA, blasB, alpha, A21, B12, beta, C.A22; parallel=false)
                recgemm!(blasA, blasB, alpha, A22, B22, 1.0, C.A22; parallel=false)
            end
        end
    else
        # Sequential updates
        recgemm!(blasA, blasB, alpha, A11, B11, beta, C.A11; parallel=false)
        recgemm!(blasA, blasB, alpha, A12, B21, 1.0, C.A11; parallel=false)

        _gemm_dispatch!(blasA, blasB, alpha, A11, B12, beta, C.A12)
        _gemm_dispatch!(blasA, blasB, alpha, A12, B22, 1.0, C.A12)

        _gemm_dispatch!(blasA, blasB, alpha, A21, B11, beta, C.A21)
        _gemm_dispatch!(blasA, blasB, alpha, A22, B21, 1.0, C.A21)

        recgemm!(blasA, blasB, alpha, A21, B12, beta, C.A22; parallel=false)
        recgemm!(blasA, blasB, alpha, A22, B22, 1.0, C.A22; parallel=false)
    end
end
