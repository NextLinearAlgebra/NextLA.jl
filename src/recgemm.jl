export recgemm!

function _gemm_dispatch!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    # Base hardware dispatch (Terminal Node)
    TA, TB, TC = eltype(A), eltype(B), eltype(C)
    
    if TA == TB == TC && TC in (Float32, Float64)
        CUBLAS.gemm!('N', 'N', TC(alpha), A, B, TC(beta), C)
    elseif TA == Float16 && TB == Float16 && TC in (Float16, Float32)
        CUBLAS.gemmEx!('N', 'N', alpha, A, B, beta, C)
    else
        A_final = (TA == TC) ? A : TC.(A)
        B_final = (TB == TC) ? B : TC.(B)
        if TC in (Float32, Float64)
            CUBLAS.gemm!('N', 'N', TC(alpha), A_final, B_final, TC(beta), C)
        else
            CUBLAS.gemmEx!('N', 'N', alpha, A_final, B_final, beta, C)
        end
    end
end

function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    _gemm_dispatch!(alpha, A, B, beta, C)
end

function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec; parallel::Bool=(size(C, 1) > 512))
    # Base case check
    if C.BaseCase !== nothing
        _gemm_dispatch!(alpha, A, B, beta, C.BaseCase)
        return
    end

    n = size(C, 1)
    mid = size(C.A11, 1)

    # View generation for A
    A11 = view(A, 1:mid, 1:mid)
    A12 = view(A, 1:mid, mid+1:n)
    A21 = view(A, mid+1:n, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)

    # View generation for B
    B11 = view(B, 1:mid, 1:mid)
    B12 = view(B, 1:mid, mid+1:n)
    B21 = view(B, mid+1:n, 1:mid)
    B22 = view(B, mid+1:n, mid+1:n)

    if parallel
        @sync begin
            @async begin
                recgemm!(alpha, A11, B11, beta, C.A11; parallel=false)
                recgemm!(alpha, A12, B21, 1.0, C.A11; parallel=false)
            end
            @async begin
                _gemm_dispatch!(alpha, A11, B12, beta, C.A12)
                _gemm_dispatch!(alpha, A12, B22, 1.0, C.A12)
            end
            @async begin
                _gemm_dispatch!(alpha, A21, B11, beta, C.A21)
                _gemm_dispatch!(alpha, A22, B21, 1.0, C.A21)
            end
            @async begin
                recgemm!(alpha, A21, B12, beta, C.A22; parallel=false)
                recgemm!(alpha, A22, B22, 1.0, C.A22; parallel=false)
            end
        end
    else
        # Sequential updates
        recgemm!(alpha, A11, B11, beta, C.A11; parallel=false)
        recgemm!(alpha, A12, B21, 1.0, C.A11; parallel=false)

        _gemm_dispatch!(alpha, A11, B12, beta, C.A12)
        _gemm_dispatch!(alpha, A12, B22, 1.0, C.A12)

        _gemm_dispatch!(alpha, A21, B11, beta, C.A21)
        _gemm_dispatch!(alpha, A22, B21, 1.0, C.A21)

        recgemm!(alpha, A21, B12, beta, C.A22; parallel=false)
        recgemm!(alpha, A22, B22, 1.0, C.A22; parallel=false)
    end
end
