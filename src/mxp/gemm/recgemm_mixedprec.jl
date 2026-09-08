function recgemm!(alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::FullMixedPrec; parallel::Bool=(size(C, 1) > 512))
    if C.BaseCase !== nothing
        _gemm_dispatch!(alpha, A, B, beta, C.BaseCase)
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
