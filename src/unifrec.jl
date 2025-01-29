include("matmul.jl")
include("trsm_base_cases.jl")
include("trmm_base_cases.jl")


function unified_rec(func::Char, side::Char, uplo::Char, A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, threshold::Int=256) where T <: AbstractFloat
    if n <= threshold
        if func == 'S'
            if side == 'L' && uplo == 'L'
                LeftLowerTRSM!(A, B)
            elseif side == 'L' && uplo == 'U'
                LeftUpperTRSM!(A, B)
            elseif side == 'R' && uplo == 'L'
                RightLowerTRSM!(A, B)
            else
                RightUpperTRSM!(A, B)
            end
        else
            if side == 'L' && uplo == 'L'
                LeftLowerTRMM!(A, B)
            elseif side == 'L' && uplo == 'U'
                LeftUpperTRMM!(A, B)
            elseif side == 'R' && uplo == 'L'
                RightLowerTRMM!(A, B)
            else
                RightUpperTRMM!(A, B)
            end
        end
        return B
    end

    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end
    mid_remainder = n - mid

    A11 = view(A, 1:mid, 1:mid)
    A22 = view(A, mid+1:n, mid+1:n)
    A21 = view(A, mid+1:n, 1:mid)
    A12 = view(A, 1:mid, mid+1:n)

    if side == 'L'
        B1 = view(B, 1:mid, :)
        B2 = view(B, mid+1:n, :)
    else
        B1 = view(B, :, 1:mid)
        B2 = view(B, :, mid+1:n)
    end

    if (side == 'L' && uplo == 'L' && func == 'S') || 
        (side == 'R' && uplo == 'U' && func == 'S') || 
        (side == 'L' && uplo == 'U' && func == 'M') || 
        (side == 'R' && uplo == 'L' && func == 'M')
        unified_rec(func, side, uplo, A11, mid, B1, threshold)
        if side == 'L'
            if func == 'S'
                GEMM_SUB!(B2, A21, B1)
            else
                GEMM_ADD!(A12, B2, B1)
            end
        else
            if func == 'S'
                GEMM_SUB!(B2, B1, A12)
            else
                GEMM_ADD!(B2, A21, B1)
            end
        end
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold)
    else
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold)
        if side == 'L'
            if func == 'S'
                GEMM_SUB!(B1, A12, B2)
            else
                GEMM_ADD!(A21, B1, B2)
            end
        else
            if func == 'S'
                GEMM_SUB!(B1, B2, A21)
            else
                GEMM_ADD!(B1, A12, B2)
            end
        end
        unified_rec(func, side, uplo, A11, mid, B1, threshold)
    end
    return B
end
