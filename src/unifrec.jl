include("matmul.jl")
include("trsm_base_cases.jl")
include("trmm_base_cases.jl")

function unified_rec(func::Char, side::Char, uplo::Char, A::AbstractMatrix{T}, n, B::AbstractMatrix{T}, threshold::Int=256, depth::Int=1) where T <: AbstractFloat
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
        unified_rec(func, side, uplo, A11, mid, B1, threshold, depth + 1)
        if side == 'L'
            if func == 'S'
                if depth >= 2
                    B2_copy = Float16.(B2)
                    GEMM_SUB!(B2_copy, Float16.(A21), Float16.(B1)) # WANT TO USE C (B2 IN THIS CASE) IN FP 32 AND A AND B IN FP 16 
                    copy!(B2, B2_copy)
                else
                    partition_size = div(size(A21, 1), 2)
            
                    X11 = view(A21, 1:partition_size, 1:partition_size)
                    X12 = view(A21, 1:partition_size, partition_size+1:size(A21, 2))
                    X21 = view(A21, partition_size+1:size(A21, 1), 1:partition_size)
                    X22 = view(A21, partition_size+1:size(A21, 1), partition_size+1:size(A21, 2))
            
                    X11 = Float32.(X11)
                    X21 = Float32.(X21)
                    X22 = Float32.(X22)
            
                    B11 = view(B1, 1:partition_size, :)
                    B12 = view(B1, partition_size+1:size(B1, 1), :)
            
                    B11 = Float32.(B11)
                    B12 = Float32.(B12)
            
                    C1 = view(B2, 1:partition_size, :)
                    C2 = view(B2, partition_size+1:size(B2, 1), :)
                    C1_copy = Float32.(C1)
                    C2_copy = Float32.(C2)
            
                    GEMM_SUB!(C1_copy, X11, B11)
                    GEMM_SUB!(C1_copy, X12, B12)
                    GEMM_SUB!(C2_copy, X21, B11)
                    GEMM_SUB!(C2_copy, X22, B12)
            
                    copy!(C1, C1_copy)
                    copy!(C2, C2_copy)
                end
            else
                if depth >= 2
                    GEMM_ADD!(Float32.(A12), B2, B1)
                else
                    GEMM_ADD!(A12, B2, B1)
                end
            end
        else
            if func == 'S'
                if depth >= 2
                    GEMM_SUB!(B2, B1, Float32.(A12))
                else
                    GEMM_SUB!(B2, B1, A12)
                end
            else
                if depth >= 2
                    GEMM_ADD!(B2, Float32.(A21), B1)
                else
                    GEMM_ADD!(B2, A21, B1)
                end
            end
        end
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold, depth + 1)
    else
        unified_rec(func, side, uplo, A22, mid_remainder, B2, threshold, depth + 1)
        if side == 'L'
            if func == 'S'
                if depth >= 2
                    GEMM_SUB!(B1, Float32.(A12), B2)
                else
                    GEMM_SUB!(B1, A12, B2)
                end
            else
                if depth >= 2
                    GEMM_ADD!(Float32.(A21), B1, B2)
                else
                    GEMM_ADD!(A21, B1, B2)
                end
            end
        else
            if func == 'S'
                if depth >= 2
                    GEMM_SUB!(B1, B2, Float32.(A21))
                else
                    GEMM_SUB!(B1, B2, A21)
                end
            else
                if depth >= 2
                    GEMM_ADD!(B1, Float32.(A12), B2)
                else
                    GEMM_ADD!(B1, A12, B2)
                end
            end
        end
        unified_rec(func, side, uplo, A11, mid, B1, threshold, depth + 1)
    end
    return B
end