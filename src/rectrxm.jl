export unified_rectrxm!


function quantize(matrix::AbstractMatrix{T}) where T <: AbstractFloat
    FP16_MAX_VAL = 65504.0f0
    alpha = maximum(abs, matrix) 
    
    if iszero(alpha)
        return similar(matrix, Float16), 1.0f0
    end

    if alpha > FP16_MAX_VAL
        s = Float32(alpha / FP16_MAX_VAL)
        
        quantized_matrix = similar(matrix, Float16, size(matrix))
        
        @. quantized_matrix = Float16(round(clamp(matrix / s, -FP16_MAX_VAL, FP16_MAX_VAL)))
    else
        s = 1.0f0

        quantized_matrix = similar(matrix, Float16, size(matrix))
        
        @. quantized_matrix = Float16(matrix)
    end

    return quantized_matrix, s
end

function dequantize(quantized_matrix::AbstractMatrix{Float16}, s::Float32, original_eltype::DataType)
    dequantized_matrix = similar(quantized_matrix, original_eltype, size(quantized_matrix))
    
    @. dequantized_matrix = quantized_matrix * s

    return dequantized_matrix
end


function GEMM_ADD_cublas!(A, B, C, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    if eltype(A_mat) == Float16 && eltype(B_mat) == Float16
        if eltype(C) == Float16
            C_op = Float32.(C)
            CUBLAS.gemmEx!(transA, transB, scale, A_mat, B_mat, 1.0f0, C_op)
            clamp!(C_op, floatmin(Float16), floatmax(Float16))
            copy!(C, C_op)
        else
            CUBLAS.gemmEx!(transA, transB, scale, A_mat, B_mat, 1.0f0, C)
        end
    else
        T_C = eltype(C)
        CUBLAS.gemm!(transA, transB, T_C(scale), A_mat, B_mat, T_C(1.0), C)
    end
end

function GEMM_SUB_cublas!(C, A, B, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    if eltype(A_mat) == Float16 && eltype(B_mat) == Float16
        if eltype(C) == Float16
            C_op = Float32.(C)
            CUBLAS.gemmEx!(transA, transB, -scale, A_mat, B_mat, 1.0f0, C_op)
            clamp!(C_op, floatmin(Float16), floatmax(Float16))
            copy!(C, C_op)
        else
            CUBLAS.gemmEx!(transA, transB, -scale, A_mat, B_mat, 1.0f0, C)
        end
    else
        T_C = eltype(C)
        CUBLAS.gemm!(transA, transB, T_C(-scale), A_mat, B_mat, T_C(1.0), C)
    end
end



"""
Unified recursive function for triangular matrix solve (TRSM) and multiply (TRMM) operations.

This function supports both solving triangular systems of equations and performing triangular matrix multiplications.

Arguments:
- side::Char: Specifies the side of the operation:
    - 'L': Left multiplication (A * B or inv(A) * B).
    - 'R': Right multiplication (B * A or B * inv(A)).
- uplo::Char: Specifies the triangular part of the matrix to reference:
    - 'U': Use the upper triangle.
    - 'L': Use the lower triangle.
- transpose::Char: Specifies the transposition operation:
    - 'N': No transpose.
    - 'T': Transpose.
    - 'C': Conjugate transpose.
- alpha::Number: Scalar multiplier applied to the operation.
- func::Char: Specifies the function type:
    - 'S': Solve (TRSM, A * X = alpha * B).
    - 'M': Multiply (TRMM, Update B = alpha * A * B or alpha * B * A).
- A::AbstractMatrix: The triangular matrix.
- B::AbstractMatrix: The matrix to multiply or solve for.

Returns:
- Updated matrix `B` after performing the specified operation.

Notes:
- The function modifies `B` in place.
"""
function unified_rectrxm!(
        side::Char, 
        uplo::Char, 
        transpose::Char, 
        alpha::Number, 
        func::Char, 
        A::AbstractMatrix, 
        B::AbstractMatrix
    )
    threshold = 16
    n = size(A, 1)

    if transpose == 'T' || transpose == 'C'
        A = (transpose == 'T') ? Transpose(A) : Adjoint(A)
        uplo = (uplo == 'L') ? 'U' : 'L'
    end    
    
    if func == 'S'
        threshold = 256
        B .= alpha .* B
    end
    unified_rec(func, side, uplo, A, B, threshold)
    if func == 'M'
        B .= alpha .* B
    end
    return B
end


# unified rec for transposed matrices
function unified_rec(func::Char, side::Char, uplo::Char,
    A::Transpose{T, M},
    B::StridedMatrix{T}, threshold::Int=256;
    A_scale::Float32=1.0f0
) where {T <: AbstractFloat, M <: AbstractMatrix{T}}

    A_orig = parent(A)
    n = size(A, 1)

    if n <= threshold
        if func == 'S'
            CUBLAS.trsm!(side, uplo, 'N', 'N', one(T), copy(A), B)
        else 
            CUBLAS.trmm!(side, uplo, 'N', 'N', one(T), copy(A), B)
        end
        return B
    end

    mid = isinteger(log2(n)) ? div(n, 2) : 2^floor(Int, log2(n))

    A11 = transpose(view(A_orig, 1:mid,     1:mid))
    A22 = transpose(view(A_orig, mid+1:n,   mid+1:n))
    A21 = transpose(view(A_orig, 1:mid,     mid+1:n)) 
    A12 = transpose(view(A_orig, mid+1:n,   1:mid))  

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

        unified_rec(func, side, uplo, A11, B1, threshold; A_scale = A_scale)

        if side == 'L'
            func == 'S' ? GEMM_SUB_cublas!(B2, A21, B1, A_scale) : GEMM_ADD_cublas!(A12, B2, B1, A_scale)
        else 
            func == 'S' ? GEMM_SUB_cublas!(B2, B1, A12, A_scale) : GEMM_ADD_cublas!(B2, A21, B1, A_scale)
        end

        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

    else
        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

        if side == 'L'
            func == 'S' ? GEMM_SUB_cublas!(B1, A12, B2, A_scale) : GEMM_ADD_cublas!(B1, A21, B2, A_scale)
        else 
            func == 'S' ? GEMM_SUB_cublas!(B1, B2, A21, A_scale) : GEMM_ADD_cublas!(B1, A12, B2, A_scale)
        end

        unified_rec(func, side, uplo, A11, B1, threshold; A_scale = A_scale)
    end

    return B
end

# unified rec with no mixed prec
function unified_rec(func::Char, side::Char, uplo::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T}, threshold::Int=256;
    A_scale::Float32=1.0f0) where T <: AbstractFloat

    n = size(A, 1)
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

    # split point
    if isinteger(log2(n))
        mid = div(n, 2)
    else
        mid = 2 ^ floor(Int, log2(n))
    end

    mid_remainder = n - mid

    # block views
    A11 = view(A, 1:mid,       1:mid)
    A22 = view(A, mid+1:n,     mid+1:n)
    A21 = view(A, mid+1:n,     1:mid)
    A12 = view(A, 1:mid,       mid+1:n)

    if side == 'L'
        B1 = view(B, 1:mid,     :)
        B2 = view(B, mid+1:n,   :)
    else
        B1 = view(B, :,         1:mid)
        B2 = view(B, :,         mid+1:n)
    end

    # first half
    if (side == 'L' && uplo == 'L' && func == 'S') || 
        (side == 'R' && uplo == 'U' && func == 'S') || 
        (side == 'L' && uplo == 'U' && func == 'M') || 
        (side == 'R' && uplo == 'L' && func == 'M')

        unified_rec(func, side, uplo, A11, B1, threshold; A_scale = A_scale)

        # GEMM update in mixed precision if deep enough
        if side == 'L'
            if func == 'S'
                # GEMM_SUB!(B2, A21, B1)
                GEMM_SUB_cublas!(B2, A21, B1, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(A12, B2, B1)
                GEMM_ADD_cublas!(A12, B2, B1, A_scale)
            end
        else  # side == 'R'
            if func == 'S'
                # GEMM_SUB!(B2, B1, A12)
                GEMM_SUB_cublas!(B2, B1, A12, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(B2, A21, B1)
                GEMM_ADD_cublas!(B2, A21, B1, A_scale)
            end
        end

        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

    # second half
    else
        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

        if side == 'L'
            if func == 'S'
                # GEMM_SUB!(B1, A12, B2)
                GEMM_SUB_cublas!(B1, A12, B2, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(A21, B1, B2)
                GEMM_ADD_cublas!(A21, B1, B2, A_scale)
            end
        else  # side == 'R'
            if func == 'S'
                # GEMM_SUB!(B1, B2, A21)
                GEMM_SUB_cublas!(B1, B2, A21, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(B1, A12, B2)
                GEMM_ADD_cublas!(B1, A12, B2, A_scale)
            end
        end

        unified_rec(func, side, uplo, A11, B1, threshold; A_scale = A_scale)
    end

    return B
end



# This is the multiple dispatch for the recursive data structure for A
"""
Unified recursive function for triangular matrix solve (TRSM) and multiply (TRMM) operations.

This function supports both solving triangular systems of equations and performing triangular matrix multiplications.

Arguments:
- side::Char: Specifies the side of the operation:
    - 'L': Left multiplication (A * B or inv(A) * B).
    - 'R': Right multiplication (B * A or B * inv(A)).
- uplo::Char: Specifies the triangular part of the matrix to reference:
    - 'U': Use the upper triangle.
    - 'L': Use the lower triangle.
- transpose::Char: Specifies the transposition operation:
    - 'N': No transpose.
    - 'T': Transpose.
    - 'C': Conjugate transpose.
- alpha::Number: Scalar multiplier applied to the operation.
- func::Char: Specifies the function type:
    - 'S': Solve (TRSM, A * X = alpha * B).
    - 'M': Multiply (TRMM, Update B = alpha * A * B or alpha * B * A).
- A::TriMixedPrec: The triangular matrix, with mixed precision data structure.
- B::AbstractMatrix: The matrix to multiply or solve for.

Returns:
- Updated matrix `B` after performing the specified operation.

Notes:
- The function modifies `B` in place.
"""
function unified_rectrxm!(
        side::Char, 
        uplo::Char, 
        trans::Char, 
        alpha::Number, 
        func::Char, 
        A::TriMixedPrec, 
        B::StridedMatrix
    )
    threshold = 16

    # We do not support transpose 'T' or 'C' yet for this mixed prec matrix
    # @assert transpose == 'N' "Transpose on TriMixedPrec not yet supported."
    

    if trans == 'T' || trans == 'C'
        A = transpose(A) 
        uplo = (uplo == 'L') ? 'U' : 'L'
    end    
    
    if func == 'S'
        threshold = 256
        B .= alpha .* B
    end
    unified_rec_mixed(func, side, uplo, A, B, threshold)
    if func == 'M'
        B .= alpha .* B
    end
    return B
end

function unified_rec_mixed(
    func::Char, side::Char, uplo::Char,
    A::TriMixedPrec{T_Base},
    B::StridedMatrix,
    threshold::Int=256
) where {T_Base}
    if A.BaseCase !== nothing
        A_block = A.BaseCase
        A_scale = A.base_scale !== nothing ? A.base_scale : 1.0f0
        B_type = eltype(B) 

        if eltype(A_block) == Float16 
            B_quant, B_scale = quantize(B) 

            unified_rec(func, side, uplo, A_block, B_quant, threshold; A_scale=A_scale)

            B_dequant = dequantize(B_quant, B_scale, B_type)
            copy!(B, B_dequant)

            if func == 'S'
                B ./= A_scale
            else
                temp_B_f32 = Float32.(B) .* A_scale
                clamp!(temp_B_f32, floatmin(eltype(B)), floatmax(eltype(B)))
                copy!(B, temp_B_f32)
            end
        else
            if eltype(A.BaseCase) == B_type
                unified_rec(func, side, uplo, A.BaseCase, B, threshold)
            else
                B_converted = eltype(A.BaseCase).(B)
                unified_rec(func, side, uplo, A.BaseCase, B_converted, threshold)
                B .= B_converted
            end
        end
        return B
    else
        mid = size(A.A11, 1)
        n = size(A, 1)

        if side == 'L'
            B1 = view(B, 1:mid,     :)
            B2 = view(B, mid+1:n,   :)
        else
            B1 = view(B, :,         1:mid)
            B2 = view(B, :,         mid+1:n)
        end

        OffDiag_block = A.OffDiag

        if (side == 'L' && uplo == 'L' && func == 'S') || 
        (side == 'R' && uplo == 'U' && func == 'S') || 
        (side == 'L' && uplo == 'U' && func == 'M') || 
        (side == 'R' && uplo == 'L' && func == 'M')
        
            unified_rec_mixed(func, side, uplo, A.A11, B1, threshold)

            A_type = eltype(OffDiag_block)
            A_scale = A.offDiag_scale !== nothing ? A.offDiag_scale : 1.0f0
            B_type = eltype(B) 

            if A_type != B_type
                if side == 'L' && func == 'S'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_SUB_cublas!(B2, OffDiag_block, A_type.(B1), A_scale)
                        else
                            B2_lp = Float32.(B2) 
                            GEMM_SUB_cublas!(B2_lp, OffDiag_block, A_type.(B1), A_scale)
                            copy!(B2, B2_lp)
                        end
                    else
                        B2_lp = A_type.(B2) 
                        # GEMM_SUB!(B2_lp, OffDiag_block, A_type.(B1))
                        GEMM_SUB_cublas!(B2_lp, OffDiag_block, A_type.(B1), A_scale)
                        copy!(B2, B2_lp)
                    end
                elseif side == 'L' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD_cublas!(OffDiag_block, A_type.(B2), B1, A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_ADD_cublas!(OffDiag_block, A_type.(B2), B1_lp, A_scale)
                            copy!(B1, B1_lp)
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_ADD!(OffDiag_block, A_type.(B2), B1_lp)
                        GEMM_ADD_cublas!(OffDiag_block, A_type.(B2), B1_lp, A_scale)
                        copy!(B1, B1_lp)
                    end
                elseif side == 'R' && func == 'S'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_SUB_cublas!(B2, A_type.(B1), OffDiag_block, A_scale)
                        else
                            B2_lp = Float32.(B2) 
                            GEMM_SUB_cublas!(B2_lp, A_type.(B1), OffDiag_block, A_scale)
                            copy!(B2, B2_lp)
                        end
                    else
                        B2_lp = A_type.(B2) 
                        # GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block)
                        GEMM_SUB_cublas!(B2_lp, A_type.(B1), OffDiag_block, A_scale)
                        copy!(B2, B2_lp)
                    end
                else # side == 'R' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD_cublas!(A_type.(B2), OffDiag_block, B1, A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_ADD_cublas!(A_type.(B2), OffDiag_block, B1_lp, A_scale)
                            copy!(B1, B1_lp) 
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_ADD!(A_type.(B2), OffDiag_block, B1_lp) 
                        GEMM_ADD_cublas!(A_type.(B2), OffDiag_block, B1_lp, A_scale)
                        copy!(B1, B1_lp) 
                    end
                end
            else
                if side == 'L' && func == 'S'
                    # GEMM_SUB!(B2, OffDiag_block, B1)
                    GEMM_SUB_cublas!(B2, OffDiag_block, B1, A_scale)
                elseif side == 'L' && func == 'M'
                    # GEMM_ADD!(OffDiag_block, B2, B1)
                    GEMM_ADD_cublas!(OffDiag_block, B2, B1, A_scale)
                elseif side == 'R' && func == 'S'
                    # GEMM_SUB!(B2, B1, OffDiag_block)
                    GEMM_SUB_cublas!(B2, B1, OffDiag_block, A_scale)
                else # side == 'R' && func == 'M'
                    # GEMM_ADD!(B2, OffDiag_block, B1)
                    GEMM_ADD_cublas!(B2, OffDiag_block, B1, A_scale)
                end
            end

            unified_rec_mixed(func, side, uplo, A.A22, B2, threshold)
        else 
            unified_rec_mixed(func, side, uplo, A.A22, B2, threshold)

            A_type = eltype(OffDiag_block)
            A_scale = A.offDiag_scale !== nothing ? A.offDiag_scale : 1.0f0
            B_type = eltype(B)
            
            if A_type != B_type
                if side == 'L' && func == 'S'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_SUB_cublas!(B1, OffDiag_block, A_type.(B2), A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_SUB_cublas!(B1_lp, OffDiag_block, A_type.(B2), A_scale)
                            copy!(B1, B1_lp)
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_SUB!(B1_lp, OffDiag_block, A_type.(copy(B2)))
                        GEMM_SUB_cublas!(B1_lp, OffDiag_block, A_type.(B2), A_scale)
                        copy!(B1, B1_lp)
                    end
                elseif side == 'L' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD_cublas!(OffDiag_block, A_type.(B1), B2, A_scale)
                        else
                            B2_lp = Float32.(B2)
                            GEMM_ADD_cublas!(OffDiag_block, A_type.(B1), B2_lp, A_scale)
                            copy!(B2, B2_lp)
                        end
                    else
                        B2_lp = A_type.(B2)
                        # GEMM_ADD!(OffDiag_block, A_type.(B1), B2_lp)
                        GEMM_ADD_cublas!(OffDiag_block, A_type.(B1), B2_lp, A_scale)
                        copy!(B2, B2_lp)
                    end
                elseif side == 'R' && func == 'S'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_SUB_cublas!(B1, A_type.(B2), OffDiag_block, A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_SUB_cublas!(B1_lp, A_type.(B2), OffDiag_block, A_scale)
                            copy!(B1, B1_lp)
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block)
                        GEMM_SUB_cublas!(B1_lp, A_type.(B2), OffDiag_block, A_scale)
                        copy!(B1, B1_lp)
                    end
                else # side == 'R' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD_cublas!(A_type.(B1), OffDiag_block, B2, A_scale)
                        else
                            B2_lp = Float32.(B2)
                            GEMM_ADD_cublas!(A_type.(B1), OffDiag_block, B2_lp, A_scale)
                            copy!(B2, B2_lp) 
                        end
                    else
                        B2_lp = A_type.(B2)
                        # GEMM_ADD!(A_type.(B1), OffDiag_block, B2_lp) 
                        GEMM_ADD_cublas!(A_type.(B1), OffDiag_block, B2_lp, A_scale)
                        copy!(B2, B2_lp) 
                    end
                end
            else
                if side == 'L' && func == 'S'
                    # GEMM_SUB!(B1, OffDiag_block, B2)
                    GEMM_SUB_cublas!(B1, OffDiag_block, B2, A_scale)
                elseif side == 'L' && func == 'M'
                    # GEMM_ADD!(OffDiag_block, B1, B2)
                    GEMM_ADD_cublas!(OffDiag_block, B1, B2, A_scale)
                elseif side == 'R' && func == 'S'
                    # GEMM_SUB!(B1, B2, OffDiag_block)
                    GEMM_SUB_cublas!(B1, B2, OffDiag_block, A_scale)
                else # side == 'R' && func == 'M'
                    # GEMM_ADD!(B1, OffDiag_block, B2)
                    GEMM_ADD_cublas!(B1, OffDiag_block, B2, A_scale)
                end
            end
            
            unified_rec_mixed(func, side, uplo, A.A11, B1, threshold)
        end

        return B

    end
    
end