export unified_rectrxm!
using StochasticRounding

function stochastic_convert(::Type{T_out}, M_in::AbstractArray) where T_out
    M_out = similar(M_in, T_out)
    @. M_out = stochastic_round(T_out, M_in)
    return M_out
end


function quantize(matrix::AbstractMatrix{T}) where T <: AbstractFloat
    FP16_MAX_VAL = 65504.0f0
    alpha = maximum(abs, matrix) 
    
    if iszero(alpha)
        return similar(matrix, Float16), 1.0f0
    end

    if alpha > FP16_MAX_VAL
        s = Float32(alpha / FP16_MAX_VAL)
        
        quantized_matrix = similar(matrix, Float16, size(matrix))
        print("CLAMPING3")
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


function GEMM_ADD!(A, B, C::AnyGPUArray, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    if eltype(A_mat) == Float16 && eltype(B_mat) == Float16
        if eltype(C) == Float16
            C_op = Float32.(C)
            gemmEx!(transA, transB, scale, A_mat, B_mat, 1.0f0, C_op)
            #print("CLAMPING4")
            clamp!(C_op, floatmin(Float16), floatmax(Float16))
            copy!(C, C_op)
        else
            gemmEx!(transA, transB, scale, A_mat, B_mat, 1.0f0, C)
        end
    else
        T_C = eltype(C)
        gemm!(transA, transB, T_C(scale), A_mat, B_mat, T_C(1.0), C)
    end
end

function GEMM_SUB!(C::AnyGPUArray, A, B, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    if eltype(A_mat) == Float16 && eltype(B_mat) == Float16
        if eltype(C) == Float16
            C_op = Float32.(C)
            gemmEx!(transA, transB, -scale, A_mat, B_mat, 1.0f0, C_op)
            #print("CLAMPING5")
            clamp!(C_op, floatmin(Float16), floatmax(Float16))
            copy!(C, C_op)
        else
            gemmEx!(transA, transB, -scale, A_mat, B_mat, 1.0f0, C)
        end
    else
        T_C = eltype(C)
        gemm!(transA, transB, T_C(-scale), A_mat, B_mat, T_C(1.0), C)
    end
end


function GEMM_ADD!(A, B, C::oneAPI.oneDeviceArray, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    T_C = eltype(C)
    oneMKL.gemm!(transA, transB, T_C(scale), A_mat, B_mat, T_C(1.0), C)
end

function GEMM_SUB!(C::oneAPI.oneDeviceArray, A, B, scale::Float32=1.0f0)
    transA = A isa Transpose ? 'T' : 'N'
    transB = B isa Transpose ? 'T' : 'N'
    A_mat = A isa Transpose ? parent(A) : A
    B_mat = B isa Transpose ? parent(B) : B
    T_C = eltype(C)
    oneMKL.gemm!(transA, transB, T_C(-scale), A_mat, B_mat, T_C(1.0), C)
end

function dispatch_trsm!(side, uplo, trans, diag, alpha, A, B)
    if eltype(A) == Float16
        B_temp = Float32.(B)
        trsm!(side, uplo, trans, diag, alpha, Float32.(A), B_temp)
        copy!(B, B_temp)
    else
        trsm!(side, uplo, trans, diag, alpha, A, B)
    end
end

#_dispatch_trsm_kernel!(side, uplo, trans, diag, alpha, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray) = CUBLAS.trsm!(side, uplo, trans, diag, alpha, A, B)
#_dispatch_trsm_kernel!(side, uplo, trans, diag, alpha, A::AMDGPU.StridedROCArray, B::AMDGPU.StridedROCArray) = AMDGPU.rocBLAS.trsm!(side, uplo, trans, diag, alpha, A, B)
#_dispatch_trsm_kernel!(side, uplo, trans, diag, alpha, A::oneAPI.oneDeviceArray, B::oneAPI.oneDeviceArray) = oneMKL.trsm!(side, uplo, trans, diag, alpha, A, B)


function dispatch_trmm!(side, uplo, trans, diag, alpha, A, B)
    if eltype(A) == Float16
        B_temp = Float32.(B)
        trmm!(side, uplo, trans, diag, alpha, Float32.(A), B_temp, B_temp)
        copy!(B, B_temp)
    else
        trmm!(side, uplo, trans, diag, alpha, A, B, B)
    end
end

#_dispatch_trmm_kernel!(side, uplo, trans, diag, alpha, A::CUDA.StridedCuArray, B::CUDA.StridedCuArray, C::CUDA.StridedCuArray) = CUBLAS.trmm!(side, uplo, trans, diag, alpha, A, B, C)
#_dispatch_trmm_kernel!(side, uplo, trans, diag, alpha, A::AMDGPU.StridedROCArray, B::AMDGPU.StridedROCArray, C::AMDGPU.StridedROCArray) = AMDGPU.rocBLAS.trmm!(side, uplo, trans, diag, alpha, A, B, C)
#_dispatch_trmm_kernel!(side, uplo, trans, diag, alpha, A::oneAPI.oneDeviceArray, B::oneAPI.oneDeviceArray, C::oneAPI.oneDeviceArray) = oneMKL.trmm!(side, uplo, trans, diag, alpha, A, B, C)


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
        B_type = eltype(B)
        swapped_uplo = (uplo == 'U') ? 'L' : 'U'
        if func == 'S'
            dispatch_trsm!(side, swapped_uplo, 'T', 'N', one(B_type), A_orig, B)
        else 
            dispatch_trmm!(side, swapped_uplo, 'T', 'N', one(B_type), A_orig, B)
        end
        return B
        # if func == 'S'
        #     if (eltype(A) == Float16)
        #         B_temp = Float32.(B)
        #         swapped_uplo = (uplo == 'U') ? 'L' : 'U'
        #         CUBLAS.trsm!(side, swapped_uplo, 'T', 'N', one(T), Float32.(A_orig), B_temp)
        #         copy!(B, B_temp) 
        #         # if side == 'L' && uplo == 'L'
        #         #     LeftLowerTRSM!(A, B)
        #         # elseif side == 'L' && uplo == 'U'
        #         #     LeftUpperTRSM!(A, B)
        #         # elseif side == 'R' && uplo == 'L'
        #         #     RightLowerTRSM!(A, B)
        #         # else
        #         #     RightUpperTRSM!(A, B)   
        #         # end
        #     else
        #         swapped_uplo = (uplo == 'U') ? 'L' : 'U'
        #         CUBLAS.trsm!(side, swapped_uplo, 'T', 'N', one(T), A_orig, B)
        #     end
        # else 
        #     if (eltype(A) == Float16)
        #         if side == 'L' && uplo == 'L'
        #             LeftLowerTRMM!(A, B)
        #         elseif side == 'L' && uplo == 'U'
        #             LeftUpperTRMM!(A, B)
        #         elseif side == 'R' && uplo == 'L'
        #             RightLowerTRMM!(A, B)
        #         else
        #             RightUpperTRMM!(A, B)
        #         end
        #     else
        #         swapped_uplo = (uplo == 'U') ? 'L' : 'U'
        #         CUBLAS.trmm!(side, swapped_uplo, 'T', 'N', one(T), A_orig, B)
        #     end
        # end
        # return B

    end

    A_orig = parent(A)

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
            func == 'S' ? GEMM_SUB!(B2, A21, B1, A_scale) : GEMM_ADD!(A12, B2, B1, A_scale)
        else 
            func == 'S' ? GEMM_SUB!(B2, B1, A12, A_scale) : GEMM_ADD!(B2, A21, B1, A_scale)
        end

        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

    else
        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

        if side == 'L'
            func == 'S' ? GEMM_SUB!(B1, A12, B2, A_scale) : GEMM_ADD!(B1, A21, B2, A_scale)
        else 
            func == 'S' ? GEMM_SUB!(B1, B2, A21, A_scale) : GEMM_ADD!(B1, A12, B2, A_scale)
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
            dispatch_trsm!(side, uplo, 'N', 'N', one(T), A, B)
            # if (eltype(A) == Float16)
            #     B_temp = Float32.(B)
            #     CUBLAS.trsm!(side, uplo, 'N', 'N', one(T), Float32.(A), B_temp)
            #     copy!(B, B_temp) 
            #     # if side == 'L' && uplo == 'L'
            #     #     LeftLowerTRSM!(A, B)
            #     # elseif side == 'L' && uplo == 'U'
            #     #     LeftUpperTRSM!(A, B)
            #     # elseif side == 'R' && uplo == 'L'
            #     #     RightLowerTRSM!(A, B)
            #     # else
            #     #     RightUpperTRSM!(A, B)   
            #     # end
            # else
            #     CUBLAS.trsm!(side, uplo, 'N', 'N', one(T), A, B)
            # end
        else 
            dispatch_trmm!(side, uplo, 'N', 'N', one(T), A, B)
            # if (eltype(A) == Float16)
            #     if side == 'L' && uplo == 'L'
            #         LeftLowerTRMM!(A, B)
            #     elseif side == 'L' && uplo == 'U'
            #         LeftUpperTRMM!(A, B)
            #     elseif side == 'R' && uplo == 'L'
            #         RightLowerTRMM!(A, B)
            #     else
            #         RightUpperTRMM!(A, B)
            #     end
            # else
            #     CUBLAS.trmm!(side, uplo, 'N', 'N', one(T), A, B, B)
            # end
        end
        # if func == 'S'
        #     if side == 'L' && uplo == 'L'
        #         LeftLowerTRSM!(A, B)
        #     elseif side == 'L' && uplo == 'U'
        #         LeftUpperTRSM!(A, B)
        #     elseif side == 'R' && uplo == 'L'
        #         RightLowerTRSM!(A, B)
        #     else
        #         RightUpperTRSM!(A, B)   
        #     end
        # else
        #     if side == 'L' && uplo == 'L'
        #         LeftLowerTRMM!(A, B)
        #     elseif side == 'L' && uplo == 'U'
        #         LeftUpperTRMM!(A, B)
        #     elseif side == 'R' && uplo == 'L'
        #         RightLowerTRMM!(A, B)
        #     else
        #         RightUpperTRMM!(A, B)
        #     end
        # end 
        
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
                GEMM_SUB!(B2, A21, B1, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(A12, B2, B1)
                GEMM_ADD!(A12, B2, B1, A_scale)
            end
        else  # side == 'R'
            if func == 'S'
                # GEMM_SUB!(B2, B1, A12)
                GEMM_SUB!(B2, B1, A12, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(B2, A21, B1)
                GEMM_ADD!(B2, A21, B1, A_scale)
            end
        end

        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

    # second half
    else
        unified_rec(func, side, uplo, A22, B2, threshold; A_scale = A_scale)

        if side == 'L'
            if func == 'S'
                # GEMM_SUB!(B1, A12, B2)
                GEMM_SUB!(B1, A12, B2, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(A21, B1, B2)
                GEMM_ADD!(A21, B1, B2, A_scale)
            end
        else  # side == 'R'
            if func == 'S'
                # GEMM_SUB!(B1, B2, A21)
                GEMM_SUB!(B1, B2, A21, A_scale)
            else  # func == 'M'
                # GEMM_ADD!(B1, A12, B2)
                GEMM_ADD!(B1, A12, B2, A_scale)
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
        A::AbstractMixedPrec, 
        B::StridedMatrix
    )
    threshold = 16
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
    A::AbstractMixedPrec{T_Base},
    B::StridedMatrix,
    threshold::Int=256
) where {T_Base}
    if A.BaseCase !== nothing
        A_block = A.BaseCase
        A_scale = A.base_scale !== nothing ? A.base_scale : 1.0f0
        B_type = eltype(B) 

        if eltype(A_block) == Float16 
            if B_type == eltype(A_block)
                unified_rec(func, side, uplo, A_block, B, threshold; A_scale=A_scale)
            else 
                B_quant, B_scale = quantize(B)

                unified_rec(func, side, uplo, A_block, B_quant, threshold; A_scale=A_scale)

                B_dequant = dequantize(B_quant, B_scale, B_type)
                copy!(B, B_dequant)
            end

            if func == 'S'
                B ./= A_scale
            else
                temp_B_f32 = Float32.(B) .* A_scale
                print("CLAMPING8")
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
                            GEMM_SUB!(B2, OffDiag_block, A_type.(B1), A_scale)
                        else
                            B2_lp = Float32.(B2) 
                            GEMM_SUB!(B2_lp, OffDiag_block, A_type.(B1), A_scale)
                            copy!(B2, B2_lp)
                        end
                    else
                        B2_lp = A_type.(B2) 
                        # GEMM_SUB!(B2_lp, OffDiag_block, A_type.(B1))
                        GEMM_SUB!(B2_lp, OffDiag_block, A_type.(B1), A_scale)
                        copy!(B2, B2_lp)
                    end
                elseif side == 'L' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD!(OffDiag_block, A_type.(B2), B1, A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_ADD!(OffDiag_block, A_type.(B2), B1_lp, A_scale)
                            copy!(B1, B1_lp)
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_ADD!(OffDiag_block, A_type.(B2), B1_lp)
                        GEMM_ADD!(OffDiag_block, A_type.(B2), B1_lp, A_scale)
                        copy!(B1, B1_lp)
                    end
                elseif side == 'R' && func == 'S'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_SUB!(B2, A_type.(B1), OffDiag_block, A_scale)
                        else
                            B2_lp = Float32.(B2) 
                            GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block, A_scale)
                            copy!(B2, B2_lp)
                        end
                    else
                        B2_lp = A_type.(B2) 
                        # GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block)
                        GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block, A_scale)
                        copy!(B2, B2_lp)
                    end
                else # side == 'R' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD!(A_type.(B2), OffDiag_block, B1, A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_ADD!(A_type.(B2), OffDiag_block, B1_lp, A_scale)
                            copy!(B1, B1_lp) 
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_ADD!(A_type.(B2), OffDiag_block, B1_lp) 
                        GEMM_ADD!(A_type.(B2), OffDiag_block, B1_lp, A_scale)
                        copy!(B1, B1_lp) 
                    end
                end
            else
                if side == 'L' && func == 'S'
                    # GEMM_SUB!(B2, OffDiag_block, B1)
                    GEMM_SUB!(B2, OffDiag_block, B1, A_scale)
                elseif side == 'L' && func == 'M'
                    # GEMM_ADD!(OffDiag_block, B2, B1)
                    GEMM_ADD!(OffDiag_block, B2, B1, A_scale)
                elseif side == 'R' && func == 'S'
                    # GEMM_SUB!(B2, B1, OffDiag_block)
                    GEMM_SUB!(B2, B1, OffDiag_block, A_scale)
                else # side == 'R' && func == 'M'
                    # GEMM_ADD!(B2, OffDiag_block, B1)
                    GEMM_ADD!(B2, OffDiag_block, B1, A_scale)
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
                            GEMM_SUB!(B1, OffDiag_block, A_type.(B2), A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_SUB!(B1_lp, OffDiag_block, A_type.(B2), A_scale)
                            copy!(B1, B1_lp)
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_SUB!(B1_lp, OffDiag_block, A_type.(copy(B2)))
                        GEMM_SUB!(B1_lp, OffDiag_block, A_type.(B2), A_scale)
                        copy!(B1, B1_lp)
                    end
                elseif side == 'L' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD!(OffDiag_block, A_type.(B1), B2, A_scale)
                        else
                            B2_lp = Float32.(B2)
                            GEMM_ADD!(OffDiag_block, A_type.(B1), B2_lp, A_scale)
                            copy!(B2, B2_lp)
                        end
                    else
                        B2_lp = A_type.(B2)
                        # GEMM_ADD!(OffDiag_block, A_type.(B1), B2_lp)
                        GEMM_ADD!(OffDiag_block, A_type.(B1), B2_lp, A_scale)
                        copy!(B2, B2_lp)
                    end
                elseif side == 'R' && func == 'S'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_SUB!(B1, A_type.(B2), OffDiag_block, A_scale)
                        else
                            B1_lp = Float32.(B1)
                            GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block, A_scale)
                            copy!(B1, B1_lp)
                        end
                    else
                        B1_lp = A_type.(B1)
                        # GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block)
                        GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block, A_scale)
                        copy!(B1, B1_lp)
                    end
                else # side == 'R' && func == 'M'
                    if A_type == Float16
                        if B_type !== Float64
                            GEMM_ADD!(A_type.(B1), OffDiag_block, B2, A_scale)
                        else
                            B2_lp = Float32.(B2)
                            GEMM_ADD!(A_type.(B1), OffDiag_block, B2_lp, A_scale)
                            copy!(B2, B2_lp) 
                        end
                    else
                        B2_lp = A_type.(B2)
                        # GEMM_ADD!(A_type.(B1), OffDiag_block, B2_lp) 
                        GEMM_ADD!(A_type.(B1), OffDiag_block, B2_lp, A_scale)
                        copy!(B2, B2_lp) 
                    end
                end
            else
                if side == 'L' && func == 'S'
                    # GEMM_SUB!(B1, OffDiag_block, B2)
                    GEMM_SUB!(B1, OffDiag_block, B2, A_scale)
                elseif side == 'L' && func == 'M'
                    # GEMM_ADD!(OffDiag_block, B1, B2)
                    GEMM_ADD!(OffDiag_block, B1, B2, A_scale)
                elseif side == 'R' && func == 'S'
                    # GEMM_SUB!(B1, B2, OffDiag_block)
                    GEMM_SUB!(B1, B2, OffDiag_block, A_scale)
                else # side == 'R' && func == 'M'
                    # GEMM_ADD!(B1, OffDiag_block, B2)
                    GEMM_ADD!(B1, OffDiag_block, B2, A_scale)
                end
            end
            
            unified_rec_mixed(func, side, uplo, A.A11, B1, threshold)
        end

        return B

    end
    
end


# mixed precision rectrxm for transpose
function unified_rec_mixed(
    func::Char, side::Char, uplo::Char,
    A::TransposedMixedPrec,
    B::StridedMatrix,
    threshold::Int=256
)
    A_orig = parent(A)
    
    if A_orig.BaseCase !== nothing
        A_block = A_orig.BaseCase
        scale = A_orig.base_scale !== nothing ? A_orig.base_scale : 1.0f0
        if eltype(A_block) == Float16
            B_converted = Float32.(B)
            unified_rec(func, side, uplo, transpose(Float32.(A)), B_converted, threshold; A_scale=scale)
            copy!(B, B_converted)
        else
            A_block_transposed = transpose(A_block) 
            if eltype(A_block) != eltype(B)
                B_converted = eltype(A_block).(B)
                unified_rec(func, side, uplo, A_block_transposed, B_converted, threshold; A_scale=scale)
                copy!(B, B_converted)
            else
                unified_rec(func, side, uplo, A_block_transposed, B, threshold; A_scale=scale)
            end
        end
        return B
    end

    mid = size(A_orig.A11, 1)
    n = size(A_orig, 1)

    if side == 'L'
        B1 = view(B, 1:mid,   :)
        B2 = view(B, mid+1:n, :)
    else
        B1 = view(B, :, 1:mid)
        B2 = view(B, :, mid+1:n)
    end

    A11_trans = transpose(A_orig.A11)
    A22_trans = transpose(A_orig.A22)
    OffDiag_block_trans = transpose(A_orig.OffDiag) 

    if (side == 'L' && uplo == 'L' && func == 'S') || 
       (side == 'R' && uplo == 'U' && func == 'S') || 
       (side == 'L' && uplo == 'U' && func == 'M') || 
       (side == 'R' && uplo == 'L' && func == 'M')
        
        unified_rec_mixed(func, side, uplo, A11_trans, B1, threshold)
        
        
        A_type = eltype(A_orig.OffDiag)
        A_scale = A_orig.offDiag_scale !== nothing ? A_orig.offDiag_scale : 1.0f0
        B_type = eltype(B) 

        if A_type != B_type
            if side == 'L' && func == 'S'
                if A_type == Float16
                    if B_type !== Float64
                        GEMM_SUB!(B2, OffDiag_block_trans, A_type.(B1), A_scale)
                    else
                        B2_lp = Float32.(B2) 
                        GEMM_SUB!(B2_lp, OffDiag_block_trans, A_type.(B1), A_scale)
                        copy!(B2, B2_lp)
                    end
                else
                    B2_lp = A_type.(B2) 
                    GEMM_SUB!(B2_lp, OffDiag_block_trans, A_type.(B1), A_scale)
                    copy!(B2, B2_lp)
                end
            elseif side == 'L' && func == 'M'
                 if A_type == Float16
                     if B_type !== Float64
                         GEMM_ADD!(B1, OffDiag_block_trans, A_type.(B2), A_scale)
                     else
                         B1_lp = Float32.(B1)
                         GEMM_ADD!(B1_lp, OffDiag_block_trans, A_type.(B2), A_scale)
                         copy!(B1, B1_lp)
                     end
                 else
                     B1_lp = A_type.(B1)
                     GEMM_ADD!(B1_lp, OffDiag_block_trans, A_type.(B2), A_scale)
                     copy!(B1, B1_lp)
                 end
            elseif side == 'R' && func == 'S'
                 if A_type == Float16
                     if B_type !== Float64
                         GEMM_SUB!(B2, A_type.(B1), OffDiag_block_trans, A_scale)
                     else
                         B2_lp = Float32.(B2) 
                         GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block_trans, A_scale)
                         copy!(B2, B2_lp)
                     end
                 else
                     B2_lp = A_type.(B2) 
                     GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block_trans, A_scale)
                     copy!(B2, B2_lp)
                 end
            else 
                 if A_type == Float16
                     if B_type !== Float64
                         GEMM_ADD!(B1, A_type.(B2), OffDiag_block_trans, A_scale)
                     else
                         B1_lp = Float32.(B1)
                         GEMM_ADD!(B1_lp, A_type.(B2), OffDiag_block_trans, A_scale)
                         copy!(B1, B1_lp) 
                     end
                 else
                     B1_lp = A_type.(B1)
                     GEMM_ADD!(B1_lp, A_type.(B2), OffDiag_block_trans, A_scale)
                     copy!(B1, B1_lp) 
                 end
            end
        else 
            if side == 'L' && func == 'S'
                GEMM_SUB!(B2, OffDiag_block_trans, B1, A_scale)
            elseif side == 'L' && func == 'M'
                GEMM_ADD!(B1, OffDiag_block_trans, B2, A_scale)
            elseif side == 'R' && func == 'S'
                GEMM_SUB!(B2, B1, OffDiag_block_trans, A_scale)
            else
                GEMM_ADD!(B1, B2, OffDiag_block_trans, A_scale)
            end
        end
        
        unified_rec_mixed(func, side, uplo, A22_trans, B2, threshold)
    else 
        unified_rec_mixed(func, side, uplo, A22_trans, B2, threshold)

        A_type = eltype(A_orig.OffDiag)
        A_scale = A_orig.offDiag_scale !== nothing ? A_orig.offDiag_scale : 1.0f0
        B_type = eltype(B)
        
        if A_type != B_type
            if side == 'L' && func == 'S'
                if A_type == Float16
                    if B_type !== Float64
                        GEMM_SUB!(B1, OffDiag_block_trans, A_type.(B2), A_scale)
                    else
                        B1_lp = Float32.(B1)
                        GEMM_SUB!(B1_lp, OffDiag_block_trans, A_type.(B2), A_scale)
                        copy!(B1, B1_lp)
                    end
                else
                    B1_lp = A_type.(B1)
                    GEMM_SUB!(B1_lp, OffDiag_block_trans, A_type.(B2), A_scale)
                    copy!(B1, B1_lp)
                end
            elseif side == 'L' && func == 'M'
                if A_type == Float16
                    if B_type !== Float64
                        GEMM_ADD!(B2, OffDiag_block_trans, A_type.(B1), A_scale)
                    else
                        B2_lp = Float32.(B2)
                        GEMM_ADD!(B2_lp, OffDiag_block_trans, A_type.(B1), A_scale)
                        copy!(B2, B2_lp)
                    end
                else
                    B2_lp = A_type.(B2)
                    GEMM_ADD!(B2_lp, OffDiag_block_trans, A_type.(B1), A_scale)
                    copy!(B2, B2_lp)
                end
            elseif side == 'R' && func == 'S'
                if A_type == Float16
                    if B_type !== Float64
                        GEMM_SUB!(B1, A_type.(B2), OffDiag_block_trans, A_scale)
                    else
                        B1_lp = Float32.(B1)
                        GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block_trans, A_scale)
                        copy!(B1, B1_lp)
                    end
                else
                    B1_lp = A_type.(B1)
                    GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block_trans, A_scale)
                    copy!(B1, B1_lp)
                end
            else # side == 'R' && func == 'M'
                if A_type == Float16
                    if B_type !== Float64
                        GEMM_ADD!(B2, A_type.(B1), OffDiag_block_trans, A_scale)
                    else
                        B2_lp = Float32.(B2)
                        GEMM_ADD!(B2_lp, A_type.(B1), OffDiag_block_trans, A_scale)
                        copy!(B2, B2_lp) 
                    end
                else
                    B2_lp = A_type.(B2)
                    GEMM_ADD!(B2_lp, A_type.(B1), OffDiag_block_trans, A_scale)
                    copy!(B2, B2_lp) 
                end
            end
        else 
            if side == 'L' && func == 'S'
                GEMM_SUB!(B1, OffDiag_block_trans, B2, A_scale)
            elseif side == 'L' && func == 'M'
                GEMM_ADD!(B2, OffDiag_block_trans, B1, A_scale)
            elseif side == 'R' && func == 'S'
                GEMM_SUB!(B1, B2, OffDiag_block_trans, A_scale)
            else
                GEMM_ADD!(B2, B1, OffDiag_block_trans, A_scale)
            end
        end
        
        unified_rec_mixed(func, side, uplo, A11_trans, B1, threshold)
    end

    return B
end