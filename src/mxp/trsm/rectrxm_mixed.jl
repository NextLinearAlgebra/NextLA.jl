"""
    unified_rec_mixed(func::Char, side::Char, uplo::Char, diag::Char, A, B, threshold::Int=256)

Recursive core function for mixed-precision triangular solve/multiply operations.
Recursively divides matrices into blocks, handling necessary scaling and element type
conversions between the precision hierarchies of `A` and `B`.
"""
function unified_rec_mixed(
    func::Char, side::Char, uplo::Char, diag::Char,
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
                unified_rec(func, side, uplo, diag, A_block, B, threshold; A_scale=A_scale)
            else 
                B_quant, B_scale = quantize(B)

                unified_rec(func, side, uplo, diag, A_block, B_quant, threshold; A_scale=A_scale)

                B_dequant = dequantize(B_quant, B_scale, B_type)
                copy!(B, B_dequant)
            end

            if func == 'S'
                B ./= A_scale
            else
                temp_B_f32 = Float32.(B) .* A_scale
                clamp!(temp_B_f32, floatmin(eltype(B)), floatmax(eltype(B)))
                copy!(B, temp_B_f32)
            end
        else
            if eltype(A.BaseCase) == B_type
                unified_rec(func, side, uplo, diag, A.BaseCase, B, threshold)
            else
                B_converted = eltype(A.BaseCase).(B)
                unified_rec(func, side, uplo, diag, A.BaseCase, B_converted, threshold)
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

        if hasproperty(A, :A21)
            OffDiag_block = (uplo == 'L') ? A.A21 : A.A12
        else
            OffDiag_block = A.OffDiag
        end

        if (side == 'L' && uplo == 'L' && func == 'S') || 
        (side == 'R' && uplo == 'U' && func == 'S') || 
        (side == 'L' && uplo == 'U' && func == 'M') || 
        (side == 'R' && uplo == 'L' && func == 'M')
        
            unified_rec_mixed(func, side, uplo, diag, A.A11, B1, threshold)

            A_type = eltype(OffDiag_block)
            if hasproperty(A, :A21_scale)
                A_scale = (uplo == 'L') ? (A.A21_scale !== nothing ? A.A21_scale : 1.0f0) : (A.A12_scale !== nothing ? A.A12_scale : 1.0f0)
            else
                A_scale = A.offDiag_scale !== nothing ? A.offDiag_scale : 1.0f0
            end
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
                        GEMM_SUB!(B2_lp, A_type.(B1), OffDiag_block, A_scale)
                        copy!(B2, B2_lp)
                    end
                else 
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
                        GEMM_ADD!(A_type.(B2), OffDiag_block, B1_lp, A_scale)
                        copy!(B1, B1_lp) 
                    end
                end
            else
                if side == 'L' && func == 'S'
                    GEMM_SUB!(B2, OffDiag_block, B1, A_scale)
                elseif side == 'L' && func == 'M'
                    GEMM_ADD!(OffDiag_block, B2, B1, A_scale)
                elseif side == 'R' && func == 'S'
                    GEMM_SUB!(B2, B1, OffDiag_block, A_scale)
                else 
                    GEMM_ADD!(B2, OffDiag_block, B1, A_scale)
                end
            end

            unified_rec_mixed(func, side, uplo, diag, A.A22, B2, threshold)
        else 
            unified_rec_mixed(func, side, uplo, diag, A.A22, B2, threshold)

            A_type = eltype(OffDiag_block)
            if hasproperty(A, :A21_scale)
                A_scale = (uplo == 'L') ? (A.A21_scale !== nothing ? A.A21_scale : 1.0f0) : (A.A12_scale !== nothing ? A.A12_scale : 1.0f0)
            else
                A_scale = A.offDiag_scale !== nothing ? A.offDiag_scale : 1.0f0
            end
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
                        GEMM_SUB!(B1_lp, A_type.(B2), OffDiag_block, A_scale)
                        copy!(B1, B1_lp)
                    end
                else 
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
                        GEMM_ADD!(A_type.(B1), OffDiag_block, B2_lp, A_scale)
                        copy!(B2, B2_lp) 
                    end
                end
            else
                if side == 'L' && func == 'S'
                    GEMM_SUB!(B1, OffDiag_block, B2, A_scale)
                elseif side == 'L' && func == 'M'
                    GEMM_ADD!(OffDiag_block, B1, B2, A_scale)
                elseif side == 'R' && func == 'S'
                    GEMM_SUB!(B1, B2, OffDiag_block, A_scale)
                else 
                    GEMM_ADD!(B1, OffDiag_block, B2, A_scale)
                end
            end
            
            unified_rec_mixed(func, side, uplo, diag, A.A11, B1, threshold)
        end

        return B
    end
    
end

function unified_rec_mixed(
    func::Char, side::Char, uplo::Char, diag::Char,
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
            unified_rec(func, side, uplo, diag, transpose(Float32.(A)), B_converted, threshold; A_scale=scale)
            copy!(B, B_converted)
        else
            A_block_transposed = transpose(A_block) 
            if eltype(A_block) != eltype(B)
                B_converted = eltype(A_block).(B)
                unified_rec(func, side, uplo, diag, A_block_transposed, B_converted, threshold; A_scale=scale)
                copy!(B, B_converted)
            else
                unified_rec(func, side, uplo, diag, A_block_transposed, B, threshold; A_scale=scale)
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
       
        unified_rec_mixed(func, side, uplo, diag, A11_trans, B1, threshold)
        
        
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
        
        unified_rec_mixed(func, side, uplo, diag, A22_trans, B2, threshold)
    else 
        unified_rec_mixed(func, side, uplo, diag, A22_trans, B2, threshold)

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
            else 
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
        
        unified_rec_mixed(func, side, uplo, diag, A11_trans, B1, threshold)
    end

    return B
end

function unified_rec_mixed(
    func::Char, side::Char, uplo::Char,
    A::TransposedMixedPrec,
    B::StridedMatrix,
    threshold::Int=256
)
    return unified_rec_mixed(func, side, uplo, 'N', A, B, threshold)
end
