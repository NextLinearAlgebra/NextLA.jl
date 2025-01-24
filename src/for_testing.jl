using CUDA
include("unified_rec.jl")

function left_upper_no_transpose(A, B)
    # A and B are assumed to be CuArrays
    unified_rectrxm!('L', 'U', 'N', 1.0, 'S', A, B)
end

function left_upper_transpose(A, B)
    # A and B are assumed to be CuArrays
    A_transposed = transpose(A)
    B_transposed = transpose(B)
    unified_rectrxm!('R', 'L', 'N', 1.0, 'S', A_transposed, B_transposed)
    B .= transpose(B_transposed)  # Modify B in-place on GPU
end
