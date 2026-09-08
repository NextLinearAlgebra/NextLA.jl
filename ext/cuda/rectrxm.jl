# CUBLAS fast path for unified_rectrxm!.
#
# src/trsm/rectrxm.jl solves and multiplies by recursing over NextLA's own
# KernelAbstractions TRSM/TRMM kernels. That is what makes the routine portable,
# and it stays the path for CPU, oneAPI and Metal. On CUDA the vendor library is
# much faster — measured on an RTX 2000 Ada, Float32, lower/left solve:
# 7.47 ms vs 2.92 ms at n=512, and 32.45 ms vs 5.10 ms at n=2048.
#
# The approach is from vicki-development, which routes the base case to
# CUBLAS/rocBLAS. That branch does it by replacing the generic method, which
# also removes the routine from every backend without a vendor BLAS. Here it is
# an added method instead: typed on StridedCuMatrix it is strictly more specific
# than the AbstractMatrix method in src/trsm/rectrxm.jl, so nothing is overwritten
# and the recursive path is untouched.
#
# The argument mapping is not one-to-one with the generic version:
#
#   - src/trsm/rectrxm.jl handles trans='T'/'C' by wrapping A in Transpose/Adjoint
#     and flipping uplo. CUBLAS takes transa itself, so the ORIGINAL uplo is
#     passed through unflipped.
#   - For a solve it scales B by alpha before recursing, and for a multiply it
#     scales after. CUBLAS takes alpha directly, so neither scaling is applied
#     here.
#
# Typed on the four element types cuBLAS trsm/trmm take -- Float32, Float64,
# ComplexF32 and ComplexF64 -- which are also the types the generic path in
# src/trsm/rectrxm.jl accepts, so what is supported does not depend on the
# backend. Float16 keeps the portable path, which widens it to Float32.

function NextLA.unified_rectrxm!(side::Char,
                                 uplo::Char,
                                 transpose::Char,
                                 diag::Char,
                                 alpha::Number,
                                 func::Char,
                                 A::CUDA.StridedCuMatrix{T},
                                 B::CUDA.StridedCuMatrix{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    if diag != 'N' && diag != 'U'
        throw(ArgumentError("diag must be 'N' or 'U', got '$diag'"))
    end
    if func == 'S'
        CUBLAS.trsm!(side, uplo, transpose, diag, T(alpha), A, B)
    elseif func == 'M'
        # CUBLAS trmm! writes into a separate output; passing B as both operand
        # and destination gives the in-place form the generic method provides.
        CUBLAS.trmm!(side, uplo, transpose, diag, T(alpha), A, B, B)
    else
        throw(ArgumentError("func must be 'S' or 'M', got '$func'"))
    end
    return B
end
