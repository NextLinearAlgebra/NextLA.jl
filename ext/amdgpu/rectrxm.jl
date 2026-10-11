# rocBLAS fast path for unified_rectrxm!, the AMD counterpart of
# ext/cuda/rectrxm.jl.
#
# Same design as the CUDA file: an added method, strictly more specific than
# the AbstractMatrix method in src/trsm/rectrxm.jl, so the recursive
# KernelAbstractions path stays for everything this does not cover. The
# argument mapping is the CUDA one -- rocBLAS takes transa itself, so the
# original uplo is passed unflipped, and alpha goes to rocBLAS directly with no
# scaling of B.
#
# Two differences from CUDA, both deliberate:
#
#   - Typed on ROCMatrix, not StridedROCMatrix. AMDGPU 2.1 (the version the test
#     manifest pins) defines rocBLAS.trsm!/trmm! on ROCMatrix only; a view would
#     reach them and throw MethodError. Views -- the panels recursive LU and
#     Cholesky pass -- keep the portable path. AMDGPU 2.8 accepts strided views,
#     so this can widen once the pin moves.
#   - trmm! writes into a separate C here rather than aliasing C = B. cuBLAS
#     documents the in-place form; for rocBLAS this path has not been run, so it
#     does not rely on it and copies back instead.
#
# Typed on Float32, Float64, ComplexF32 and ComplexF64, the four element types
# rocBLAS trsm!/trmm! take in AMDGPU 2.1 and the ones the generic path accepts.
#
# Never executed: there is no AMD device on the development machine. It is
# checked to compile and to be selected for ROCMatrix arguments; correctness and
# speed-up on AMD hardware are unmeasured. See KNOWN_ISSUES.md.

function NextLA.unified_rectrxm!(side::Char,
                                 uplo::Char,
                                 transpose::Char,
                                 diag::Char,
                                 alpha::Number,
                                 func::Char,
                                 A::AMDGPU.ROCMatrix{T},
                                 B::AMDGPU.ROCMatrix{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    if diag != 'N' && diag != 'U'
        throw(ArgumentError("diag must be 'N' or 'U', got '$diag'"))
    end
    if func == 'S'
        rocBLAS.trsm!(side, uplo, transpose, diag, T(alpha), A, B)
    elseif func == 'M'
        C = similar(B)
        rocBLAS.trmm!(side, uplo, transpose, diag, T(alpha), A, B, C)
        copyto!(B, C)
    else
        throw(ArgumentError("func must be 'S' or 'M', got '$func'"))
    end
    return B
end
