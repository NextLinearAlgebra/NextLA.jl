function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::oneAPI.oneArray{<:Any, 2},
                        B::oneAPI.oneArray{<:Any, 2},
                        beta,
                        C::oneAPI.oneArray{<:Any, 2})
    throw(ArgumentError("NextLA.gemmEx! is not supported on oneAPI"))
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:oneAPI.oneArray{<:Any,2}},
                              B::AbstractVector{<:oneAPI.oneArray{<:Any,2}},
                              beta,
                              C::AbstractVector{<:oneAPI.oneArray{<:Any,2}})
    length(A) == length(B) == length(C) || throw(DimensionMismatch("gemm_batched!: matrix batches must have matching lengths"))
    oneMKL.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    return C
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::oneAPI.oneStridedArray{<:Any,3},
                              B::oneAPI.oneStridedArray{<:Any,3},
                              beta,
                              C::oneAPI.oneStridedArray{<:Any,3})
    oneMKL.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
    return C
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:oneAPI.oneArray{<:Any, 2}},
                                B::AbstractVector{<:oneAPI.oneArray{<:Any, 2}},
                                beta,
                                C::AbstractVector{<:oneAPI.oneArray{<:Any, 2}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    if eltype(A) == eltype(B) == eltype(C) &&
       compute_type == NextLA.default_compute_type(alpha, A, B, beta, C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    throw(ArgumentError("NextLA.gemmEx_batched! mixed-type batched GEMM is not supported on oneAPI"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::oneAPI.oneStridedArray{<:Any, 3},
                                B::oneAPI.oneStridedArray{<:Any, 3},
                                beta,
                                C::oneAPI.oneStridedArray{<:Any, 3};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    if eltype(A) == eltype(B) == eltype(C) &&
       compute_type == NextLA.default_compute_type(alpha, A, B, beta, C)
        return NextLA.gemm_batched!(transA, transB, alpha, A, B, beta, C)
    end
    throw(ArgumentError("NextLA.gemmEx_batched! mixed-type batched GEMM is not supported on oneAPI"))
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              Aptrs::oneAPI.oneArray,
                              Aref::AbstractMatrix,
                              Bptrs::oneAPI.oneArray,
                              Bref::AbstractMatrix,
                              beta,
                              Cptrs::oneAPI.oneArray,
                              Cref::AbstractMatrix,
                              batch_count::Integer)
    throw(ArgumentError("NextLA.gemm_batched! pointer-batched GEMM is not supported on oneAPI"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                Aptrs::oneAPI.oneArray,
                                Aref::AbstractMatrix,
                                Bptrs::oneAPI.oneArray,
                                Bref::AbstractMatrix,
                                beta,
                                Cptrs::oneAPI.oneArray,
                                Cref::AbstractMatrix,
                                batch_count::Integer;
                                compute_type::Type = NextLA.default_compute_type(alpha, Aref, Bref, beta, Cref))
    throw(ArgumentError("NextLA.gemmEx_batched! pointer-batched mixed-type GEMM is not supported on oneAPI"))
end
