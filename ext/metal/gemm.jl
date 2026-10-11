function _gemm_batched_mps!(transA::Char,
                            transB::Char,
                            alpha,
                            A::Metal.MtlArray{<:Any, 3},
                            B::Metal.MtlArray{<:Any, 3},
                            beta,
                            C::Metal.MtlArray{<:Any, 3})
    batchA, batchB, batchC = NextLA._check_batch_dims(A, B, C)

    if batchA == batchB == batchC &&
       _supports_mps_batched_matmul(eltype(A), eltype(B), eltype(C))
        return MPS.matmul!(
            C,
            A,
            B,
            alpha,
            beta,
            transA != 'N',
            transB != 'N',
        )
    end
    throw(ArgumentError("Metal batched GEMM requires equal batch sizes and an MPS-supported eltype combination"))
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::Metal.MtlArray{<:Any, 2},
                        B::Metal.MtlArray{<:Any, 2},
                        beta,
                        C::Metal.MtlArray{<:Any, 2};
                        compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    throw(ArgumentError("NextLA.gemmEx! is not supported on Metal"))
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::Metal.MtlArray{<:Any, 3},
                              B::Metal.MtlArray{<:Any, 3},
                              beta,
                              C::Metal.MtlArray{<:Any, 3})
    return _gemm_batched_mps!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::MtlMatrixBatchView,
                              B::MtlMatrixBatchView,
                              beta,
                              C::MtlMatrixBatchView)
    Adense = _dense_mtl_batch(A)
    Bdense = _dense_mtl_batch(B)
    Cdense = _dense_mtl_batch(C)
    NextLA.gemm_batched!(transA, transB, alpha, Adense, Bdense, beta, Cdense)
    return _copyback_mtl_batch!(C, Cdense)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView},
                              B::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView},
                              beta,
                              C::Union{Metal.MtlArray{<:Any, 3}, MtlMatrixBatchView})
    Adense = _dense_mtl_batch(A)
    Bdense = _dense_mtl_batch(B)
    Cdense = _dense_mtl_batch(C)
    _gemm_batched_mps!(transA, transB, alpha, Adense, Bdense, beta, Cdense)
    return _copyback_mtl_batch!(C, Cdense)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::Metal.MtlArray{<:Any, 3},
                                B::Metal.MtlArray{<:Any, 3},
                                beta,
                                C::Metal.MtlArray{<:Any, 3};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    r = NextLA._try_same_type_batched!(transA, transB, alpha, A, B, beta, C, compute_type)
    r === nothing || return r
    throw(ArgumentError("NextLA.gemmEx_batched! mixed-type batched GEMM is not supported on Metal"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::MtlMatrixBatchView,
                                B::MtlMatrixBatchView,
                                beta,
                                C::MtlMatrixBatchView;
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    r = NextLA._try_same_type_batched!(transA, transB, alpha, A, B, beta, C, compute_type)
    r === nothing || return r
    throw(ArgumentError("NextLA.gemmEx_batched! mixed-type batched GEMM is not supported on Metal"))
end

# Metal batches a strided 3-D array through MPS; it has no primitive for a
# list of separately allocated matrices. Without this method such a call -- and
# Metal's own gemmEx_batched! for the same shape, which routes here through
# _try_same_type_batched! -- falls through to the generic CPU loop in
# src/gemm/gemm_batched.jl and hands BLAS.gemm! a Metal device matrix. Refuse
# it here instead, with an error that names the backend.
function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                              B::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                              beta,
                              C::AbstractVector{<:Metal.MtlArray{<:Any, 2}})
    throw(ArgumentError("NextLA.gemm_batched! for a vector of matrices is not supported on Metal; use a 3-D MtlArray"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                                B::AbstractVector{<:Metal.MtlArray{<:Any, 2}},
                                beta,
                                C::AbstractVector{<:Metal.MtlArray{<:Any, 2}};
                                compute_type::Type = NextLA.default_compute_type(alpha, A, B, beta, C))
    r = NextLA._try_same_type_batched!(transA, transB, alpha, A, B, beta, C, compute_type)
    r === nothing || return r
    throw(ArgumentError("NextLA.gemmEx_batched! mixed-type batched GEMM is not supported on Metal"))
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              Aptrs::Metal.MtlArray,
                              Aref::AbstractMatrix,
                              Bptrs::Metal.MtlArray,
                              Bref::AbstractMatrix,
                              beta,
                              Cptrs::Metal.MtlArray,
                              Cref::AbstractMatrix,
                              batch_count::Integer)
    throw(ArgumentError("NextLA.gemm_batched! pointer-batched GEMM is not supported on Metal"))
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                Aptrs::Metal.MtlArray,
                                Aref::AbstractMatrix,
                                Bptrs::Metal.MtlArray,
                                Bref::AbstractMatrix,
                                beta,
                                Cptrs::Metal.MtlArray,
                                Cref::AbstractMatrix,
                                batch_count::Integer;
                                compute_type::Type = NextLA.default_compute_type(alpha, Aref, Bref, beta, Cref))
    throw(ArgumentError("NextLA.gemmEx_batched! pointer-batched mixed-type GEMM is not supported on Metal"))
end
