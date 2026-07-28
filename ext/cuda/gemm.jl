@inline NextLA.supports_grouped_gemm(::Type{<:CUDA.CUDABackend}) = true

@inline _grouped_cuptr(A::CUDA.StridedCuArray{T}) where {T} = pointer(A)
@inline _grouped_cuptr(A::Base.ReshapedArray) = _grouped_cuptr(parent(A))
@inline _grouped_cuptr(A::SubArray{T}) where {T} = Base.unsafe_convert(CUDA.CuPtr{T}, A)

"""
Call the actual cuBLAS grouped-Ex entry point with device-resident pointer
tables. CUDA.jl's generated binding currently declares `Aarray`/`Barray`/
`Carray` as host `Ptr{Ptr{Cvoid}}`; the cuBLAS API expects pointer arrays in
device memory, as do the typed grouped-GEMM bindings. Keep this narrow shim
until that generated signature is corrected upstream.
"""
function _cublas_gemm_grouped_batched_ex!(transa, transb, m, n, k, alpha,
                                           Aptrs, Atype, lda, Bptrs, Btype, ldb,
                                           beta, Cptrs, Ctype, ldc, group_count,
                                           group_size, compute)
    CUBLAS.initialize_context()
    CUBLAS.check() do
        @ccall CUBLAS.libcublas.cublasGemmGroupedBatchedEx(
            CUBLAS.handle()::CUBLAS.cublasHandle_t,
            transa::Ptr{CUBLAS.cublasOperation_t},
            transb::Ptr{CUBLAS.cublasOperation_t},
            m::Ptr{Cint}, n::Ptr{Cint}, k::Ptr{Cint},
            alpha::Ptr{Cvoid},
            Aptrs::CUDA.CuPtr{CUDA.CuPtr{Cvoid}}, Atype::CUBLAS.cudaDataType_t,
            lda::Ptr{Cint},
            Bptrs::CUDA.CuPtr{CUDA.CuPtr{Cvoid}}, Btype::CUBLAS.cudaDataType_t,
            ldb::Ptr{Cint},
            beta::Ptr{Cvoid},
            Cptrs::CUDA.CuPtr{CUDA.CuPtr{Cvoid}}, Ctype::CUBLAS.cudaDataType_t,
            ldc::Ptr{Cint},
            group_count::Cint, group_size::Ptr{Cint},
            compute::CUBLAS.cublasComputeType_t,
        )::CUBLAS.cublasStatus_t
    end
    return nothing
end

function _cuda_grouped_gemm_ex!(tasks::AbstractVector{<:NextLA.GroupedGemmTask},
                                 ::Type{ScalarT}, compute_enum) where {ScalarT}
    isempty(tasks) && return tasks
    TaskT = eltype(tasks)
    AT = typeof(first(tasks).A)
    BT = typeof(first(tasks).B)
    CT = typeof(first(tasks).C)
    keys = Tuple[]
    buckets = Vector{Vector{TaskT}}()
    @inbounds for task in tasks
        m, n, k, lda, ldb, ldc = NextLA._gemm_dims(
            task.transA, task.transB, task.A, task.B, task.C)
        # cuBLAS supplies alpha/beta once per group, not once per member.
        # Shapes with different scalars therefore need distinct groups.
        key = (task.transA, task.transB, m, n, k, lda, ldb, ldc,
               ScalarT(task.alpha), ScalarT(task.beta))
        pos = findfirst(isequal(key), keys)
        if pos === nothing
            push!(keys, key)
            push!(buckets, TaskT[])
            pos = length(buckets)
        end
        push!(buckets[pos], task)
    end

    transA = Char[]; transB = Char[]; alpha = ScalarT[]; beta = ScalarT[]
    m = Int32[]; n = Int32[]; k = Int32[]
    lda = Int32[]; ldb = Int32[]; ldc = Int32[]; group_size = Int32[]
    flatA = AT[]; flatB = BT[]; flatC = CT[]
    for bucket in buckets
        task0 = first(bucket)
        gm, gn, gk, glda, gldb, gldc = NextLA._gemm_dims(
            task0.transA, task0.transB, task0.A, task0.B, task0.C)
        push!(transA, bucket[1].transA)
        push!(transB, bucket[1].transB)
        push!(alpha, ScalarT(bucket[1].alpha))
        push!(beta, ScalarT(bucket[1].beta))
        push!(m, Int32(gm)); push!(n, Int32(gn)); push!(k, Int32(gk))
        push!(lda, Int32(glda)); push!(ldb, Int32(gldb)); push!(ldc, Int32(gldc))
        push!(group_size, Int32(length(bucket)))
        for task in bucket
            push!(flatA, task.A); push!(flatB, task.B); push!(flatC, task.C)
        end
    end
    # The grouped API consumes one flat device pointer table, ordered by
    # groups. Group metadata and per-group scalars remain host arrays.
    Aptrs = CUDA.CuArray(reinterpret.(CUDA.CuPtr{Cvoid}, _grouped_cuptr.(flatA)))
    Bptrs = CUDA.CuArray(reinterpret.(CUDA.CuPtr{Cvoid}, _grouped_cuptr.(flatB)))
    Cptrs = CUDA.CuArray(reinterpret.(CUDA.CuPtr{Cvoid}, _grouped_cuptr.(flatC)))
    transa = convert.(CUBLAS.cublasOperation_t, transA)
    transb = convert.(CUBLAS.cublasOperation_t, transB)
    try
        # cuBLAS grouped GEMM takes per-group scalars from host memory.
        CUBLAS.cublasSetPointerMode_v2(CUBLAS.handle(), CUBLAS.CUBLAS_POINTER_MODE_HOST)
        _cublas_gemm_grouped_batched_ex!(
            transa, transb, m, n, k, alpha,
            Aptrs, convert(CUBLAS.cudaDataType_t, eltype(first(flatA))), lda,
            Bptrs, convert(CUBLAS.cudaDataType_t, eltype(first(flatB))), ldb,
            beta, Cptrs, convert(CUBLAS.cudaDataType_t, eltype(first(flatC))), ldc,
            Cint(length(buckets)), group_size, compute_enum,
        )
    finally
        CUBLAS.cublasSetPointerMode_v2(CUBLAS.handle(), CUBLAS.CUBLAS_POINTER_MODE_DEVICE)
        CUDA.unsafe_free!(Cptrs)
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
    end
    return tasks
end

function NextLA._precision_gemm_grouped!(tasks::AbstractVector{<:NextLA.GroupedGemmTask},
                                          mode::NextLA.GEMMCompute{ComputeT}) where {ComputeT}
    return _cuda_grouped_gemm_ex!(tasks, _cublas_scalar_type(ComputeT),
                                  _cublas_compute_type(ComputeT))
end

function NextLA._precision_gemm_grouped!(tasks::AbstractVector{<:NextLA.GroupedGemmTask},
                                          ::NextLA.TF32)
    return _cuda_grouped_gemm_ex!(tasks, Float32,
                                  CUBLAS.CUBLAS_COMPUTE_32F_FAST_TF32)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:CUDA.CuArray{T,2}},
                              B::AbstractVector{<:CUDA.CuArray{T,2}},
                              beta,
                              C::AbstractVector{<:CUDA.CuArray{T,2}}) where {T}
    return CUBLAS.gemm_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              B::AbstractVector{<:CUDA.StridedCuMatrix{T}},
                              beta,
                              C::AbstractVector{<:CUDA.StridedCuMatrix{T}}) where {T}
    NextLA._check_batch_lengths(A, B, C)
    isempty(A) && return C

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A[1], B[1], C[1])

    Aptrs = _unsafe_batch_strided(A)
    Bptrs = _unsafe_batch_strided(B)
    Cptrs = _unsafe_batch_strided(C)

    try
        compute_enum = _cublas_compute_type(T)
        scalar_type = _cublas_scalar_type(T)
        CUBLAS.cublasGemmBatchedEx(
            CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
            Aptrs, eltype(A[1]), lda, Bptrs, eltype(B[1]), ldb,
            CUDA.CuRef{scalar_type}(beta), Cptrs, eltype(C[1]), ldc,
            length(A), compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
        )
    finally
        CUDA.unsafe_free!(Cptrs)
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
    end

    return C
end

function NextLA.gemm_batched!(transA::Char,
                              transB::Char,
                              alpha,
                              A::CUDA.StridedCuArray{<:Any,3},
                              B::CUDA.StridedCuArray{<:Any,3},
                              beta,
                              C::CUDA.StridedCuArray{<:Any,3})
    return CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end

function NextLA.gemm_batched_ptrs!(transA::Char,
                                   transB::Char,
                                   alpha,
                                   Aptrs::CUDA.CuArray,
                                   Aref::AbstractMatrix{T},
                                   Bptrs::CUDA.CuArray,
                                   Bref::AbstractMatrix{T},
                                   beta,
                                   Cptrs::CUDA.CuArray,
                                   Cref::AbstractMatrix{T},
                                   batch_count::Integer) where {T}
    batch_count <= 0 && return Cptrs

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, Aref, Bref, Cref)
    compute_enum = _cublas_compute_type(T)
    scalar_type = _cublas_scalar_type(T)

    CUBLAS.cublasGemmBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        Aptrs, eltype(Aref), lda, Bptrs, eltype(Bref), ldb,
        CUDA.CuRef{scalar_type}(beta), Cptrs, eltype(Cref), ldc,
        Int(batch_count), compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return Cptrs
end

function NextLA.gemmEx_batched_ptrs!(transA::Char,
                                     transB::Char,
                                     alpha,
                                     Aptrs::CUDA.CuArray,
                                     Aref::AbstractMatrix,
                                     Bptrs::CUDA.CuArray,
                                     Bref::AbstractMatrix,
                                     beta,
                                     Cptrs::CUDA.CuArray,
                                     Cref::AbstractMatrix,
                                     batch_count::Integer;
                                     compute_type::Type=NextLA.default_compute_type(alpha, Aref, Bref, beta, Cref))
    batch_count <= 0 && return Cptrs

    NextLA._check_compute_type(compute_type)
    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, Aref, Bref, Cref)
    compute_enum = _cublas_compute_type(compute_type)
    scalar_type = _cublas_scalar_type(compute_type)

    CUBLAS.cublasGemmBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        Aptrs, eltype(Aref), lda, Bptrs, eltype(Bref), ldb,
        CUDA.CuRef{scalar_type}(beta), Cptrs, eltype(Cref), ldc,
        Int(batch_count), compute_enum, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return Cptrs
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                                B::AbstractVector{<:CUDA.CuArray{<:Any,2}},
                                beta,
                                C::AbstractVector{<:CUDA.CuArray{<:Any,2}};
                                compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    return _gemmEx_batched_vector!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function _gemmEx_batched_vector!(transA::Char, transB::Char, alpha,
                                 A::AbstractVector{<:CUDA.StridedCuMatrix},
                                 B::AbstractVector{<:CUDA.StridedCuMatrix},
                                 beta,
                                 C::AbstractVector{<:CUDA.StridedCuMatrix},
                                 compute_type::Type)
    NextLA._check_batch_lengths(A, B, C)
    isempty(A) && return C

    Aptrs = _unsafe_batch_strided(A)
    Bptrs = _unsafe_batch_strided(B)
    Cptrs = _unsafe_batch_strided(C)

    try
        NextLA.gemmEx_batched_ptrs!(
            transA, transB, alpha, Aptrs, A[1], Bptrs, B[1], beta, Cptrs, C[1], length(C);
            compute_type,
        )
    finally
        CUDA.unsafe_free!(Cptrs)
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
    end

    return C
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::AbstractVector{<:CUDA.StridedCuMatrix{<:Any}},
                                B::AbstractVector{<:CUDA.StridedCuMatrix{<:Any}},
                                beta,
                                C::AbstractVector{<:CUDA.StridedCuMatrix{<:Any}};
                                compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    return _gemmEx_batched_vector!(transA, transB, alpha, A, B, beta, C, compute_type)
end

function NextLA.gemmEx_batched!(transA::Char,
                                transB::Char,
                                alpha,
                                A::CUDA.StridedCuArray{<:Any,3},
                                B::CUDA.StridedCuArray{<:Any,3},
                                beta,
                                C::CUDA.StridedCuArray{<:Any,3};
                                compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)
    m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchC =
        NextLA._strided_batch_layout(transA, transB, A, B, C)
    scalar_type = _cublas_scalar_type(compute_type)

    CUBLAS.cublasGemmStridedBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        A, eltype(A), lda, strideA, B, eltype(B), ldb, strideB,
        CUDA.CuRef{scalar_type}(beta), C, eltype(C), ldc, strideC, batchC,
        _cublas_compute_type(compute_type), CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return C
end

function NextLA.gemmEx!(transA::Char,
                        transB::Char,
                        alpha,
                        A::CUDA.StridedCuMatrix,
                        B::CUDA.StridedCuMatrix,
                        beta,
                        C::CUDA.StridedCuMatrix;
                        compute_type::Type=NextLA.default_compute_type(alpha, A, B, beta, C))
    NextLA._check_compute_type(compute_type)

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A, B, C)
    scalar_type = _cublas_scalar_type(compute_type)

    CUBLAS.cublasGemmEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{scalar_type}(alpha),
        A, eltype(A), lda, B, eltype(B), ldb, CUDA.CuRef{scalar_type}(beta),
        C, eltype(C), ldc, _cublas_compute_type(compute_type), CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return C
end

function NextLA._gemm_compute!(::NextLA.TF32, transA, transB, alpha,
                               A::CUDA.StridedCuMatrix, B::CUDA.StridedCuMatrix,
                               beta, C::CUDA.StridedCuMatrix)
    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A, B, C)
    CUBLAS.cublasGemmEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{Float32}(alpha),
        A, Float32, lda, B, Float32, ldb, CUDA.CuRef{Float32}(beta), C, Float32, ldc,
        CUBLAS.CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return C
end

function NextLA._gemm_compute_batched!(::NextLA.TF32, transA, transB, alpha,
                                       A::AbstractVector{<:CUDA.StridedCuMatrix},
                                       B::AbstractVector{<:CUDA.StridedCuMatrix}, beta,
                                       C::AbstractVector{<:CUDA.StridedCuMatrix})
    NextLA._check_batch_lengths(A, B, C)
    isempty(C) && return C

    m, n, k, lda, ldb, ldc = NextLA._gemm_dims(transA, transB, A[1], B[1], C[1])
    Aptrs = _unsafe_batch_strided(A)
    Bptrs = _unsafe_batch_strided(B)
    Cptrs = _unsafe_batch_strided(C)

    try
        CUBLAS.cublasGemmBatchedEx(
            CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{Float32}(alpha),
            Aptrs, Float32, lda, Bptrs, Float32, ldb, CUDA.CuRef{Float32}(beta),
            Cptrs, Float32, ldc, length(C), CUBLAS.CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS.CUBLAS_GEMM_DEFAULT,
        )
    finally
        CUDA.unsafe_free!(Cptrs)
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
    end
    return C
end

function NextLA._gemm_compute_batched!(::NextLA.TF32, transA, transB, alpha,
                                       A::CUDA.StridedCuArray{<:Any,3},
                                       B::CUDA.StridedCuArray{<:Any,3}, beta,
                                       C::CUDA.StridedCuArray{<:Any,3})
    m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchC =
        NextLA._strided_batch_layout(transA, transB, A, B, C)
    CUBLAS.cublasGemmStridedBatchedEx(
        CUBLAS.handle(), transA, transB, m, n, k, CUDA.CuRef{Float32}(alpha),
        A, Float32, lda, strideA, B, Float32, ldb, strideB,
        CUDA.CuRef{Float32}(beta), C, Float32, ldc, strideC, batchC,
        CUBLAS.CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS.CUBLAS_GEMM_DEFAULT,
    )
    return C
end
