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

mutable struct CUDAPreparedGroupedGemm{ScalarT,AT,BT,CT} <: NextLA.AbstractPreparedGroupedGemm
    transa::Vector{CUBLAS.cublasOperation_t}
    transb::Vector{CUBLAS.cublasOperation_t}
    m::Vector{Int32}
    n::Vector{Int32}
    k::Vector{Int32}
    alpha::Vector{ScalarT}
    beta::Vector{ScalarT}
    lda::Vector{Int32}
    ldb::Vector{Int32}
    ldc::Vector{Int32}
    group_size::Vector{Int32}
    hostA::Vector{CUDA.CuPtr{Cvoid}}
    hostB::Vector{CUDA.CuPtr{Cvoid}}
    hostC::Vector{CUDA.CuPtr{Cvoid}}
    Aptrs::AT
    Bptrs::BT
    Cptrs::CT
    Atype::CUBLAS.cudaDataType_t
    Btype::CUBLAS.cudaDataType_t
    Ctype::CUBLAS.cudaDataType_t
    compute::CUBLAS.cublasComputeType_t
    closed::Bool
end

@inline _grouped_voidptr(A) = reinterpret(CUDA.CuPtr{Cvoid}, _grouped_cuptr(A))
@inline _grouped_pointer_aligned(A) = iszero(UInt(_grouped_cuptr(A)) & UInt(0x0f))
@inline NextLA._grouped_gemm_task_alignment_safe(task::NextLA.GroupedGemmTask, mode) =
    _grouped_pointer_aligned(task.A) && _grouped_pointer_aligned(task.B) &&
    _grouped_pointer_aligned(task.C)

function _destroy_cuda_prepared_grouped_gemm!(prepared::CUDAPreparedGroupedGemm)
    prepared.closed && return prepared
    prepared.closed = true
    CUDA.unsafe_free!(prepared.Cptrs)
    CUDA.unsafe_free!(prepared.Bptrs)
    CUDA.unsafe_free!(prepared.Aptrs)
    return prepared
end

function NextLA._destroy_prepared_grouped_gemm!(prepared::CUDAPreparedGroupedGemm)
    return _destroy_cuda_prepared_grouped_gemm!(prepared)
end

function _prepare_cuda_grouped_gemm_ex!(tasks::AbstractVector{<:NextLA.GroupedGemmTask},
                                        ::Type{ScalarT}, compute_enum) where {ScalarT}
    isempty(tasks) && return nothing
    ntask = length(tasks)
    keys = NextLA.GroupedGemmKey{ScalarT}[]
    sizehint!(keys, min(ntask, 64))
    lookup = Dict{NextLA.GroupedGemmKey{ScalarT},Int}()
    sizehint!(lookup, min(ntask, 64))
    task_group = Vector{Int}(undef, ntask)
    group_size = Int32[]
    sizehint!(group_size, min(ntask, 64))

    Atype = convert(CUBLAS.cudaDataType_t, eltype(first(tasks).A))
    Btype = convert(CUBLAS.cudaDataType_t, eltype(first(tasks).B))
    Ctype = convert(CUBLAS.cudaDataType_t, eltype(first(tasks).C))
    @inbounds for (task_index, task) in enumerate(tasks)
        convert(CUBLAS.cudaDataType_t, eltype(task.A)) == Atype &&
            convert(CUBLAS.cudaDataType_t, eltype(task.B)) == Btype &&
            convert(CUBLAS.cudaDataType_t, eltype(task.C)) == Ctype ||
            throw(ArgumentError("one prepared grouped GEMM requires uniform operand storage types"))
        gm, gn, gk, glda, gldb, gldc = NextLA._gemm_dims(
            task.transA, task.transB, task.A, task.B, task.C)
        key = NextLA.GroupedGemmKey{ScalarT}(
            task.transA, task.transB, Int32(gm), Int32(gn), Int32(gk),
            Int32(glda), Int32(gldb), Int32(gldc),
            ScalarT(task.alpha), ScalarT(task.beta))
        group = get(lookup, key, 0)
        if group == 0
            push!(keys, key)
            push!(group_size, 0)
            group = length(keys)
            lookup[key] = group
        end
        task_group[task_index] = group
        group_size[group] += 1
    end

    ngroup = length(keys)
    offsets = Vector{Int}(undef, ngroup + 1)
    offsets[1] = 1
    @inbounds for group in 1:ngroup
        offsets[group + 1] = offsets[group] + group_size[group]
    end
    cursor = copy(offsets)
    hostA = Vector{CUDA.CuPtr{Cvoid}}(undef, ntask)
    hostB = similar(hostA)
    hostC = similar(hostA)
    @inbounds for (task_index, task) in enumerate(tasks)
        group = task_group[task_index]
        dest = cursor[group]
        cursor[group] += 1
        hostA[dest] = _grouped_voidptr(task.A)
        hostB[dest] = _grouped_voidptr(task.B)
        hostC[dest] = _grouped_voidptr(task.C)
    end
    # Registration is persistent for the descriptor's lifetime and allows the
    # pipeline to refresh a slot with an asynchronous host-to-device copy.
    CUDA.pin(hostA); CUDA.pin(hostB); CUDA.pin(hostC)
    Aptrs = CUDA.CuArray(hostA)
    Bptrs = CUDA.CuArray(hostB)
    Cptrs = CUDA.CuArray(hostC)

    transa = Vector{CUBLAS.cublasOperation_t}(undef, ngroup)
    transb = similar(transa)
    m = Vector{Int32}(undef, ngroup); n = similar(m); k = similar(m)
    lda = similar(m); ldb = similar(m); ldc = similar(m)
    alpha = Vector{ScalarT}(undef, ngroup)
    beta = similar(alpha)
    @inbounds for group in 1:ngroup
        key = keys[group]
        transa[group] = convert(CUBLAS.cublasOperation_t, key.transA)
        transb[group] = convert(CUBLAS.cublasOperation_t, key.transB)
        m[group] = key.m; n[group] = key.n; k[group] = key.k
        lda[group] = key.lda; ldb[group] = key.ldb; ldc[group] = key.ldc
        alpha[group] = key.alpha; beta[group] = key.beta
    end
    prepared = CUDAPreparedGroupedGemm(
        transa, transb, m, n, k, alpha, beta, lda, ldb, ldc, group_size,
        hostA, hostB, hostC, Aptrs, Bptrs, Cptrs,
        Atype, Btype, Ctype, compute_enum, false)
    finalizer(prepared) do descriptor
        try
            _destroy_cuda_prepared_grouped_gemm!(descriptor)
        catch
            # CUDA may already be torn down during process finalization.
        end
    end
    return prepared
end

function NextLA._prepare_precision_gemm_grouped(
    tasks::AbstractVector{<:NextLA.GroupedGemmTask},
    ::NextLA.GEMMCompute{ComputeT}) where {ComputeT}
    return _prepare_cuda_grouped_gemm_ex!(
        tasks, _cublas_scalar_type(ComputeT), _cublas_compute_type(ComputeT))
end

function NextLA._prepare_precision_gemm_grouped(
    tasks::AbstractVector{<:NextLA.GroupedGemmTask}, ::NextLA.TF32)
    return _prepare_cuda_grouped_gemm_ex!(
        tasks, Float32, CUBLAS.CUBLAS_COMPUTE_32F_FAST_TF32)
end

function _submit_cuda_prepared_grouped_gemm!(prepared::CUDAPreparedGroupedGemm)
    prepared.closed && throw(ArgumentError("prepared grouped GEMM descriptor is closed"))
    _cublas_gemm_grouped_batched_ex!(
        prepared.transa, prepared.transb, prepared.m, prepared.n, prepared.k,
        prepared.alpha, prepared.Aptrs, prepared.Atype, prepared.lda,
        prepared.Bptrs, prepared.Btype, prepared.ldb,
        prepared.beta, prepared.Cptrs, prepared.Ctype, prepared.ldc,
        Cint(length(prepared.group_size)), prepared.group_size, prepared.compute)
    return prepared
end

function NextLA._precision_gemm_grouped_prepared!(prepared::CUDAPreparedGroupedGemm)
    return _submit_cuda_prepared_grouped_gemm!(prepared)
end

function NextLA._precision_gemm_grouped_prepared!(prepared::CUDAPreparedGroupedGemm,
                                                   alpha, beta)
    fill!(prepared.alpha, convert(eltype(prepared.alpha), alpha))
    fill!(prepared.beta, convert(eltype(prepared.beta), beta))
    return _submit_cuda_prepared_grouped_gemm!(prepared)
end

function NextLA._with_grouped_host_pointer_mode(f, ::CUDA.CUDABackend)
    CUBLAS.cublasSetPointerMode_v2(CUBLAS.handle(), CUBLAS.CUBLAS_POINTER_MODE_HOST)
    try
        return f()
    finally
        CUBLAS.cublasSetPointerMode_v2(CUBLAS.handle(), CUBLAS.CUBLAS_POINTER_MODE_DEVICE)
    end
end

function NextLA._with_grouped_device_pointer_mode(f, ::CUDA.CUDABackend)
    CUBLAS.cublasSetPointerMode_v2(CUBLAS.handle(), CUBLAS.CUBLAS_POINTER_MODE_DEVICE)
    try
        return f()
    finally
        CUBLAS.cublasSetPointerMode_v2(CUBLAS.handle(), CUBLAS.CUBLAS_POINTER_MODE_DEVICE)
    end
end

function _cuda_grouped_gemm_ex!(tasks::AbstractVector{<:NextLA.GroupedGemmTask}, mode)
    isempty(tasks) && return tasks
    prepared = NextLA.prepare_precision_gemm_grouped(tasks, mode)
    try
        NextLA._with_grouped_host_pointer_mode(NextLA.get_backend(first(tasks).C)) do
            NextLA.precision_gemm_grouped_prepared!(prepared)
        end
    finally
        NextLA.destroy_prepared_grouped_gemm!(prepared)
    end
    return tasks
end

function NextLA._precision_gemm_grouped!(tasks::AbstractVector{<:NextLA.GroupedGemmTask},
                                         mode::Union{NextLA.GEMMCompute,NextLA.TF32})
    return _cuda_grouped_gemm_ex!(tasks, mode)
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
