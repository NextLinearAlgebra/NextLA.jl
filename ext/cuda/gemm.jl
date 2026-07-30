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

mutable struct CUDAReusableGroupedGemmSlot{ScalarT,AT,BT,CT} <:
               NextLA.AbstractReusableGroupedGemmSlot
    max_tasks::Int
    max_groups::Int
    lookup::Dict{NextLA.GroupedGemmKey{ScalarT},Int}
    task_group::Vector{Int}
    offsets::Vector{Int}
    cursor::Vector{Int}
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
    ntask::Int
    ngroup::Int
    closed::Bool
end

function _destroy_cuda_reusable_grouped_gemm!(slot::CUDAReusableGroupedGemmSlot)
    slot.closed && return slot
    slot.closed = true
    CUDA.unsafe_free!(slot.Cptrs)
    CUDA.unsafe_free!(slot.Bptrs)
    CUDA.unsafe_free!(slot.Aptrs)
    return slot
end

NextLA._destroy_reusable_grouped_gemm!(slot::CUDAReusableGroupedGemmSlot) =
    _destroy_cuda_reusable_grouped_gemm!(slot)

@inline _reusable_compute(::NextLA.GEMMCompute{T}) where {T} =
    (_cublas_scalar_type(T), _cublas_compute_type(T))
@inline _reusable_compute(::NextLA.TF32) =
    (Float32, CUBLAS.CUBLAS_COMPUTE_32F_FAST_TF32)

function NextLA._create_reusable_grouped_gemm_slot(
    ::CUDA.CUDABackend, max_tasks::Int, max_groups::Int,
    ::Type{Tin}, ::Type{Tout}, mode::Union{NextLA.GEMMCompute,NextLA.TF32}) where {Tin,Tout}
    ScalarT, compute_enum = _reusable_compute(mode)
    lookup = Dict{NextLA.GroupedGemmKey{ScalarT},Int}()
    sizehint!(lookup, max_groups)
    task_group = Vector{Int}(undef, max_tasks)
    offsets = Vector{Int}(undef, max_groups + 1)
    cursor = Vector{Int}(undef, max_groups + 1)
    transa = Vector{CUBLAS.cublasOperation_t}(undef, max_groups)
    transb = similar(transa)
    m = Vector{Int32}(undef, max_groups); n = similar(m); k = similar(m)
    lda = similar(m); ldb = similar(m); ldc = similar(m)
    alpha = Vector{ScalarT}(undef, max_groups); beta = similar(alpha)
    group_size = Vector{Int32}(undef, max_groups)
    hostA = Vector{CUDA.CuPtr{Cvoid}}(undef, max_tasks)
    hostB = similar(hostA); hostC = similar(hostA)
    CUDA.pin(hostA); CUDA.pin(hostB); CUDA.pin(hostC)
    Aptrs = CUDA.CuArray{CUDA.CuPtr{Cvoid}}(undef, max_tasks)
    Bptrs = similar(Aptrs); Cptrs = similar(Aptrs)
    slot = CUDAReusableGroupedGemmSlot(
        max_tasks, max_groups, lookup, task_group, offsets, cursor,
        transa, transb, m, n, k, alpha, beta, lda, ldb, ldc, group_size,
        hostA, hostB, hostC, Aptrs, Bptrs, Cptrs,
        convert(CUBLAS.cudaDataType_t, Tin), convert(CUBLAS.cudaDataType_t, Tin),
        convert(CUBLAS.cudaDataType_t, Tout), compute_enum, 0, 0, false)
    finalizer(slot) do object
        try
            _destroy_cuda_reusable_grouped_gemm!(object)
        catch
        end
    end
    return slot
end

function NextLA._refresh_reusable_grouped_gemm!(
    slot::CUDAReusableGroupedGemmSlot{ScalarT}, tasks, mode) where {ScalarT}
    slot.closed && throw(ArgumentError("reusable grouped GEMM slot is closed"))
    if tasks === nothing || isempty(tasks)
        slot.ntask = 0
        slot.ngroup = 0
        return slot
    end
    ntask = length(tasks)
    ntask <= slot.max_tasks || throw(ArgumentError(
        "grouped metadata slot capacity $(slot.max_tasks) is smaller than $ntask tasks"))
    empty!(slot.lookup)
    ngroup = 0
    @inbounds for (task_index, task) in enumerate(tasks)
        convert(CUBLAS.cudaDataType_t, eltype(task.A)) == slot.Atype &&
            convert(CUBLAS.cudaDataType_t, eltype(task.B)) == slot.Btype &&
            convert(CUBLAS.cudaDataType_t, eltype(task.C)) == slot.Ctype ||
            throw(ArgumentError("task storage types do not match reusable grouped slot"))
        gm, gn, gk, glda, gldb, gldc = NextLA._gemm_dims(
            task.transA, task.transB, task.A, task.B, task.C)
        key = NextLA.GroupedGemmKey{ScalarT}(
            task.transA, task.transB, Int32(gm), Int32(gn), Int32(gk),
            Int32(glda), Int32(gldb), Int32(gldc),
            ScalarT(task.alpha), ScalarT(task.beta))
        group = get(slot.lookup, key, 0)
        if group == 0
            ngroup += 1
            ngroup <= slot.max_groups || throw(ArgumentError(
                "grouped metadata slot capacity $(slot.max_groups) is too small"))
            group = ngroup
            slot.lookup[key] = group
            slot.transa[group] = convert(CUBLAS.cublasOperation_t, key.transA)
            slot.transb[group] = convert(CUBLAS.cublasOperation_t, key.transB)
            slot.m[group] = key.m; slot.n[group] = key.n; slot.k[group] = key.k
            slot.lda[group] = key.lda; slot.ldb[group] = key.ldb; slot.ldc[group] = key.ldc
            slot.alpha[group] = key.alpha; slot.beta[group] = key.beta
            slot.group_size[group] = 0
        end
        slot.task_group[task_index] = group
        slot.group_size[group] += 1
    end
    slot.offsets[1] = 1
    @inbounds for group in 1:ngroup
        slot.offsets[group + 1] = slot.offsets[group] + slot.group_size[group]
        slot.cursor[group] = slot.offsets[group]
    end
    @inbounds for (task_index, task) in enumerate(tasks)
        group = slot.task_group[task_index]
        dest = slot.cursor[group]
        slot.cursor[group] += 1
        slot.hostA[dest] = _grouped_voidptr(task.A)
        slot.hostB[dest] = _grouped_voidptr(task.B)
        slot.hostC[dest] = _grouped_voidptr(task.C)
    end
    # With pinned host buffers these copies are enqueued on the current
    # preparation stream; the pipeline records an event immediately afterward.
    copyto!(slot.Aptrs, 1, slot.hostA, 1, ntask)
    copyto!(slot.Bptrs, 1, slot.hostB, 1, ntask)
    copyto!(slot.Cptrs, 1, slot.hostC, 1, ntask)
    slot.ntask = ntask
    slot.ngroup = ngroup
    return slot
end

function _submit_cuda_reusable_grouped_gemm!(slot::CUDAReusableGroupedGemmSlot)
    slot.closed && throw(ArgumentError("reusable grouped GEMM slot is closed"))
    slot.ngroup == 0 && return slot
    _cublas_gemm_grouped_batched_ex!(
        slot.transa, slot.transb, slot.m, slot.n, slot.k, slot.alpha,
        slot.Aptrs, slot.Atype, slot.lda, slot.Bptrs, slot.Btype, slot.ldb,
        slot.beta, slot.Cptrs, slot.Ctype, slot.ldc,
        Cint(slot.ngroup), slot.group_size, slot.compute)
    return slot
end

NextLA._submit_reusable_grouped_gemm!(slot::CUDAReusableGroupedGemmSlot) =
    _submit_cuda_reusable_grouped_gemm!(slot)

function NextLA._submit_reusable_grouped_gemm!(slot::CUDAReusableGroupedGemmSlot,
                                                alpha, beta)
    @inbounds for group in 1:slot.ngroup
        slot.alpha[group] = convert(eltype(slot.alpha), alpha)
        slot.beta[group] = convert(eltype(slot.beta), beta)
    end
    return _submit_cuda_reusable_grouped_gemm!(slot)
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
