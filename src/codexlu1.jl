using CUDA
using CUDA.CUBLAS
using LinearAlgebra

const _libcusolver = CUDA.CUSOLVER.libcusolver

@inline function _cusolver_check(status::Integer)
    status == 0 || error("cuSOLVER error, status = $status")
    return nothing
end

function _with_cusolver_handle(f::Function)
    h = Ref{Ptr{Cvoid}}(C_NULL)
    _cusolver_check(ccall((:cusolverDnCreate, _libcusolver), Cint, (Ref{Ptr{Cvoid}},), h))
    try
        return f(h[])
    finally
        _cusolver_check(ccall((:cusolverDnDestroy, _libcusolver), Cint, (Ptr{Cvoid},), h[]))
    end
end

function _cusolver_getrf_nopiv!(A::CuMatrix{Float32}, handle::Ptr{Cvoid}; check::Bool=true)
    m, n = size(A)
    lda = stride(A, 2)
    lwork = Ref{Cint}(0)
    _cusolver_check(ccall((:cusolverDnSgetrf_bufferSize, _libcusolver), Cint,
        (Ptr{Cvoid}, Cint, Cint, CUDA.CuPtr{Float32}, Cint, Ref{Cint}),
        handle, m, n, pointer(A), lda, lwork))
    work = CuVector{Float32}(undef, lwork[])
    devinfo = CuVector{Cint}(undef, 1)
    _cusolver_check(ccall((:cusolverDnSgetrf, _libcusolver), Cint,
        (Ptr{Cvoid}, Cint, Cint, CUDA.CuPtr{Float32}, Cint, CUDA.CuPtr{Float32}, CUDA.CuPtr{Cint}, CUDA.CuPtr{Cint}),
        handle, m, n, pointer(A), lda, pointer(work), CUDA.CuPtr{Cint}(0), pointer(devinfo)))
    if check
        info = Array(devinfo)[1]
        info == 0 || throw(SingularException(info))
    end
    return A
end

function _cusolver_getrf_nopiv!(A::CuMatrix{Float64}, handle::Ptr{Cvoid}; check::Bool=true)
    m, n = size(A)
    lda = stride(A, 2)
    lwork = Ref{Cint}(0)
    _cusolver_check(ccall((:cusolverDnDgetrf_bufferSize, _libcusolver), Cint,
        (Ptr{Cvoid}, Cint, Cint, CUDA.CuPtr{Float64}, Cint, Ref{Cint}),
        handle, m, n, pointer(A), lda, lwork))
    work = CuVector{Float64}(undef, lwork[])
    devinfo = CuVector{Cint}(undef, 1)
    _cusolver_check(ccall((:cusolverDnDgetrf, _libcusolver), Cint,
        (Ptr{Cvoid}, Cint, Cint, CUDA.CuPtr{Float64}, Cint, CUDA.CuPtr{Float64}, CUDA.CuPtr{Cint}, CUDA.CuPtr{Cint}),
        handle, m, n, pointer(A), lda, pointer(work), CUDA.CuPtr{Cint}(0), pointer(devinfo)))
    if check
        info = Array(devinfo)[1]
        info == 0 || throw(SingularException(info))
    end
    return A
end

function _leaf_mp_lu_nopiv!(
    Ablk::AbstractMatrix{Thi},
    handle::Ptr{Cvoid},
    ::Type{Tlow};
    check::Bool=true
) where {Thi<:AbstractFloat, Tlow<:Union{Float32,Float64}}
    n, m = size(Ablk)
    @assert n == m
    B = CuMatrix{Tlow}(undef, n, n)
    B .= Ablk
    _cusolver_getrf_nopiv!(B, handle; check=check)
    Ablk .= B
    return nothing
end

function _reclu_nopiv_mp!(
    A::AbstractMatrix{Thi},
    handle::Ptr{Cvoid},
    leaf::Int,
    ::Type{Tlow};
    check::Bool=true
) where {Thi<:AbstractFloat, Tlow<:Union{Float32,Float64}}
    n, m = size(A)
    @assert n == m

    if n <= leaf
        _leaf_mp_lu_nopiv!(A, handle, Tlow; check=check)
        return A
    end

    n1 = n >>> 1
    r1 = 1:n1
    r2 = (n1 + 1):n

    A11 = view(A, r1, r1)
    A12 = view(A, r1, r2)
    A21 = view(A, r2, r1)
    A22 = view(A, r2, r2)

    _reclu_nopiv_mp!(A11, handle, leaf, Tlow; check=check)

    CUBLAS.trsm!('L', 'L', 'N', 'U', one(Thi), A11, A12)
    CUBLAS.trsm!('R', 'U', 'N', 'N', one(Thi), A11, A21)
    CUBLAS.gemm!('N', 'N', -one(Thi), A21, A12, one(Thi), A22)

    _reclu_nopiv_mp!(A22, handle, leaf, Tlow; check=check)

    return A
end

function lu_nopiv_recursive_mixed!(
    A::CuMatrix{Thi};
    leaf::Int = 128,
    Tlow::Type{<:Union{Float32,Float64}} = Float32,
    check::Bool = true
) where {Thi<:AbstractFloat}
    n, m = size(A)
    @assert n == m
    @assert leaf ≥ 8
    _with_cusolver_handle() do h
        _reclu_nopiv_mp!(A, h, leaf, Tlow; check=check)
    end
    return A
end