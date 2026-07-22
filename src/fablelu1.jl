module RecursiveMixedLU

using LinearAlgebra
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER

export mixed_lu!, MixedLUWorkspace

# ---------------------------------------------------------------------
# Base case: CUSOLVER getrf with pivoting DISABLED.
#
# The legacy dense API cusolverDn<t>getrf documents that passing
# devIpiv == NULL skips row pivoting entirely, which is exactly the
# non-pivoting base case we want.
# ---------------------------------------------------------------------
for (T, c) in ((:Float32, "S"), (:Float64, "D"))
    fbuf = Symbol("cusolverDn", c, "getrf_bufferSize")
    fgetrf = Symbol("cusolverDn", c, "getrf")
    @eval function getrf_nopivot!(A::StridedCuMatrix{$T})
        m, n = size(A)
        lda = max(1, stride(A, 2))
        dh = CUSOLVER.dense_handle()

        lwork = Ref{Cint}(0)
        CUSOLVER.$fbuf(dh, m, n, A, lda, lwork)
        work = CuVector{$T}(undef, max(1, Int(lwork[])))
        devinfo = CUDA.zeros(Cint, 1)

        # devIpiv = CU_NULL  =>  no pivoting
        CUSOLVER.$fgetrf(dh, m, n, A, lda, work, CU_NULL, devinfo)

        info = CUDA.@allowscalar devinfo[1]
        info < 0 && throw(ArgumentError("getrf: illegal argument #$(-info)"))
        info > 0 && throw(LinearAlgebra.SingularException(Int(info)))
        # NB: without pivoting, near-zero (but not exactly zero) pivots do
        # NOT raise info > 0 -- they just poison the factors. Caller beware.
        return A
    end
end

# ---------------------------------------------------------------------
# Workspace: preallocated device scratch, reused at every recursion
# level via reshaped views (no per-level allocation).
#   lo1  : low-precision copy of A21   (<= ceil(n/2)^2 elements)
#   lo2  : low-precision copy of A12
#   prod : Tlow product buffer, only touched on the indirect path
# ---------------------------------------------------------------------
struct MixedLUWorkspace{Tlow}
    lo1::CuVector{Tlow}
    lo2::CuVector{Tlow}
    prod::CuVector{Tlow}
end

function MixedLUWorkspace(::Type{Tlow}, ::Type{Twork}, n::Integer) where {Tlow,Twork}
    h = cld(n, 2)
    len = h * h
    if Tlow === Twork
        z = CuVector{Tlow}(undef, 0)
        return MixedLUWorkspace{Tlow}(z, z, z)          # uniform precision: no scratch needed
    end
    lo1 = CuVector{Tlow}(undef, len)
    lo2 = CuVector{Tlow}(undef, len)
    prod = direct_accumulate(Tlow, Twork) ? CuVector{Tlow}(undef, 0) :
                                            CuVector{Tlow}(undef, len)
    return MixedLUWorkspace{Tlow}(lo1, lo2, prod)
end

# Can cublasGemmEx accumulate Tlow x Tlow directly into a Twork C?
direct_accumulate(::Type{Float16}, ::Type{Float32}) = true   # tensor-core path
direct_accumulate(::Type, ::Type) = false

default_low(::Type{Float32}) = Float16
default_low(::Type{Float64}) = Float32

# ---------------------------------------------------------------------
# Mixed-precision Schur update:  A22 <- A22 - A21 * A12
# ---------------------------------------------------------------------
function schur_update!(A22::StridedCuMatrix{Twork},
                       A21::StridedCuMatrix{Twork},
                       A12::StridedCuMatrix{Twork},
                       ws::MixedLUWorkspace{Tlow}) where {Twork,Tlow}
    m, k = size(A21)
    n = size(A12, 2)

    if Tlow === Twork
        # uniform-precision baseline
        CUBLAS.gemm!('N', 'N', -one(Twork), A21, A12, one(Twork), A22)
        return A22
    end

    # down-convert panels on device (fused conversion kernels)
    L21 = reshape(view(ws.lo1, 1:m*k), m, k)
    L12 = reshape(view(ws.lo2, 1:k*n), k, n)
    L21 .= A21
    L12 .= A12

    if direct_accumulate(Tlow, Twork)
        # FP16 x FP16 -> FP32 accumulate; cublasGemmEx dispatches to
        # tensor cores on Volta+ automatically.
        CUBLAS.gemmEx!('N', 'N', -one(Twork), L21, L12, one(Twork), A22)
    else
        # e.g. Twork = Float64, Tlow = Float32: form the product in Tlow,
        # subtract in Twork with a single broadcast kernel.
        P = reshape(view(ws.prod, 1:m*n), m, n)
        CUBLAS.gemm!('N', 'N', one(Tlow), L21, L12, zero(Tlow), P)
        A22 .-= P
    end
    return A22
end

# ---------------------------------------------------------------------
# Nested recursion
# ---------------------------------------------------------------------

# Split near the midpoint, rounded up to a multiple of 16 so every panel
# leading dimension stays tensor-core / coalescing friendly.
split_size(n::Int) = clamp(((n >> 1) + 15) & -16, 1, n - 1)

function _reclu!(A::StridedCuMatrix{Twork}, nb::Int,
                 ws::MixedLUWorkspace) where {Twork}
    n = size(A, 1)
    n <= nb && return getrf_nopivot!(A)

    n1 = split_size(n)
    r1, r2 = 1:n1, (n1 + 1):n

    A11 = view(A, r1, r1)
    A12 = view(A, r1, r2)
    A21 = view(A, r2, r1)
    A22 = view(A, r2, r2)

    _reclu!(A11, nb, ws)                                    # A11 = L11 U11
    CUBLAS.trsm!('L', 'L', 'N', 'U', one(Twork), A11, A12)  # A12 <- L11 \ A12
    CUBLAS.trsm!('R', 'U', 'N', 'N', one(Twork), A11, A21)  # A21 <- A21 / U11
    schur_update!(A22, A21, A12, ws)                        # A22 <- A22 - A21 A12
    _reclu!(A22, nb, ws)                                    # A22 = L22 U22
    return A
end

"""
    mixed_lu!(A; nb = 256, Tlow = default_low(eltype(A)), ws = nothing)

In-place, non-pivoting, recursive LU factorization of the square CUDA
matrix `A`. On return, `A` holds `U` in its upper triangle and the
strictly-lower part of unit-diagonal `L` below it (LAPACK layout).

Keyword arguments:
  * `nb`   -- base-case size below which CUSOLVER `getrf` (no pivoting)
              is invoked directly. 128-512 is the usual sweet spot.
  * `Tlow` -- precision used for the Schur-complement GEMM inputs.
              Defaults: `Float16` for `Float32` matrices, `Float32` for
              `Float64` matrices. Pass `Tlow = eltype(A)` for a
              uniform-precision baseline.
  * `ws`   -- optional preallocated `MixedLUWorkspace` (reuse across
              repeated factorizations of same-size matrices).

Returns `A`.
"""
function mixed_lu!(A::StridedCuMatrix{Twork};
                   nb::Integer = 256,
                   Tlow::Type = default_low(Twork),
                   ws::Union{Nothing,MixedLUWorkspace} = nothing) where {Twork<:Union{Float32,Float64}}
    n = LinearAlgebra.checksquare(A)
    nb >= 1 || throw(ArgumentError("nb must be >= 1"))
    if !(Tlow === Twork || (Tlow, Twork) === (Float16, Float32) ||
                            (Tlow, Twork) === (Float32, Float64))
        throw(ArgumentError("unsupported precision pair (Tlow=$Tlow, Twork=$Twork)"))
    end
    w = ws === nothing ? MixedLUWorkspace(Tlow, Twork, n) : ws
    return _reclu!(A, Int(nb), w)
end

# ---------------------------------------------------------------------
# Quick self-check / demo
# ---------------------------------------------------------------------
"""
    demo(n = 4096; Twork = Float32, nb = 256)

Factor a strictly diagonally dominant random matrix (safe without
pivoting), report the relative residual ||A - L*U|| / ||A|| for both the
mixed-precision and uniform-precision runs, plus rough timings.
"""
function demo(n::Integer = 4096; Twork::Type = Float32, nb::Integer = 256)
    CUDA.functional() || error("no functional CUDA device")

    A0 = CUDA.rand(Twork, n, n)
    A0 .+= Twork(n) .* CuMatrix{Twork}(I, n, n)   # make it diagonally dominant
    normA = norm(A0)

    residual = A -> begin
        L = UnitLowerTriangular(A)
        U = UpperTriangular(A)
        norm(Matrix(L) * Matrix(U) - Matrix(A0)) / Float64(normA)
    end

    for Tlow in (default_low(Twork), Twork)
        A = copy(A0)
        mixed_lu!(A; nb, Tlow)                                # warm-up / compile
        A = copy(A0)
        t = CUDA.@elapsed mixed_lu!(A; nb, Tlow)
        tag = Tlow === Twork ? "uniform $(Twork)" : "mixed $(Twork)/$(Tlow)"
        @info "recursive LU ($tag)" n nb time_s = t rel_residual = residual(A)
    end
    return nothing
end

end # module