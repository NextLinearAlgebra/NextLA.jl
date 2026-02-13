"""
Shared LAPACK reference wrappers for NextLA tests.
These call LAPACK directly via libblastrampoline for comparison testing.
Uses @eval loop to generate type-specific methods (ccall requires compile-time constant names).
"""

using LinearAlgebra: libblastrampoline, BlasInt
using LinearAlgebra.LAPACK: chklapackerror
using LinearAlgebra.BLAS: @blasfunc

const TEST_TYPES = (Float32, Float64, ComplexF32, ComplexF64)

# Tolerance helper: single‑precision types get looser tolerance
test_rtol(::Type{T}) where {T} = (T <: Union{Float32, ComplexF32}) ? 1e-5 : 1e-12

# ── xLARFG — elementary reflector generation ─────────────────────────────────
# Reference for larfg! comparison tests.
for (elty, func) in ((Float64,    :dlarfg_),
                      (Float32,    :slarfg_),
                      (ComplexF64, :zlarfg_),
                      (ComplexF32, :clarfg_))
    @eval begin
        function lapack_larfg!(x::AbstractVector{$elty})
            N     = BlasInt(length(x))
            alpha = Ref{$elty}(x[1])
            incx  = BlasInt(1)
            tau   = Ref{$elty}(0)
            ccall((@blasfunc($func), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                  N, alpha, pointer(x, 2), incx, tau)
            return tau[], alpha[]
        end
    end
end

# ── xTPQRT — triangular‑pentagonal QR factorization ─────────────────────────
# Reference for tsqrt! (l=0) and ttqrt! (l=n).
for (elty, func) in ((Float64,    :dtpqrt_),
                      (Float32,    :stpqrt_),
                      (ComplexF64, :ztpqrt_),
                      (ComplexF32, :ctpqrt_))
    @eval begin
        function lapack_tpqrt!(::Type{$elty}, m::Int, n::Int, l::Int, nb::Int,
                               A::AbstractMatrix{$elty}, lda::Int,
                               B::AbstractMatrix{$elty}, ldb::Int,
                               Tau::AbstractMatrix{$elty}, ldt::Int,
                               work)
            info = Ref{BlasInt}(0)
            ccall((@blasfunc($func), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                  m, n, l, nb, A, lda, B, ldb, Tau, ldt, work, info)
            chklapackerror(info[])
        end
    end
end

# ── xTPMQRT — apply Q from xTPQRT factorization ─────────────────────────────
# Reference for tsmqr! (l=0) and ttmqr! (l>0).
for (elty, func) in ((Float64,    :dtpmqrt_),
                      (Float32,    :stpmqrt_),
                      (ComplexF64, :ztpmqrt_),
                      (ComplexF32, :ctpmqrt_))
    @eval begin
        function lapack_tpmqrt!(::Type{$elty}, side::Char, trans::Char, l::Int,
                                V::AbstractMatrix{$elty}, Tau::AbstractMatrix{$elty},
                                A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            m1, n1 = size(A)
            m, n = size(B)
            nb, k = size(Tau)

            ldv = max(1, stride(V, 2))
            ldt = max(1, stride(Tau, 2))
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))

            work = if side == 'L'
                Matrix{$elty}(undef, nb, n1)
            else
                Matrix{$elty}(undef, m1, nb)
            end

            transflag = (trans == 'C' && $elty <: Real) ? 'T' : trans

            info = Ref{BlasInt}(0)
            ccall((@blasfunc($func), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}),
                  side, transflag, m, n, k, l, nb,
                  V, ldv, Tau, ldt, A, lda, B, ldb, work, info)
            chklapackerror(info[])
        end
    end
end

# ── xTPRFB — apply block reflector ──────────────────────────────────────────
# Reference for parfb!.
for (elty, func) in ((Float64,    :dtprfb_),
                      (Float32,    :stprfb_),
                      (ComplexF64, :ztprfb_),
                      (ComplexF32, :ctprfb_))
    @eval begin
        function lapack_tprfb!(::Type{$elty}, side::Char, trans::Char,
                               direct::Char, storev::Char, l::Int,
                               V::AbstractMatrix{$elty}, Tee::AbstractMatrix{$elty},
                               A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            m, n = size(B)
            ldt, k = size(Tee)
            ldv = max(1, stride(V, 2))
            lda = max(1, stride(A, 2))
            ldb = max(1, m)

            if side == 'L'
                work = Matrix{$elty}(undef, k, n)
                ldw = k
            else
                work = Matrix{$elty}(undef, m, k)
                ldw = m
            end

            if m > 0 && n > 0
                ccall((@blasfunc($func), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}),
                      side, trans, direct, storev,
                      m, n, k, l,
                      V, ldv, Tee, ldt, A, lda, B, ldb, work, ldw)
            end
            return work
        end
    end
end
