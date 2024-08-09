using LinearAlgebra 
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools

function larfb!(::Type{T}, side::AbstractChar, trans::AbstractChar, direct::AbstractChar, 
    storev::AbstractChar, V::AbstractMatrix{T}, Tau::AbstractMatrix{T}, 
    C::AbstractMatrix{T}) where {T<: Number}

    m,n = size(C)
    ldt,k = size(Tau)
    ldv = max(1, stride(V,2))
    ldc = max(1, stride(C,2))


    if side == 'L'
        ldw = max(1,n)
    else
        ldw = max(1,m)
    end

    work = Vector{T}(undef, ldw*k)

    if m > 0 && n > 0
        if T == ComplexF64
            ccall((@blasfunc(zlarfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, V, ldv, Tau, ldt, C, ldc, work, ldw)

        elseif T == ComplexF32
            ccall((@blasfunc(clarfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, V, ldv, Tau, ldt, C, ldc, work, ldw)

        elseif T == Float64
            ccall((@blasfunc(dlarfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, V, ldv, Tau, ldt, C, ldc, work, ldw)

        else #  T == Float32
            ccall((@blasfunc(slarfb_), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8},Ref{UInt8},Ref{UInt8},
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
                Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            side, trans, direct, storev, m, n, k, V, ldv, Tau, ldt, C, ldc, work, ldw)

        end
    end
end
