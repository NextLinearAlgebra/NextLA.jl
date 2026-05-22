using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

"""
    lasq1!(n, d, e, work, info)

Julia translation of LAPACK `SLASQ1`/`DLASQ1`.
Computes singular values of a real `n×n` bidiagonal matrix with diagonal `d`
and off-diagonal `e`.
"""
function lasq1!(n::S, d::AbstractVector{T}, e::AbstractVector{T},
                work::AbstractVector{T}, info::AbstractArray{S}) where {T <: AbstractFloat, S <: Integer}
    info[1] = 0

    if n < 0
        info[1] = -1
        return
    elseif n == 0
        return
    elseif n == 1
        d[1] = abs(d[1])
        return
    elseif n == 2
        a = UpperTriangular(T[d[1] e[1]; zero(T) d[2]])
        las2!(a)
        d[1] = a[1, 1]
        d[2] = a[2, 2]
        return
    end

    sigmx = zero(T)
    for i in 1:(n - 1)
        d[i] = abs(d[i])
        sigmx = max(sigmx, abs(e[i]))
    end
    d[n] = abs(d[n])

    if sigmx == zero(T)
        sort!(view(d, 1:n), rev = true)
        return
    end

    for i in 1:n
        sigmx = max(sigmx, d[i])
    end

    epsv = if T == Float32
        ccall((@blasfunc(slamch_), libblastrampoline), Float32, (Ref{UInt8},), Ref{UInt8}('P'))
    else
        ccall((@blasfunc(dlamch_), libblastrampoline), Float64, (Ref{UInt8},), Ref{UInt8}('P'))
    end
    safmin = if T == Float32
        ccall((@blasfunc(slamch_), libblastrampoline), Float32, (Ref{UInt8},), Ref{UInt8}('S'))
    else
        ccall((@blasfunc(dlamch_), libblastrampoline), Float64, (Ref{UInt8},), Ref{UInt8}('S'))
    end
    scale = sqrt(epsv / safmin)

    @inbounds for i in 1:n
        work[2 * i - 1] = d[i]
    end
    @inbounds for i in 1:(n - 1)
        work[2 * i] = e[i]
    end

    mul = scale / sigmx
    for i in 1:(2 * n - 1)
        work[i] *= mul
    end

    for i in 1:(2 * n - 1)
        work[i] = work[i] * work[i]
    end
    work[2 * n] = zero(T)

    lasq2!(n, work, info)

    if info[1] == 0
        for i in 1:n
            d[i] = sqrt(work[i])
        end
        rscale = sigmx / scale
        for i in 1:n
            d[i] *= rscale
        end
    elseif info[1] == 2
        for i in 1:n
            d[i] = sqrt(work[2 * i - 1])
            if i <= length(e)
                e[i] = sqrt(work[2 * i])
            end
        end
        rscale = sigmx / scale
        for i in 1:n
            d[i] *= rscale
            if i <= length(e)
                e[i] *= rscale
            end
        end
    end

    return
end
