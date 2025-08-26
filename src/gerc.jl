function gerc!(alpha::T, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    m, n = size(A)

    if m < 0
        return 1
    end

    if n < 0
        return 2
    end

    if m == 0 || n == 0 || alpha == zero(T)
        return
    end

    jy = 1

    for j in 1:n
        if y[jy] != zero(T)
            temp = alpha * conj(y[jy])
            for i in 1:m
                A[i, j] += x[i] * temp
            end
        end

        jy += 1
    end

    return
end
