function axpy!(a::T, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    n = length(x)

    if n <= 0
        return
    end

    if abs(real(a)) + abs(imag(a)) == zero(eltype(x))
        return 
    end

    for i in 1:n
        y[i] = y[i] + a*x[i]
    end
end