function axpy!(a::T, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    n = length(x)

    if n <= 0
        return
    end

    if abs(real(a)) + abs(imag(a)) == zero(eltype(x))
        return
    end

    # Broadcast is GPU-agnostic (no scalar indexing)
    y .= y .+ a .* x
end