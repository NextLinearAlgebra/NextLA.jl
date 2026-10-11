"""
    quantize(A, T=Float16, S=Float32) -> (Q, s)

Stores `A` in `T` with one scale factor of type `S`, as the containers store each block:
`Q = A / s`, with `s = 1` unless the values of `A` exceed `floatmax(T)`.
"""
function quantize(A::AbstractMatrix{<:AbstractFloat}, ::Type{T}=Float16,
                  ::Type{S}=Float32) where {T<:AbstractFloat, S<:AbstractFloat}
    isempty(A) && return (similar(A, T), one(S))
    return _store(A, T, S)
end

"""
    dequantize(Q, s, T) -> Matrix of T

Inverse of `quantize`: returns `Q * s` in `T`.
"""
dequantize(Q::AbstractMatrix, s::Real, ::Type{T}) where {T<:AbstractFloat} = T.(Q) .* T(s)
