"""
    quantize(A, T=Float16) -> (Q, s)

Stores `A` in `T` with one `Float32` scale factor, as the containers store each block:
`Q = A / s`, with `s = 1` unless the values of `A` exceed `floatmax(T)`. An empty `A` gives
an empty `Q` and `s = 1`. A non-finite `A`, or one whose values need a scale past
`floatmax(Float32)`, throws an `ArgumentError`.
"""
function quantize(A::AbstractMatrix{<:AbstractFloat}, ::Type{T}=Float16) where {T<:AbstractFloat}
    isempty(A) && return (similar(A, T), 1f0)
    return _quantize_mixedprec_block(A, T)
end

"""
    dequantize(Q, s, T) -> AbstractMatrix{T}

Inverse of `quantize`: returns `Q * s` in `T`, of the same array type as `Q`.
"""
dequantize(Q::AbstractMatrix, s::Real, ::Type{T}) where {T<:AbstractFloat} = T.(Q) .* T(s)
