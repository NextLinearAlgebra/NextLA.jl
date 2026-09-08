"""
    quantize(matrix::AbstractMatrix{T}) where T <: AbstractFloat

Quantizes a floating-point matrix into `Float16` representation to prevent overflow.
Scales the matrix by a factor `s` if the maximum absolute value exceeds the `Float16` maximum.
Returns the quantized matrix and the scaling factor `s`.
"""
function quantize(matrix::AbstractMatrix{T}) where T <: AbstractFloat
    FP16_MAX_VAL = 65504.0f0
    alpha = maximum(abs, matrix) 
    
    if iszero(alpha)
        return similar(matrix, Float16), 1.0f0
    end

    if alpha > FP16_MAX_VAL
        s = Float32(alpha / FP16_MAX_VAL)
        quantized_matrix = similar(matrix, Float16, size(matrix))
        @. quantized_matrix = Float16(round(clamp(matrix / s, -FP16_MAX_VAL, FP16_MAX_VAL)))
    else
        s = 1.0f0
        quantized_matrix = similar(matrix, Float16, size(matrix))
        @. quantized_matrix = Float16(matrix)
    end

    return quantized_matrix, s
end

"""
    dequantize(quantized_matrix::AbstractMatrix{Float16}, s::Float32, original_eltype::DataType)

Dequantizes a `Float16` matrix back to its original element type using the scaling factor `s`.
"""
function dequantize(quantized_matrix::AbstractMatrix{Float16}, s::Float32, original_eltype::DataType)
    dequantized_matrix = similar(quantized_matrix, original_eltype, size(quantized_matrix))
    @. dequantized_matrix = quantized_matrix * s
    return dequantized_matrix
end
