"""
    gerc!(alpha, x, y, A)

Perform the rank-1 update: A := A + alpha * x * y^H

This function computes a rank-1 update to the matrix A using the outer product
of vectors x and y, scaled by the scalar alpha. The operation performed is:
A[i,j] := A[i,j] + alpha * x[i] * conj(y[j])

This is the complex version of the rank-1 update (GER Complex), where the 
conjugate of y is used in the outer product.

# Arguments
- `alpha`: Scalar multiplier for the rank-1 update
- `x`: Vector of length m (first dimension)
- `y`: Vector of length n (second dimension)  
- `A`: m×n matrix to be updated in-place

# Algorithm
The algorithm efficiently computes the outer product by:
1. For each column j, compute temp = alpha * conj(y[j])
2. If temp ≠ 0, update column j: A[:,j] += temp * x
3. Skip columns where y[j] = 0 to avoid unnecessary computation

# Input Validation
- Matrix A must have non-negative dimensions
- Vectors x and y must have lengths matching A dimensions
- All inputs must have compatible numeric types

# Performance Notes
- Optimized for cache efficiency by operating column-wise
- Skips zero elements in y to minimize operations
- In-place operation minimizes memory allocation

# Example
```julia
m, n = 4, 3
A = zeros(ComplexF64, m, n)
x = complex.([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])
y = complex.([1.0, 0.0, 2.0], [0.5, 0.0, 1.0])
alpha = 2.0 + 1.0im
gerc!(alpha, x, y, A)  # A updated with rank-1 modification
```
"""
function gerc!(alpha::T, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    m, n = size(A)

    # Input validation with descriptive error messages
    if length(x) != m
        throw(ArgumentError("Vector x length ($(length(x))) must match matrix row dimension ($m)"))
    end
    
    if length(y) != n
        throw(ArgumentError("Vector y length ($(length(y))) must match matrix column dimension ($n)"))
    end

    # Early return for degenerate cases
    if m == 0 || n == 0 || alpha == zero(T)
        return
    end

    # Perform rank-1 update: A := A + alpha * x * y^H
    for j in 1:n
        if y[j] != zero(T)
            temp = alpha * conj(y[j])
            for i in 1:m
                A[i, j] += x[i] * temp
            end
        end
    end
end
