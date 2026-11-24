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

# Examples
```jldoctest
julia> x = rand(ComplexF64, 1024); y = rand(ComplexF64, 1024);

julia> A = rand(ComplexF64, 1024, 1024); A0 = copy(A);

julia> NextLA.gerc!(2.0 + 1.0im, x, y, A);   # alpha must have the element type of A

julia> A ≈ A0 + (2.0 + 1.0im) * x * y'
true
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

    # Rank-1 update: A := A + alpha * x * y^H (GPU-agnostic: broadcast instead of scalar indexing)
    # y' is adjoint (conjugate transpose), so y'[1,j] = conj(y[j])
    A .+= alpha .* (x .* y')
end
