# Batched GEMM/SYRK tests cover both plain arrays and pointer-batched
# Vector-of-matrix inputs, so backend conversion needs to recurse into batches.
_to_backend(::Type{Array}, x) = x

_to_backend(::Type{Array}, x::AbstractVector) = [_to_backend(Array, xi) for xi in x]

_to_backend(::Type{Array}, x::AbstractArray) = x

_to_backend(AT, x::AbstractArray) = AT(x)

_to_backend(AT, x::AbstractVector) = [_to_backend(AT, xi) for xi in x]
