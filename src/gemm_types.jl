export GEMM_COMPUTE_TYPES, default_compute_type

"""
    GEMM_COMPUTE_TYPES

`compute_type` values accepted by the advanced `NextLA.gemmEx!` and
`NextLA.gemmEx_batched!` APIs.

`compute_type` is the Julia type used to request the GEMM accumulation type.
These are plain Julia types, not backend-specific math modes such as TF32 or
other fast-math settings.

`NextLA` separates its primary BLAS surface from the advanced `Ex` interfaces:

1. Ordinary dense GEMM is intentionally left to the backend itself, e.g.
   `C = A * B` for dense matrices.
2. `gemm_batched!` and `syrk!` form the main public surface for GEMM-like BLAS
   wrappers.
3. `NextLA.gemmEx!` and `NextLA.gemmEx_batched!` are advanced APIs to use when
   you want to choose `compute_type` explicitly.

If `compute_type` matches the backend's ordinary GEMM behavior, the `Ex`
methods fall back to the corresponding standard GEMM path. If you request a
non-default `compute_type`, support depends on the backend.

The table below summarizes the current behavior.

Standard GEMM behavior:

| Operation | CPU | CUDA | AMDGPU | oneAPI | Metal |
| --- | --- | --- | --- | --- | --- |
| `gemmEx!` with default `compute_type` | unsupported | `CUBLAS.gemm!` | `rocBLAS.gemm!` | unsupported | unsupported |
| `gemm_batched!` | loop of standard GEMMs | pointer or strided batched GEMM | pointer or strided batched GEMM | pointer or strided batched GEMM | strided MPS matmul |
| `gemmEx_batched!` with default `compute_type` | unsupported | falls back to `gemm_batched!` | falls back to `gemm_batched!` | unsupported | unsupported |
| `syrk!` | `BLAS.syrk!` | `CUBLAS.syrk!` | native rocBLAS SYRK | `oneMKL.syrk!` | MPS GEMM fallback |

Representative non-default `compute_type` combinations:

| A/B storage | C storage | `compute_type` | CUDA | AMDGPU | oneAPI | Metal |
| --- | --- | --- | --- | --- | --- | --- |
| `Float16` | `Float16` | `Float32` | yes | yes | unsupported | unsupported |
| `Float16` | `Float32` | `Float32` | yes | yes | unsupported | unsupported |
| `Float32` | `Float32` | `Float16` | yes | yes | unsupported | unsupported |
| `Int8` | `Int32` | `Int32` | yes | yes | unsupported | unsupported |

TF32 and other backend-specific fast-math modes are intentionally excluded from
these APIs for now.
"""
const GEMM_COMPUTE_TYPES = (
    Float16,
    Float32,
    Float64,
    ComplexF32,
    ComplexF64,
    Int32,
)

@inline _gemm_typeof(x) = typeof(x)
@inline _gemm_typeof(x::AbstractArray{T}) where {T} = T
@inline _gemm_typeof(x::AbstractVector{<:AbstractArray{T}}) where {T} = T

@inline function _default_compute_type(::Type{T}) where {T}
    if T <: Complex
        return _default_compute_type(T.parameters[1]) === Float64 ? ComplexF64 : ComplexF32
    elseif T === Float16
        return Float16
    elseif T <: AbstractFloat
        return T <: Float64 ? Float64 : Float32
    elseif T <: Integer
        return Int32
    end
    return Float32
end

@inline function _check_compute_type(compute_type::Type)
    compute_type in GEMM_COMPUTE_TYPES ||
        throw(ArgumentError("unsupported GEMM compute_type `$compute_type`; supported compute types are $(join(string.(GEMM_COMPUTE_TYPES), ", "))"))
    return compute_type
end

"""
    default_compute_type(alpha, A, B, beta, C)

Return the default `compute_type` used by `NextLA` when `compute_type` is not
specified explicitly.

The rule is intentionally simple and backend-agnostic:
- `Float16` defaults to `Float16`
- `Float32` defaults to `Float32`
- `Float64` defaults to `Float64`
- integer inputs default to `Int32`
- complex inputs follow the corresponding real compute type

Backends may still reject unsupported storage and compute type combinations.
"""
@inline function default_compute_type(alpha, A, B, beta, C)
    T = promote_type(
        _gemm_typeof(alpha),
        _gemm_typeof(A),
        _gemm_typeof(B),
        _gemm_typeof(beta),
        _gemm_typeof(C),
    )
    return _default_compute_type(T)
end
