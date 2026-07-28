export GEMM_COMPUTE_TYPES, default_compute_type

"""
    GEMM_COMPUTE_TYPES

`compute_type` values inferred or used internally by the advanced
`NextLA.gemmEx!` and `NextLA.gemmEx_batched!` APIs.

`compute_type` is the Julia type used to request the GEMM accumulation type.
These are plain Julia types, not backend-specific math modes such as TF32 or
other fast-math settings.

`NextLA` separates its primary BLAS surface from the advanced `Ex` interfaces:

1. Ordinary dense GEMM is intentionally left to the backend itself, e.g.
   `C = A * B` for dense matrices.
2. `gemm_batched!` and `syrk!` form the main public surface for GEMM-like BLAS
   wrappers.
3. `NextLA.gemmEx!` and `NextLA.gemmEx_batched!` are advanced APIs for
   mixed-type GEMM where the result storage may differ from the operand
   storage.

`compute_type` is inferred from `alpha`, `beta`, and the operand/result types
using [`default_compute_type`](@ref). Support for the inferred combination
depends on the backend.

The table below summarizes the current behavior.

Standard GEMM behavior:

| Operation | CPU | CUDA | AMDGPU | oneAPI | Metal |
| --- | --- | --- | --- | --- | --- |
| `gemmEx!` with default `compute_type` | unsupported | cuBLAS GEMMEx | rocBLAS GEMMEx | unsupported | unsupported |
| `gemm_batched!` | loop of standard GEMMs | pointer or strided batched GEMM | pointer or strided batched GEMM | pointer or strided batched GEMM | strided MPS matmul |
| `gemmEx_batched!` with default `compute_type` | same-type fallback only | cuBLAS batched GEMMEx | rocBLAS batched GEMMEx | same-type fallback only | same-type fallback only |
| `syrk!` | `BLAS.syrk!` | `CUBLAS.syrk!` | native rocBLAS SYRK | `oneMKL.syrk!` | MPS GEMM fallback |

Representative supported `compute_type` combinations:

| A/B storage | C storage | `compute_type` | CUDA | AMDGPU | oneAPI | Metal |
| --- | --- | --- | --- | --- | --- | --- |
| `Float16` | `Float16` | `Float32` | yes | yes | unsupported | unsupported |
| `Float16` | `Float32` | `Float32` | yes | yes | unsupported | unsupported |
| `BFloat16` | `BFloat16` | `Float32` | CUDA grouped GEMMEx (SM80+) | unsupported | unsupported | unsupported |
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

"""
    NATIVE_GEMM_TYPES

Element types for which the GPU backends expose a native same-type GEMM. Used by
the backend extensions to choose between the native BLAS call and the generic
`Ex` path.
"""
const NATIVE_GEMM_TYPES = Union{Float16, Float32, Float64, ComplexF32, ComplexF64}

@inline _supports_native_gemm(::Type{T}, ::Type{T}, ::Type{T}) where {T<:NATIVE_GEMM_TYPES} = true
@inline _supports_native_gemm(::Type, ::Type, ::Type) = false

@inline function _default_compute_type(::Type{T}) where {T}
    if T <: Complex
        return _default_compute_type(real(T)) === Float64 ? ComplexF64 : ComplexF32
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

Return the default `compute_type` inferred by `NextLA` for mixed-type GEMM.

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
        eltype(alpha),
        _batch_eltype(A),
        _batch_eltype(B),
        eltype(beta),
        _batch_eltype(C),
    )
    return _default_compute_type(T)
end
