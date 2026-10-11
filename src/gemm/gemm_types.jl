export GEMM_COMPUTE_TYPES, default_compute_type
export gemm_alignment_quantum, aligned_leading_dimension, is_gemm_aligned

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
2. `gemm_batched!` provides the batched GEMM surface used by TLR algorithms.
3. `NextLA.gemmEx!` and `NextLA.gemmEx_batched!` are advanced APIs for
   mixed-type GEMM where the result storage may differ from the operand
   storage.

`compute_type` is chosen from the storage types of `A`, `B` and `C` by
[`default_compute_type`](@ref); `alpha` and `beta` take no part and are
converted to it. Support for the chosen combination depends on the backend.

The table below summarizes the current behavior.

Standard GEMM behavior:

| Operation | CPU | CUDA | AMDGPU | oneAPI | Metal |
| --- | --- | --- | --- | --- | --- |
| `gemmEx!` with default `compute_type` | BLAS, or a wide accumulation | cuBLAS GEMMEx | rocBLAS GEMMEx | throws | throws |
| `gemm_batched!`, 3-D strided | BLAS loop, same-type BLAS types only; a batch count of 1 broadcasts | strided batched, or GEMMEx | strided batched, or GEMMEx | oneMKL strided | MPS; equal batch counts only |
| `gemm_batched!`, vector of matrices | BLAS loop, same-type BLAS types only | pointer batched; non-native types via GEMMEx | rocBLAS batched | oneMKL batched | throws |
| `gemm_batched_ptrs!` | throws | pointer batched | pointer batched | throws | throws |
| `gemmEx_batched!` with default `compute_type` | same-type fallback only | cuBLAS batched GEMMEx | rocBLAS batched GEMMEx | same-type fallback only | same-type fallback only |
| grouped GEMM, unequal shapes | loop of `precision_gemm!` | grouped GEMMEx | throws | throws | throws |
| TF32 | throws | single, batched and pointer batched | throws | throws | throws |

The CPU column is served by BLAS. Single-matrix `gemmEx!` is the exception: there
is no CPU GEMMEx primitive, so a triple BLAS accepts goes to `BLAS.gemm!` and
everything else is accumulated in `compute_type` and rounded once on store. The
batched and precision entry points do not extend that to the CPU -- they take
what BLAS takes and raise `ArgumentError` for the rest. The CPU has no
half-precision arithmetic, so a wide accumulation there is a convenience, not a
speed-up.

A `throws` above is a vendor limitation or an unverified path rather than an
oversight. Neither oneMKL nor MPS exposes a pointer-array batched GEMM or a
heterogeneous grouped GEMM, and TF32 is CUDA hardware. `gemmEx!` on oneAPI and
Metal, and Metal's vector-of-matrices batch, throw because no device was
available to verify a same-type path; a 3-D batch is the supported Metal form. `supports_pointer_batched` and
`supports_grouped_gemm` answer for the backend, so a caller can ask before
committing to a path.

The oneAPI and Metal entries are written from the vendor APIs and have never
been executed -- neither device was available during the integration.

Representative supported `compute_type` combinations, as
`gemm_signature_supported` answers them:

| A/B storage | C storage | `compute_type` | CPU | CUDA | AMDGPU | oneAPI | Metal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Float16` | `Float16` | `Float32` | unsupported | yes | yes | unsupported | unsupported |
| `Float16` | `Float32` | `Float32` | unsupported | yes | yes | unsupported | unsupported |
| `BFloat16` | `BFloat16` | `Float32` | unsupported | yes | yes | unsupported | unsupported |
| `BFloat16` | `Float32` | `Float32` | unsupported | yes | yes | unsupported | unsupported |
| `Int8` | `Int32` | `Int32` | unsupported | yes | yes | unsupported | unsupported |
| `Float32` | `Float32` | `Float32` | BLAS | yes | yes | yes | yes |
| `Float64` | `Float64` | `Float64` | BLAS | yes | yes | yes | yes |

AMDGPU serves `BFloat16` as CUDA does: `_rocblas_datatype` maps it to
rocBLAS's `rocblas_datatype_bf16_r`, and `rocblas_gemm_ex` takes it with a
`Float32` compute type. Like the rest of the AMDGPU column it compiles but has
not run on a device.

TF32 is reached through `precision_gemm!` and its batched siblings with the
`TF32()` mode rather than a `compute_type`, and only CUDA implements it.
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

Only the storage types of `A`, `B` and `C` decide. `alpha` and `beta` are
accepted for the signature but take no part, so an untyped literal such as
`0.5` on `Float32` data cannot raise the compute type to `Float64`.

Backends may still reject unsupported storage and compute type combinations.
"""
@inline function default_compute_type(alpha, A, B, beta, C)
    T = promote_type(_batch_eltype(A), _batch_eltype(B), _batch_eltype(C))
    return _default_compute_type(T)
end

"""
    gemm_alignment_quantum(T) -> Int

Number of `T` elements spanning 16 bytes — the granularity at which offsets into
a dense operand remain 16-byte aligned.

cuBLAS documents 16-byte alignment of pointers, leading dimensions and the `m`
extent as a *performance* condition for Tensor Core kernels, not a legality one.
Unaligned operands compute correct results but select a slower kernel; measured
cost is ~2.2-2.5x for grouped GEMM on H100 and ~1.9-2.2x on Turing. Size tiles
and leading dimensions against this quantum to keep the fast kernels selected.

| `T`                  | quantum |
|:---------------------|--------:|
| `Float16`/`BFloat16` |       8 |
| `Float32`            |       4 |
| `Float64`            |       2 |

See also [`aligned_leading_dimension`](@ref), [`is_gemm_aligned`](@ref).
"""
@inline gemm_alignment_quantum(::Type{T}) where {T} = 16 ÷ gcd(16, sizeof(T))

"""
    aligned_leading_dimension(T, m) -> Int

Smallest leading dimension `>= m` that keeps every column of a column-major
dense `T` matrix 16-byte aligned.

Allocate the padded array and take a logical view over the first `m` rows:

```julia
ld = aligned_leading_dimension(Float16, 4097)   # 4104
store = CuArray{Float16}(undef, ld, n)
C = view(store, 1:4097, :)                      # stride(C, 2) == 4104
```

A column offset costs `c * ld * sizeof(T)` bytes, so an aligned `ld` makes every
column start aligned regardless of which column a panel begins at. Row offsets
are `ld`-independent — those are governed by the tile extent instead, which is
what [`is_gemm_aligned`](@ref) checks.
"""
@inline function aligned_leading_dimension(::Type{T}, m::Integer) where {T}
    q = gemm_alignment_quantum(T)
    return q * cld(m, q)
end

"""
    is_gemm_aligned(T, n) -> Bool

Whether `n` elements of `T` span a whole number of 16-byte units.

Applied to a nominal tile size this reports whether tile boundaries land on
aligned addresses (tile row `i` of a dense output starts at element offset
`(i-1) * bm`, which is independent of the leading dimension). Applied to a
leading dimension it reports whether columns do.
"""
@inline is_gemm_aligned(::Type{T}, n::Integer) where {T} =
    iszero(n % gemm_alignment_quantum(T))
