# TLR-specific GEMM precision policy.
#
# The generic compute-mode machinery (GEMMCompute, TF32, gemm_compute_mode,
# gemm_compute_type, validate_gemm_signature) lives in NextLA. The two helpers
# below encode TLR's operand-type policy — only Float16/Float32/Float64 are
# supported, with Float16 accumulated in Float32 — and are used as the default
# `compute` mode throughout the region terms.

@inline default_gemm_compute_mode(::Type{Float16}) = GEMMCompute{Float32}()
@inline default_gemm_compute_mode(::Type{Float32}) = GEMMCompute{Float32}()
@inline default_gemm_compute_mode(::Type{Float64}) = GEMMCompute{Float64}()

function default_gemm_compute_mode(::Type{T}) where {T}
    throw(ArgumentError("TLR GEMM does not support operand type $T; supported operand types are Float16, Float32, and Float64"))
end

function validate_tlr_gemm_precision(backend, ::Type{Tin}, ::Type{Tout}, mode) where {Tin,Tout}
    Tin in (Float16, Float32, Float64) ||
        throw(ArgumentError("TLR GEMM operand type must be Float16, Float32, or Float64; got $Tin"))
    validate_gemm_signature(backend, Tin, Tin, Tin, mode)   # Stage 1/2 workspace
    validate_gemm_signature(backend, Tin, Tin, Tout, mode)  # Stage 3 destination
    return nothing
end
