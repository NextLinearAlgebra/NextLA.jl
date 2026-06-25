# Batched GEMM/SYRK tests cover both plain arrays and pointer-batched
# Vector-of-matrix inputs, so backend conversion needs to recurse into batches.
_to_backend(::Type{Array}, x) = x

_to_backend(::Type{Array}, x::AbstractVector) = [_to_backend(Array, xi) for xi in x]

_to_backend(::Type{Array}, x::AbstractArray) = x

_to_backend(AT, x::AbstractArray) = AT(x)

_to_backend(AT, x::AbstractVector) = [_to_backend(AT, xi) for xi in x]

# method source files give us a cheap dispatch-path assertion in tests.
_method_file(f, args...) = String(first(Base.functionloc(which(f, Tuple{typeof.(args)...}))))

function _expected_gemm_batched_file(name::String)
    name == "CPU" && return "src/gemm_batched.jl"
    name == "CUDA" && return "ext/NextLACUDAExt.jl"
    name == "AMDGPU" && return "ext/NextLAAMDGPUExt.jl"
    name == "oneAPI" && return "ext/NextLAoneAPIExt.jl"
    name == "Metal" && return "ext/NextLAMetalExt.jl"
    error("Unknown backend `$name`")
end

function _expected_syrk_file(name::String)
    name == "CPU" && return "src/syrk.jl"
    name == "CUDA" && return "ext/NextLACUDAExt.jl"
    name == "AMDGPU" && return "ext/NextLAAMDGPUExt.jl"
    name == "oneAPI" && return "ext/NextLAoneAPIExt.jl"
    name == "Metal" && return "ext/NextLAMetalExt.jl"
    error("Unknown backend `$name`")
end

function _expected_syrk_batched_file(name::String)
    name == "CPU" && return "src/syrk_batched.jl"
    return _expected_syrk_file(name)
end

function _expected_trsm_batched_file(name::String)
    name == "CPU" && return "src/trsm_batched.jl"
    name == "CUDA" && return "ext/NextLACUDAExt.jl"
    name == "AMDGPU" && return "ext/NextLAAMDGPUExt.jl"
    name == "oneAPI" && return "ext/NextLAoneAPIExt.jl"
    name == "Metal" && return "ext/NextLAMetalExt.jl"
    error("Unknown backend `$name`")
end

function _expected_potrf_batched_file(name::String)
    name == "CPU" && return "src/potrf_batched.jl"
    name == "CUDA" && return "ext/NextLACUDAExt.jl"
    name == "AMDGPU" && return "ext/NextLAAMDGPUExt.jl"
    name == "oneAPI" && return "ext/NextLAoneAPIExt.jl"
    name == "Metal" && return "ext/NextLAMetalExt.jl"
    error("Unknown backend `$name`")
end

_syrk_batched_warns(name::String, layout::Symbol) =
    name == "CPU" || name == "CUDA" || name == "Metal" || (name == "oneAPI" && layout == :pointer)
