# Batched GEMM tests cover both plain arrays and pointer-batched
# Vector-of-matrix inputs, so backend conversion needs to recurse into batches.
_to_backend(::Type{Array}, x) = x

_to_backend(::Type{Array}, x::AbstractVector) = [_to_backend(Array, xi) for xi in x]

_to_backend(::Type{Array}, x::AbstractArray) = x

_to_backend(AT, x::AbstractArray) = AT(x)

_to_backend(AT, x::AbstractVector) = [_to_backend(AT, xi) for xi in x]

function _device_pointer_batch(name::String, batch::AbstractVector)
    if name == "CUDA"
        CUDA = Base.require(Main, :CUDA)
        return CUDA.CuArray(pointer.(batch))
    elseif name == "AMDGPU"
        AMDGPU = Base.require(Main, :AMDGPU)
        return AMDGPU.ROCArray(pointer.(batch))
    end
    throw(ArgumentError("pointer batches are not supported for backend `$name`"))
end

# method source files give us a cheap dispatch-path assertion in tests.
# Separators are normalised to "/" so the expected paths below match on
# Windows, where functionloc returns backslashes.
_method_file(f, args...) =
    replace(String(first(Base.functionloc(which(f, Tuple{typeof.(args)...})))), "\\" => "/")

function _expected_gemm_batched_file(name::String)
    name == "CPU" && return "src/gemm/gemm_batched.jl"
    name == "CUDA" && return "ext/cuda/gemm.jl"
    name == "AMDGPU" && return "ext/amdgpu/gemm.jl"
    name == "oneAPI" && return "ext/oneapi/gemm.jl"
    name == "Metal" && return "ext/metal/gemm.jl"
    error("Unknown backend `$name`")
end

function _expected_potrf_batched_file(name::String)
    name == "CPU" && return "src/cholesky/potrf_batched.jl"
    name == "CUDA" && return "ext/cuda/potrf.jl"
    name == "AMDGPU" && return "ext/amdgpu/potrf.jl"
    name == "oneAPI" && return "ext/oneapi/potrf.jl"
    name == "Metal" && return "ext/metal/potrf.jl"
    error("Unknown backend `$name`")
end

function _expected_potrf_file(name::String)
    name == "CPU" && return "src/cholesky/potrf.jl"
    name == "CUDA" && return "ext/cuda/potrf.jl"
    name == "AMDGPU" && return "ext/amdgpu/potrf.jl"
    name == "Metal" && return "ext/metal/potrf.jl"
    # oneAPI has no single-matrix potrf! wrapper; the generic method's backend
    # guard is what it hits. See KNOWN_ISSUES.md.
    name == "oneAPI" && return "src/cholesky/potrf.jl"
    error("Unknown backend `$name`")
end

function _expected_syrk_file(name::String)
    name == "CPU" && return "src/syrk/syrk_dispatch.jl"
    name == "CUDA" && return "ext/cuda/syrk.jl"
    name == "AMDGPU" && return "ext/amdgpu/syrk.jl"
    name == "oneAPI" && return "ext/oneapi/syrk.jl"
    name == "Metal" && return "ext/metal/syrk.jl"
    error("Unknown backend `$name`")
end

function _expected_trsm_batched_file(name::String)
    name == "CPU" && return "src/trsm/trsm_batched.jl"
    name == "CUDA" && return "ext/cuda/trsm.jl"
    name == "AMDGPU" && return "ext/amdgpu/trsm.jl"
    name == "oneAPI" && return "ext/oneapi/trsm.jl"
    name == "Metal" && return "ext/metal/trsm.jl"
    error("Unknown backend `$name`")
end
