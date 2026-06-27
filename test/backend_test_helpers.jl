# Batched GEMM/SYRK tests cover both plain arrays and pointer-batched
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
_method_file(f, args...) = String(first(Base.functionloc(which(f, Tuple{typeof.(args)...}))))

function _expected_gemm_batched_file(name::String)
    name == "CPU" && return "src/gemm_batched.jl"
    name == "CUDA" && return "ext/cuda/gemm.jl"
    name == "AMDGPU" && return "ext/amdgpu/gemm.jl"
    name == "oneAPI" && return "ext/oneapi/gemm.jl"
    name == "Metal" && return "ext/metal/gemm.jl"
    error("Unknown backend `$name`")
end

function _expected_syrk_file(name::String)
    name == "CPU" && return "src/syrk.jl"
    name == "CUDA" && return "ext/cuda/syrk.jl"
    name == "AMDGPU" && return "ext/amdgpu/syrk.jl"
    name == "oneAPI" && return "ext/oneapi/syrk.jl"
    name == "Metal" && return "ext/metal/syrk.jl"
    error("Unknown backend `$name`")
end

function _expected_syrk_batched_file(name::String)
    name == "CPU" && return "src/syrk_batched.jl"
    return _expected_syrk_file(name)
end

function _expected_trsm_batched_file(name::String)
    name == "CPU" && return "src/trsm_batched.jl"
    name == "CUDA" && return "ext/cuda/trsm.jl"
    name == "AMDGPU" && return "ext/amdgpu/trsm.jl"
    name == "oneAPI" && return "ext/oneapi/trsm.jl"
    name == "Metal" && return "ext/metal/trsm.jl"
    error("Unknown backend `$name`")
end

function _expected_potrf_batched_file(name::String)
    name == "CPU" && return "src/potrf_batched.jl"
    name == "CUDA" && return "ext/cuda/potrf.jl"
    name == "AMDGPU" && return "ext/amdgpu/potrf.jl"
    name == "oneAPI" && return "ext/oneapi/potrf.jl"
    name == "Metal" && return "ext/metal/potrf.jl"
    error("Unknown backend `$name`")
end

_syrk_batched_warns(name::String, layout::Symbol) =
    name == "CPU" || name == "CUDA" || name == "Metal" || (name == "oneAPI" && layout == :pointer)
