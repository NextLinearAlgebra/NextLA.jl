module NextLAAMDGPUExt

using NextLA
using AMDGPU
using LinearAlgebra

include("amdgpu/common.jl")
include("amdgpu/gemm.jl")

end
