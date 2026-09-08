module NextLAAMDGPUExt

using NextLA
using AMDGPU
using LinearAlgebra

include("amdgpu/common.jl")
include("amdgpu/gemm.jl")
include("amdgpu/potrf.jl")
include("amdgpu/syrk.jl")
include("amdgpu/trsm.jl")
include("amdgpu/rectrxm.jl")
include("amdgpu/lu.jl")

end
