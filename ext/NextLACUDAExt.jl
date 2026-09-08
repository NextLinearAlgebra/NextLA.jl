module NextLACUDAExt

using NextLA
using CUDA

include("cuda/common.jl")
include("cuda/gemm.jl")
include("cuda/trsm.jl")

end
