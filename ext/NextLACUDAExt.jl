module NextLACUDAExt

using NextLA
using CUDA

include("cuda/common.jl")
include("cuda/gemm.jl")
include("cuda/syrk.jl")
include("cuda/trsm.jl")
include("cuda/potrf.jl")
include("cuda/streams.jl")

end
