module NextLAoneAPIExt

using NextLA
using oneAPI

include("oneapi/common.jl")
include("oneapi/gemm.jl")
include("oneapi/syrk.jl")
include("oneapi/trsm.jl")
include("oneapi/potrf.jl")

end
