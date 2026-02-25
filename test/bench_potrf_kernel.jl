using Test, CUDA, LinearAlgebra, Printf, Plots, KernelAbstractions, GPUArrays, StochasticRounding
include("benchmark.jl")
include("../src/potrf.jl")
include("../src/potrf_left.jl")
include("../src/trsm.jl")
include("../src/symmmixedprec.jl")
include("../src/rectrxm.jl")
include("potrf.jl")