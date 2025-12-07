using Test, Random, LinearAlgebra, Printf, KernelAbstractions, AMDGPU, CUDA, GPUArrays, GPUArraysCore, oneAPI
 include("../src/cholesky_tree_subdiv.jl")
const backend=KernelAbstractions.get_backend(oneArray(rand(Float32, 2,2)))
include("cholesky_acc.jl")
include("cholesky_time.jl")
