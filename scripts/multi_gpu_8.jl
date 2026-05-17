#!/usr/bin/env julia
# 8-GPU 2.5D sCQR3 with c=8 (pure row-strip / 1D distribution).
# Compares against single-GPU cuSOLVER.

using LinearAlgebra, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "NextLA.jl"))
using CUDA, NCCL, NextLA, KernelAbstractions

include(joinpath(@__DIR__, "multi_gpu_scqr3.jl"))
