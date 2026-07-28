#!/usr/bin/env julia

const CONFIG_FILE = joinpath(@__DIR__, "config.jl")
include(CONFIG_FILE)
using .GemmBenchmarksConfig

benchmark = benchmark_from_args(ARGS)
script = benchmark === :dense ? "benchmark_gemm.jl" :
         benchmark === :workspace ? "benchmark_gemm_workspace.jl" :
         "benchmark_gemm_tlr_output.jl"
include(joinpath(@__DIR__, script))
