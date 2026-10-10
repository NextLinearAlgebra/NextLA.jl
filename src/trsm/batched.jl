# Batched triangular solve.
#
# Source: alecarraro-dev @ 0f1ac84 (PR #20), by Alessandro <axelcar482@gmail.com>.
# trsm_batched.jl is verbatim; this entry file is not.
#
# This folder is named for the operation rather than for batching, the same way
# gemm_batched sits inside src/gemm/. The entry file is batched.jl rather than
# trsm.jl so it is never confused with the flat src/trsm/trsm.jl, which keeps the
# KernelAbstractions kernels, the four Left/RightLower/UpperTRSM! drivers and
# the trsm front end. Nothing here replaces those; src/trsm/rectrxm.jl likewise
# keeps unified_rectrxm!.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line — the export travels with it.
#
# The vendor implementations live in ext/*/trsm.jl as package extensions,
# loaded only when the corresponding GPU package is. trsm_batched.jl carries a
# CPU fallback, so this works with no GPU package present.

include("trsm_batched.jl")
