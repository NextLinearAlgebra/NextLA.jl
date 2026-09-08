# GEMM: the extended, batched, grouped and mixed-precision forms.
#
# Source: alecarraro-dev @ 0f1ac84 (PR #20), by Alessandro <axelcar482@gmail.com>.
# The five gemm_* files are verbatim; this entry file is not.
#
# This sits alongside src/gemm/matmul.jl, which keeps the tiled KernelAbstractions
# GEMM and GEMM_ADD!/GEMM_SUB!. Nothing here replaces it.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line — the export travels with it.
#
# The vendor implementations live in ext/ as package extensions, loaded only
# when the corresponding GPU package is. gemm_batched.jl carries CPU fallbacks,
# so this feature works with no GPU package present.

include("gemm_types.jl")
include("gemmEx.jl")
include("gemm_batched.jl")
include("gemm_precision.jl")
include("gemm_grouped.jl")
