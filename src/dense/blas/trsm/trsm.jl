# Triangular solve and multiply: TRSM, TRMM, the recursive unified_rectrxm!
# and batched TRSM.
#
# trsm_kernels.jl keeps the KernelAbstractions kernels, the four
# Left/RightLower/UpperTRSM! drivers and the trsm! front end; trmm.jl does the
# same for TRMM; rectrxm.jl keeps unified_rectrxm!, which recurses over both.
#
# trsm_batched.jl: Source alecarraro-dev @ 0f1ac84 (PR #20), by Alessandro
# <axelcar482@gmail.com>, verbatim. Its vendor implementations live in
# ext/*/trsm.jl as package extensions, loaded only when the corresponding GPU
# package is; trsm_batched.jl carries a CPU fallback, so batched TRSM works with
# no GPU package present.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line — the export travels with it.

include("trmm.jl")
include("trsm_kernels.jl")
include("rectrxm.jl")
include("trsm_batched.jl")
