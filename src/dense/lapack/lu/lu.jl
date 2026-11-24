# LU factorisation: the LAPACK-style pivoted LU, an unblocked base kernel, a
# tiled driver and the recursive LU.
#
# lu_pivot.jl keeps lu!, getrf2!, getc2!, laswp, geru! and the pivoting
# strategies.
#
# lu_base.jl and lu_tiled.jl: Source vicki-development @ 694d019, by Vicki
# <vickicar@mit.edu>. Both are pure KernelAbstractions with no calls outside
# themselves; what was changed in them, and why, is in the commit that adapted
# them. lu_base! is a single-workgroup factorisation for one block, and
# tile_lu_factor! drives a tiled one over a grid of them. reclu.jl keeps
# lu_recursive! and lu_recursive_nopiv!.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line — the export travels with it.

include("lu_pivot.jl")
include("lu_base.jl")
include("lu_tiled.jl")
include("reclu.jl")
