# LU factorisation: an unblocked base kernel and a tiled driver.
#
# Source: vicki-development @ 694d019, by Vicki <vickicar@mit.edu>. Both files
# are pure KernelAbstractions with no calls outside themselves; what was
# changed in them, and why, is in the commit that adapted them.
#
# This sits alongside the flat src/lu.jl, which keeps lu!, getrf2!, getc2!,
# laswp, geru! and the pivoting strategies. Nothing here replaces those:
# lu_base! is a single-workgroup factorisation for one block, and
# tile_lu_factor! drives a tiled one over a grid of them.
#
# The entry file is factor.jl rather than lu.jl so it is never confused with
# the flat src/lu.jl, the same reason src/trsm/ uses batched.jl.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line — the export travels with it.

include("lu_base.jl")
include("lu_tiled.jl")
