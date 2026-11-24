# QR: Householder reflectors and the tiled-QR kernels.
#
# geqr2/geqrt/unmqr factor a tile and apply its Q; larfg/larf/larft/larfb build
# and apply Householder reflectors; tsqrt/tsmqr and ttqrt/ttmqr factor and
# apply Q for a triangle stacked on a square or on another triangle. parfb
# applies a block reflector to a pair of tiles, with pamm and pemv (PLASMA's
# CORE_zpamm and CORE_zpemv) as its building blocks.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line — the export travels with it.

include("geqrt.jl")
include("geqr2.jl")
include("larf.jl")
include("larfg.jl")
include("larft.jl")
include("larfb.jl")
include("unmqr.jl")
include("tsqrt.jl")
include("tsmqr.jl")
include("parfb.jl")
include("pamm.jl")
include("pemv.jl")
include("ttmqr.jl")
include("ttqrt.jl")
