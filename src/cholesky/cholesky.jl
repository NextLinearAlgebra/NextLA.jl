# Cholesky factorization of a symmetric positive definite matrix.
#
# potrf.jl is the single-matrix entry point and is new here rather than taken
# from a branch. The branches reach their Cholesky base case through
# vicki-development's src/wrappers.jl, which is a vendor shim: `using CUDA` and
# `using AMDGPU` at top level, and no CPU method at all. That file has been
# turned down five times now -- for recgemm, unified_rec_mixed, reclu, syrk,
# and here -- so potrf! is written portably instead, in the shape Alessandro
# gave syrk! in src/syrk/syrk_dispatch.jl: dimension checks, then the reference
# library on the CPU, with the vendor calls living in
# ext/{cuda,amdgpu,oneapi,metal}/potrf.jl beside the batched wrappers already
# there. See KNOWN_ISSUES.md.
#
# potrf_batched! (src/cholesky/potrf_batched.jl) is the batch entry point and stays
# where it is; it arrived with TLR, which is its only caller.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line -- the export travels with it.

include("potrf.jl")
