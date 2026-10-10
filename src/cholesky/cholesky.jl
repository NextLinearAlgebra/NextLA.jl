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
# potrf_batched! (src/cholesky/potrf_batched.jl) is the batch entry point; TLR
# is its only caller.
#
# potrf_kernel.jl and recpotrf.jl are Vicki's, ported off CUDA. Her
# SymmMixedPrec method is not here: mixed precision owns its own tree under
# src/mxp/, and recpotrf_mixedprec.jl arrives there. The kernel is src/cholesky.jl from the five vicki-development-* vendor
# pins -- it is not on the parent branch -- and the recursive driver is
# src/cholesky_tree_subdiv.jl from the parent. See KNOWN_ISSUES.md.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line -- the export travels with it.

include("potrf.jl")
include("potrf_kernel.jl")
include("potrf_blocked.jl")
# After potrf_blocked.jl: CHOL_BLOCK is defined there and this file derives
# its thread mapping from it. cholesky_blocked! only names the kernel at call
# time, so the driver coming first costs nothing.
include("potrf_register.jl")
include("recpotrf.jl")
