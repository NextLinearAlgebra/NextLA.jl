# Symmetric rank-k update: C := alpha*op(A)*op(A)' + beta*C.
#
# Source: origin/pr/19, by Alessandro <axelcar482@gmail.com>. That PR was
# closed on 19 Aug 2026 on the grounds that its wrappers had moved into
# alecarraro-dev. Only the gemm_batched! half did. No syrk survives anywhere on
# that branch -- src/syrk.jl, src/syrk_batched.jl, test/syrk.jl and every syrk
# method in ext/ went with the PR. This folder restores them; see
# KNOWN_ISSUES.md.
#
# syrk_dispatch.jl and syrk_batched.jl are verbatim but for the wording of the
# fallback @warn, which announced a fall back to batched GEMM on paths that
# actually loop single syrk! calls.
#
# The vendor methods live in ext/{cuda,amdgpu,oneapi,metal}/syrk.jl, so this
# file is CPU-only by design: syrk! forwards to BLAS.syrk!, and syrk_batched!
# loops it.
#
# Layout contract: this is the only file here that includes. Exports stay in
# the file that defines the symbol, so moving a unit to another feature is a
# git mv plus one include line -- the export travels with it.

include("syrk_dispatch.jl")
include("syrk_batched.jl")
