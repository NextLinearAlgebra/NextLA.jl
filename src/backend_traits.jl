# Backend capability traits, taken verbatim from alecarraro-dev @ 0f1ac84,
# where they sat inline in src/NextLA.jl.
#
# They are module-level rather than a feature's: every method is added by a
# vendor extension in ext/*/common.jl, and every caller is somewhere else
# again. SUBGROUP_SIZE and unwrap are read by TLR's norm reduction
# (TLR/compression/norms.jl), supports_pointer_batched by TLR's compressed
# accumulation (TLR/gemm/compressed_accumulation/run_coupling.jl). Nothing
# under src/gemm/ uses any of them, so that is not where they belong.

@inline SUBGROUP_SIZE(::Type{<:KernelAbstractions.CPU}) = Val(1)
@inline unwrap(::Val{x}) where {x} = x
@inline supports_pointer_batched(backend) = supports_pointer_batched(typeof(backend))
@inline supports_pointer_batched(::Type) = false
