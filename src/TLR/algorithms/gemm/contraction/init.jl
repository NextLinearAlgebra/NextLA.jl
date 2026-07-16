# Output/init operand policy.
#
# A contraction's output is both a destination and an **init operand**: the policy states
# how the operation combines with the value already in `C`. It names the rule the code
# follows — a region's first writer folds β, later writers accumulate.
#
# One type, two constructors. `AccumulateExisting(T)` is exactly `ScaleExisting(one(T))`
# in every behaviour, and nothing ever needed to dispatch on the difference: the only
# question anyone asks is `isone(β)`. The intent a term wants to express ("I am not this
# region's first writer") is carried by the constructor *name* at the call site, which is
# where a reader needs it — it does not need to be a second type.
#
# This says what to do, not how. Choosing *how* — fold β into the terminal GEMM, or
# pre-scale the region and accumulate — depends on the reduction placement and so lives
# with the scheduler (`lower_init!` in schedule.jl). A future TLR output sink is a third
# lowering of the same policy.

"""
    InitPolicy(beta)

How a structured contraction combines with the existing contents of its output:
`beta·C + <contraction>`. `beta` is in *compute* precision, matching the GEMM scalars
(see the precision policy). Build it through `ScaleExisting` or `AccumulateExisting`.
"""
struct InitPolicy{T<:Number}
    beta::T
end

"""First writer of an output region: the result is `beta·C + <contraction>`."""
@inline ScaleExisting(beta::Number) = InitPolicy(beta)

"""
A later writer of an output region: the existing value is kept and added to. Takes the
compute scalar type because the lowering must hand a typed `β = 1` to the GEMM.
"""
@inline AccumulateExisting(::Type{T}) where {T<:Number} = InitPolicy(one(T))

"""The compute-typed `β` the policy represents, before any placement-specific lowering."""
@inline init_beta(p::InitPolicy) = p.beta

"""True when the policy leaves the existing output untouched, so no scaling is needed."""
@inline is_accumulate(p::InitPolicy) = isone(p.beta)
