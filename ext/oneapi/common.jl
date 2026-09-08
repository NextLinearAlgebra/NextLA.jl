const oneMKL = oneAPI.oneMKL

@inline NextLA.SUBGROUP_SIZE(::Type{<:oneAPI.oneAPIBackend}) = Val(32)
