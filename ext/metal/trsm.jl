function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::Metal.MtlArray{<:Any,3},
                              B::Metal.MtlArray{<:Any,3},
                              alpha=one(eltype(A)))
    throw(ArgumentError("NextLA.trsm_batched! is not supported on Metal"))
end

function NextLA.trsm_batched!(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              A::AbstractVector{<:Metal.MtlArray{<:Any,2}},
                              B::AbstractVector{<:Metal.MtlArray{<:Any,2}},
                              alpha=one(eltype(first(A))))
    throw(ArgumentError("NextLA.trsm_batched! is not supported on Metal"))
end
