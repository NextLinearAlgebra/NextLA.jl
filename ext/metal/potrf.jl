function NextLA.potrf_batched!(uplo::Char,
                               A::Metal.MtlArray{<:Any,3})
    throw(ArgumentError("NextLA.potrf_batched! is not supported on Metal"))
end

function NextLA.potrf!(uplo::Char, A::Metal.MtlMatrix)
    throw(ArgumentError("NextLA.potrf! is not supported on Metal"))
end

# Named separately so it is not ambiguous with the generic Float16 method,
# which would otherwise make the same promise for an MtlMatrix it cannot keep.
function NextLA.potrf!(uplo::Char, A::Metal.MtlMatrix{Float16})
    throw(ArgumentError("NextLA.potrf! is not supported on Metal"))
end
