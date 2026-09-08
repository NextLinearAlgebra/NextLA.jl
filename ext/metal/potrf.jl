function NextLA.potrf_batched!(uplo::Char,
                               A::Metal.MtlArray{<:Any,3})
    throw(ArgumentError("NextLA.potrf_batched! is not supported on Metal"))
end
