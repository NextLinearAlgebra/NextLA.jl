struct TransposedMixedPrec{T, M <: AbstractMixedPrec{T}} <: AbstractMixedPrec{T}
    parent::M
end

Base.transpose(A::AbstractMixedPrec) = TransposedMixedPrec(A)
Base.transpose(A::TransposedMixedPrec) = parent(A)

Base.parent(A::TransposedMixedPrec) = A.parent
Base.size(A::TransposedMixedPrec) = reverse(size(parent(A)))

Base.getindex(A::TransposedMixedPrec, i::Int, j::Int) = parent(A)[j, i]
