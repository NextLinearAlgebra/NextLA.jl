export LowerTrapezoidal, UnitLowerTrapezoidal, UpperTrapezoidal, UnitUpperTrapezoidal, trau!, trau, tral!, tral
import LinearAlgebra: triu!, tril!, triu, tril
abstract type AbstractTrapezoidal{T} <: AbstractMatrix{T} end

# First loop through all methods that don't need special care for upper/lower and unit diagonal
for t in (:LowerTrapezoidal, :UnitLowerTrapezoidal, :UpperTrapezoidal, :UnitUpperTrapezoidal)
    @eval begin
        struct $t{T,S<:AbstractMatrix{T}} <: AbstractTrapezoidal{T}
            data::S

            function $t{T,S}(data) where {T,S<:AbstractMatrix{T}}
                Base.require_one_based_indexing(data)
                new{T,S}(data)
            end
        end
        $t(A::$t) = A
        $t{T}(A::$t{T}) where {T} = A
        $t(A::AbstractMatrix) = $t{eltype(A), typeof(A)}(A)
        $t{T}(A::AbstractMatrix) where {T} = $t(convert(AbstractMatrix{T}, A))
        $t{T}(A::$t) where {T} = $t(convert(AbstractMatrix{T}, A.data))

        AbstractMatrix{T}(A::$t) where {T} = $t{T}(A)
        AbstractMatrix{T}(A::$t{T}) where {T} = copy(A)

        Base.size(A::$t) = size(A.data)
        Base.axes(A::$t) = axes(A.data)

        Base.similar(A::$t, ::Type{T}) where {T} = $t(similar(parent(A), T))
        Base.similar(A::$t, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(A), T, dims)
        Base.parent(A::$t) = A.data

        Base.copy(A::$t) = $t(copy(A.data))

        Base.real(A::$t{<:Real}) = A
        Base.real(A::$t{<:Complex}) = (B = real(A.data); $t(B))
    end
end

Base.getindex(A::UnitLowerTrapezoidal{T}, i::Integer, j::Integer) where {T} =
    i > j ? ifelse(A.data[i,j] == nothing, zero(eltype(A.data)), A.data[i,j]) : ifelse(i == j, oneunit(T), zero(T))
Base.getindex(A::LowerTrapezoidal, i::Integer, j::Integer) =
i >= j ? ifelse(A.data[i,j] == nothing, zero(eltype(A.data)), A.data[i,j]) : zero(eltype(A.data))
Base.getindex(A::UnitUpperTrapezoidal{T}, i::Integer, j::Integer) where {T} =
    i < j ? ifelse(A.data[i,j] == nothing, zero(eltype(A.data)), A.data[i,j]) : ifelse(i == j, oneunit(T), zero(T))
Base.getindex(A::UpperTrapezoidal, i::Integer, j::Integer) =
i <= j ?  ifelse(A.data[i,j] == nothing, zero(eltype(A.data)), A.data[i,j]) : zero(eltype(A.data))