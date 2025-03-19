
export NextLAMatrix

mutable struct NextLAMatrix{T} 
    data::Matrix{T}
    function NextLAMatrix{T}(data::Matrix{T}) where {T}
        new{T}(data)
    end
end

#NextLAMatrix(data::Matrix{T}) where {T} = NextLAMatrix{T}(data)

Base.size(A::NextLAMatrix) = size(A.data)
Base.size(A::NextLAMatrix, i::Integer) = size(A.data, i)
Base.length(A::NextLAMatrix) = length(A.data)

Base.getindex(A::NextLAMatrix, i::Integer) = A.data[i]
Base.getindex(A::NextLAMatrix, i::Integer, j::Integer) = A.data[i,j]

Base.setindex!(A::NextLAMatrix, v, i::Integer) = A.data[i] = v
Base.setindex!(A::NextLAMatrix, v, i::Integer, j::Integer) = A.data[i,j] = v

#This temp
LinearAlgebra.BLAS.axpy!(α::Number, x::NextLAMatrix, y::NextLAMatrix) = LinearAlgebra.BLAS.axpy!(α, x.data, y.data)
LinearAlgebra.BLAS.gemv!(tA::Char, α::Number, A::NextLAMatrix, x::NextLAMatrix, β::Number, y::NextLAMatrix) = LinearAlgebra.BLAS.gemv!(tA, α, A.data, x.data, β, y.data)
LinearAlgebra.BLAS.gemm!(tA::Char, tB::Char, α::Number, A::NextLAMatrix, B::NextLAMatrix, β::Number, C::NextLAMatrix) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A.data, B.data, β, C.data)
LinearAlgebra.BLAS.trmm!(tA::Char, tB::Char, tC::Char, α::Number, A::NextLAMatrix, B::NextLAMatrix) = LinearAlgebra.BLAS.trmm!(tA, tB, tC, α, A.data, B.data)
LinearAlgebra.BLAS.trsm!(tA::Char, tB::Char, tC::Char, α::Number, A::NextLAMatrix, B::NextLAMatrix) = LinearAlgebra.BLAS.trsm!(tA, tB, tC, α, A.data, B.data)
LinearAlgebra.BLAS.syrk!(tA::Char, tC::Char, α::Number, A::NextLAMatrix, β::Number, C::NextLAMatrix) = LinearAlgebra.BLAS.syrk!(tA, tC, α, A.data, β, C.data)
LinearAlgebra.BLAS.herk!(tA::Char, tC::Char, α::Number, A::NextLAMatrix, β::Number, C::NextLAMatrix) = LinearAlgebra.BLAS.herk!(tA, tC, α, A.data, β, C.data)
LinearAlgebra.BLAS.syr2k!(tA::Char, tC::Char, α::Number, A::NextLAMatrix, B::NextLAMatrix, β::Number, C::NextLAMatrix) = LinearAlgebra.BLAS.syr2k!(tA, tC, α, A.data, B.data, β, C.data)
LinearAlgebra.BLAS.her2k!(tA::Char, tC::Char, α::Number, A::NextLAMatrix, B::NextLAMatrix, β::Number, C::NextLAMatrix) = LinearAlgebra.BLAS.her2k!(tA, tC, α, A.data, B.data, β, C.data)
