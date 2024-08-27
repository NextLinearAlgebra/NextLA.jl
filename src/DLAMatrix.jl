
export DLAMatrix

mutable struct DLAMatrix{T} 
    data::Matrix{T}
    function DLAMatrix{T}(data::Matrix{T}) where {T}
        new{T}(data)
    end
end

#DLAMatrix(data::Matrix{T}) where {T} = DLAMatrix{T}(data)

Base.size(A::DLAMatrix) = size(A.data)
Base.size(A::DLAMatrix, i::Integer) = size(A.data, i)
Base.length(A::DLAMatrix) = length(A.data)

Base.getindex(A::DLAMatrix, i::Integer) = A.data[i]
Base.getindex(A::DLAMatrix, i::Integer, j::Integer) = A.data[i,j]

Base.setindex!(A::DLAMatrix, v, i::Integer) = A.data[i] = v
Base.setindex!(A::DLAMatrix, v, i::Integer, j::Integer) = A.data[i,j] = v

#This temp
LinearAlgebra.BLAS.axpy!(α::Number, x::DLAMatrix, y::DLAMatrix) = LinearAlgebra.BLAS.axpy!(α, x.data, y.data)
LinearAlgebra.BLAS.gemv!(tA::Char, α::Number, A::DLAMatrix, x::DLAMatrix, β::Number, y::DLAMatrix) = LinearAlgebra.BLAS.gemv!(tA, α, A.data, x.data, β, y.data)
LinearAlgebra.BLAS.gemm!(tA::Char, tB::Char, α::Number, A::DLAMatrix, B::DLAMatrix, β::Number, C::DLAMatrix) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A.data, B.data, β, C.data)
LinearAlgebra.BLAS.trmm!(tA::Char, tB::Char, tC::Char, α::Number, A::DLAMatrix, B::DLAMatrix) = LinearAlgebra.BLAS.trmm!(tA, tB, tC, α, A.data, B.data)
LinearAlgebra.BLAS.trsm!(tA::Char, tB::Char, tC::Char, α::Number, A::DLAMatrix, B::DLAMatrix) = LinearAlgebra.BLAS.trsm!(tA, tB, tC, α, A.data, B.data)
LinearAlgebra.BLAS.syrk!(tA::Char, tC::Char, α::Number, A::DLAMatrix, β::Number, C::DLAMatrix) = LinearAlgebra.BLAS.syrk!(tA, tC, α, A.data, β, C.data)
LinearAlgebra.BLAS.herk!(tA::Char, tC::Char, α::Number, A::DLAMatrix, β::Number, C::DLAMatrix) = LinearAlgebra.BLAS.herk!(tA, tC, α, A.data, β, C.data)
LinearAlgebra.BLAS.syr2k!(tA::Char, tC::Char, α::Number, A::DLAMatrix, B::DLAMatrix, β::Number, C::DLAMatrix) = LinearAlgebra.BLAS.syr2k!(tA, tC, α, A.data, B.data, β, C.data)
LinearAlgebra.BLAS.her2k!(tA::Char, tC::Char, α::Number, A::DLAMatrix, B::DLAMatrix, β::Number, C::DLAMatrix) = LinearAlgebra.BLAS.her2k!(tA, tC, α, A.data, B.data, β, C.data)
