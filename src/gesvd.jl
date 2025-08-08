
struct LargeTiledMatrix{T} <: AbstractMatrix{T}
    parent::Union{Adjoint{<:AbstractMatrix{T}},AbstractMatrix{T}}
    TileRows::Vector{<:AbstractGPUMatrix{T}}
    backend::Backend
    nb_tiles::Int
    tilesinmem::Int
    cpucache::Vector{<:Array{T,2}}
end


const AbstractGPUorCPUMat{T} = Union{AbstractGPUArray{T, 2}, AbstractMatrix{T}, Adjoint{<:AbstractMatrix{T}}, Adjoint{<:AbstractGPUArray{T, 2}}}
const AbstractGPUorCPUArray{T} = Union{AbstractGPUArray{T}, AbstractArray{T}}
const AbstractGPUorLargeMatrix{T} = Union{AbstractGPUArray{T, 2}, LargeTiledMatrix{T}}

Base.adjoint(A::LargeTiledMatrix{T}) where T = LargeTiledMatrix{T}(A.parent', A.TileRows, A.backend, A.nb_tiles, A.tilesinmem, A.cpucache)

@inline @inbounds get_tileview(A::AbstractGPUorCPUMat{T}, row::Int , col::Int, TILE_SIZEx::Int=TILESIZE, TILE_SIZEy::Int=TILESIZE ) where T = 
            view(A, (row-1)*TILE_SIZEx.+(1:TILE_SIZEx),
                (col-1)*TILE_SIZEy.+(1:TILE_SIZEy))
@inline @inbounds get_rowview(A::AbstractGPUorCPUMat{T}, row::Int, startcol::Int, TILE_SIZEx::Int=TILESIZE, TILE_SIZEy::Int=TILESIZE; endcol::Int=Int(size(A,2)/TILE_SIZEy)) where T =  
            view(A, (row-1)*TILE_SIZEx .+(1:TILE_SIZEx),
                ((startcol-1)*TILE_SIZEy +1):endcol*TILE_SIZEy)
@inline @inbounds get_blockview(A::AbstractGPUorCPUMat{T}, startrow::Int, startcol::Int, TILE_SIZEx::Int=TILESIZE, TILE_SIZEy::Int=TILESIZE) where T =  
                view(A, ((startrow-1)*TILE_SIZEx .+1):size(A,1),
                    ((startcol-1)*TILE_SIZEy +1):size(A,2))
get_kernel_dims(::KernelAbstractions.Kernel{B,S}) where {B,S} = S.parameters[1]


QR1!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int; koffset::Int=0, singlerow::Bool=false, colinmem::Int=k) where T = 
                                    QR_unsafe_kernel_2d!(backend, (TILESIZE))( get_tileview(A, singlerow ? 1 : k+koffset,colinmem), 
                                    get_tileview(Tau, 1,k, TILESIZE,1), ndrange=(TILESIZE)) 
QR2!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, row::Int; koffset::Int=0, singlerow::Bool=false, A2::AnyGPUMatrix{T}=A, colinmem::Int=k) where T =
                                    QR_unsafe_kernel2_2d!(backend, (QRSPLIT, TILESIZE))(get_tileview(A,singlerow ? 1 : k+koffset,colinmem), 
                                    get_tileview(A2, singlerow ? 1 : row,colinmem), 
                                    get_tileview(Tau, 1,row,  TILESIZE,1), ndrange=(QRSPLIT,TILESIZE))

Qtapply1_par!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int; koffset::Int=0, singlerow::Bool=false, colinmem::Int=k) where T = 
                                    applyQorQt_unsafe_kernel_2d!(backend, (TILESIZE))(get_rowview(A, singlerow ? 1 : k+koffset, colinmem+1), 
                                    get_tileview(A, singlerow ? 1 : k+koffset,colinmem), 
                                    get_tileview(Tau, 1,k, TILESIZE,1), ndrange=( size(A,2)-colinmem*TILESIZE) )
Qtapply2_par!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, row::Int; koffset::Int=0, singlerow::Bool=false, A2::AnyGPUMatrix{T}=A, colinmem::Int=k) where T = 
                                    applyQorQt_unsafe_kernel2_2d!(backend, (TILESIZEMUL))(get_rowview(A,singlerow ? 1 : k+koffset, colinmem+1), 
                                    get_rowview(A2, singlerow ? 1 : row, colinmem+1), 
                                    get_tileview(A2, singlerow ? 1 : row,colinmem), 
                                    get_tileview(Tau, 1,row, TILESIZE,1), ndrange=( size(A,2)-colinmem*TILESIZE))
Qtapply2_parfused!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int; koffset::Int=0) where T = 
                                    applyQorQt_unsafe_kernel2_fused!(backend, (TILESIZEMUL))(get_rowview(A, k+koffset, k+1), 
                                    get_blockview(A,  k+1+koffset, k+1), get_rowview(A', k,k+1+koffset)', 
                                    get_rowview(Tau, 1,k+1+koffset, TILESIZE,1), Int((size(A,2)-(k+koffset)*TILESIZE)/TILESIZE), ndrange=( size(A,2)-k*TILESIZE))
QR2_fused!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int; koffset::Int=0) where T =
                                    QR_unsafe_kernel2_fused!(backend, (QRSPLIT, TILESIZE))(get_tileview(A, k+koffset,k),
                                     get_rowview(A', k, k+koffset+1)', get_rowview(Tau, 1,k+koffset+1,  TILESIZE,1),
                                     Int((size(A,2)-(k+koffset)*TILESIZE)/TILESIZE),  ndrange=(QRSPLIT,TILESIZE))

            


function mygeqrf!(A::AbstractGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, nbtiles::Int ;kend::Int=0) where {T}

    for k in 1:(nbtiles-kend)
        QRandmult!(A,Tau,k, nbtiles)
    end
    return A
end

function mygeqrf!(A::AbstractGPUMatrix{T}) where {T}
    Tau=CUDA.zeros(T, TILESIZE, Int(size(A,1)/TILESIZE))
    return mygeqrf!(A,Tau,Int(size(A,1)/TILESIZE))
end

function myblockdiag!(A::AbstractGPUorLargeMatrix{T}, Tau::AbstractGPUMatrix{T}, nbtiles::Int; kend::Int=0) where {T}

    for k in 1:(nbtiles-kend)
        QRandmult!(A,Tau,k, nbtiles)
        (k+BANDOFFSET<=nbtiles) && QRandmult!(A',Tau,k, nbtiles, LQ=true)
    end
    return A
end

function myblockdiag_unfused!(A::AbstractGPUorLargeMatrix{T}, Tau::AbstractGPUMatrix{T}, nbtiles::Int; kend::Int=0) where {T}

    for k in 1:(nbtiles-kend)
        QRandmult_notfused!(A,Tau,k, nbtiles)
        (k+BANDOFFSET<=nbtiles) && QRandmult_notfused!(A',Tau,k, nbtiles, LQ=true)
    end
    return A
end

function mycalcqr!(A::AbstractGPUorLargeMatrix{T}, Tau::AbstractGPUMatrix{T}, nbtiles::Int; kend::Int=0) where {T}

    for k in 1:(nbtiles-kend)
        QRcalcs!(A,Tau,k, nbtiles)
        (k+BANDOFFSET<=nbtiles) && QRcalcs!(A',Tau,k, nbtiles, LQ=true)
    end
    return A
end

function myapplyqr!(A::AbstractGPUorLargeMatrix{T}, Tau::AbstractGPUMatrix{T}, nbtiles::Int; kend::Int=0) where {T}

    for k in 1:(nbtiles-kend)
        QRapply!(A,Tau,k, nbtiles)
        (k+BANDOFFSET<=nbtiles) && QRapply!(A',Tau,k, nbtiles, LQ=true)
    end
    return A
end

function banddiagsvd(A::AbstractGPUMatrix{T}) where T
    mygbbrd!(A)
    KernelAbstractions.synchronize(get_backend(A))
    return bidiagsvd(A)
end

function bidiagsvd(A::AbstractGPUMatrix{T}) where T
    n=size(A,1)
    d=Float64.((Array(A[1:n+1:end])))
    e=Float64.((Array(A[n+1:n+1:end])))
    return T.(LAPACK.bdsdc!('U', 'N', d,e)[1])
end

function myblockdiag!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),TILESIZE,nbtiles)
    myblockdiag!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    unsafe_free!(Tau)
end

function mygesvd!(A::AbstractGPUMatrix)
    myblockdiag!(A)
    out=banddiagsvd(A)
    return out
end


function myblockdiag_unfused!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),TILESIZE,nbtiles)
    myblockdiag_unfused!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    unsafe_free!(Tau)
end

function myblockdiag_qrcalc!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),TILESIZE,nbtiles)
    mycalcqr!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    unsafe_free!(Tau)
end

function myblockdiag_applyqr!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),TILESIZE,nbtiles)
    myapplyqr!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    unsafe_free!(Tau)
end

function QRandmult!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int;LQ::Bool=false)  where {T}

    QR1!(A,Tau, k;koffset=(Int(LQ)*BANDOFFSET),singlerow=false)
    Qtapply1_par!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET), singlerow=false)
    triu!(get_tileview(A, k+(Int(LQ)*BANDOFFSET),k))

    if ( k+1+Int(LQ)<=(nbtiles))
        QR2_fused!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET)) 
        Qtapply2_parfused!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET))
        get_rowview(A',k, k+1+(Int(LQ)*BANDOFFSET)).=zero(T)
    end
end

function QRcalcs!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int;LQ::Bool=false)  where {T}

    QR1!(A,Tau, k;koffset=(Int(LQ)*BANDOFFSET),singlerow=false)
    triu!(get_tileview(A, k+(Int(LQ)*BANDOFFSET),k))

    if ( k+1+Int(LQ)<=(nbtiles))
        QR2_fused!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET)) 
        get_rowview(A',k, k+1+(Int(LQ)*BANDOFFSET)).=zero(T)
    end
end

function QRapply!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int;LQ::Bool=false)  where {T}

    Qtapply1_par!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET), singlerow=false)

    if ( k+1+Int(LQ)<=(nbtiles))
        Qtapply2_parfused!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET))
    end
end

function QRandmult_notfused!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int;LQ::Bool=false)  where {T}

    QR1!(A,Tau, k;koffset=(Int(LQ)*BANDOFFSET),singlerow=false)
    Qtapply1_par!(A, Tau, k; koffset=(Int(LQ)*BANDOFFSET), singlerow=false)
    triu!(get_tileview(A, k+(Int(LQ)*BANDOFFSET),k))

        for row in k+1+Int(LQ)*BANDOFFSET:(nbtiles)
            QR2!(A,Tau, k, row; koffset=Int(LQ), singlerow=false)
            Qtapply2_par!(A,Tau, k, row; koffset=Int(LQ), singlerow=false)
        end
    if ( k+1+Int(LQ)<=(nbtiles))
        get_rowview(A',k, k+1+(Int(LQ)*BANDOFFSET)).=zero(T)
    end
end



