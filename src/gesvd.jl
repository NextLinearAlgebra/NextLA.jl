#TODO
#non-square and non-power of two matrices, and non-multiples of tilesize
#adding singular vectors
#include openblas gbbrd from nextLA
#remove some of the copies by fixing copies between views of GPU and CPU arrays
#naming conventions

struct LargeTiledMatrix{T} <: AbstractMatrix{T}
    parent::Union{Adjoint{<:AbstractMatrix{T}},AbstractMatrix{T}}
    TileRows::Vector{<:AbstractGPUMatrix{T}}
    backend::Backend
    nb_tiles::Int
    tilesinmem::Int
    cpucache::Vector{<:Array{T,2}}
end
function LargeTiledMatrix(input::AbstractMatrix{T}, backend::Backend, tilesinmem::Int) where {T}
    tilesinmem = min(Int((size(input,1))/TILESIZE), tilesinmem)
    rows=Array{AbstractGPUMatrix{T}}(undef, 0)
    Row=KernelAbstractions.zeros(backend, T, TILESIZE, tilesinmem*TILESIZE)
    copyto!(Row,copy(view(input,1:TILESIZE,1:tilesinmem*TILESIZE)))
    push!(rows, Row)
    for _ in 1:3
        Row=KernelAbstractions.zeros(backend, T, TILESIZE, tilesinmem*TILESIZE)
        push!(rows, Row)
    end
    return LargeTiledMatrix(input,rows, backend, Int(size(input,1)/TILESIZE),tilesinmem,
                         [zeros(T, TILESIZE, tilesinmem*TILESIZE), zeros(T, TILESIZE, (tilesinmem-1)*TILESIZE),  
                         zeros(T, TILESIZE, TILESIZE), zeros(T,1,1),zeros(T,1,1)])
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

            

function OOC_alg!(A::Matrix{T}, f::Function, kswitch::Int,tilesinmem::Int) where {T}
    n=size(A,1)
    nb_tiles= Int(n/TILESIZE)
    Tau=KernelAbstractions.zeros(backend, T, TILESIZE, nb_tiles)
    
    if (n>kswitch*TILESIZE)
        inputmatrix=LargeTiledMatrix(A, backend, tilesinmem)
        f(inputmatrix ,Tau, nb_tiles;kend=kswitch)
        clear!(inputmatrix)
    end
    if (kswitch>0)
        Agpu = KernelAbstractions.allocate(backend, T, min(n,kswitch*TILESIZE), min(n,kswitch*TILESIZE))
        copyto!(Agpu,A[max(n-kswitch*TILESIZE+1,1):n,max(n-kswitch*TILESIZE+1,1):n])
        f(Agpu, Tau,min(kswitch,nb_tiles))
        KernelAbstractions.synchronize(backend)
        temp=zeros(T,min(kswitch*TILESIZE,n),min(kswitch*TILESIZE,n))
        copyto!(temp,Agpu)
        copyto!(view(A,max(n-kswitch*TILESIZE+1,1):n,max(n-kswitch*TILESIZE+1,1):n), temp)  
    end
    return A  
end

OOC_QR!(A::Matrix; kswitch::Int=256, tilesinmem::Int=max(floor(Int,kswitch^2/4)+1,2)) = OOC_alg!(A, mygeqrf!, kswitch, tilesinmem)
OOC_Bidiag!(A::Matrix; kswitch::Int=256, tilesinmem::Int=max(floor(Int,kswitch^2/4)+1,2)) = OOC_alg!(A, myblockdiag!, kswitch, tilesinmem)
OOC_SVD!(A::Matrix; kswitch::Int=256, tilesinmem::Int=max(floor(Int,kswitch^2/4)+1,2)) = banddiagsvd(OOC_Bidiag!(A,kswitch=kswitch, tilesinmem=tilesinmem))


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
    n=size(A,1)
    d=Float64.((Array(A[1:n+1:end])))
    e=Float64.((Array(A[n+1:n+1:end])))
    return T.(LAPACK.bdsdc!('U', 'N', d,e)[1])
   
end

function bidiagsvd(A::AbstractGPUMatrix{T}) where T
    n=size(A,1)
    d=Float64.((Array(A[1:n+1:end])))
    e=Float64.((Array(A[n+1:n+1:end])))
    return T.(LAPACK.bdsdc!('U', 'N', d,e)[1])
end

function bidiag(A::AbstractGPUMatrix{T}) where T
    n=size(A,1)
    d=Float64.((Array(A[1:n+1:end])))
    e=Float64.((Array(A[n+1:n+1:end])))
    return T.(LAPACK.bdsdc!('U', 'N', d,e)[1])
end


function mygesvd!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),TILESIZE,nbtiles)
    myblockdiag!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    out=banddiagsvd(A)
    unsafe_free!(Tau)
    return out
end


function myblockdiag!(A::AbstractGPUMatrix)
    nbtiles=Int(size(A,1)/TILESIZE)
    Tau=KernelAbstractions.zeros(get_backend(A),eltype(A),TILESIZE,nbtiles)
    myblockdiag!(A,Tau,nbtiles)
    KernelAbstractions.synchronize(get_backend(A))
    unsafe_free!(Tau)
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

#=
function QRandmult!(A::AnyGPUMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int, streams::Vector{streamtype};LQ::Bool=false)  where {T}
    eventsrecorded= Array{eventtype}(undef, nbtiles-k+2)
    currenteventidx=0
    
    setstream!(streams[1])
    QR1!(A,Tau, k;koffset=Int(LQ),singlerow=false)
    currenteventidx+=1
    eventsrecorded[currenteventidx]=event(streams[1])
    Qtapply1_par!(A, Tau, k; koffset=Int(LQ), singlerow=false)
    triu!(get_tileview(A, k+Int(LQ),k))

    
    if (k+1+Int(LQ) <= nbtiles)
        setstream!(streams[2])
        synchronize(eventsrecorded[currenteventidx])
        QR2!(A,Tau, k, k+1+Int(LQ); koffset=Int(LQ), singlerow=false)
        currenteventidx+=1
        eventsrecorded[currenteventidx]=event(streams[2])
    end
    

        for row in k+1+Int(LQ):(nbtiles-1)
            setstream!(streams[1])
            synchronize(eventsrecorded[currenteventidx])
            Qtapply2_par!(A,Tau, k,row; koffset=Int(LQ), singlerow=false)

            setstream!(streams[2])
            QR2!(A,Tau, k, row+1; koffset=Int(LQ), singlerow=false)
            currenteventidx+=1
            eventsrecorded[currenteventidx]=event(streams[2])

        end
    
    if (k+1+Int(LQ) <= nbtiles)
        synchronize(eventsrecorded[currenteventidx])
        Qtapply2_par!(A,Tau, k,nbtiles; koffset=Int(LQ), singlerow=false)
    end

    fill!(get_rowview(A',k, k+1+Int(LQ)),0)
end
=#

function QRandmult!(A::LargeTiledMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int, nbtiles::Int;LQ::Bool=false) where {T}
    nbcolgroups=(ceil(Int,(nbtiles-k)/(A.tilesinmem-1)))
    colgroupsize= ceil(Int,(nbtiles-k)/nbcolgroups)
    resizecache(A,colgroupsize+1)
    begincol=k+1
    endcol=min(k+colgroupsize,nbtiles)
    colinmem= A.tilesinmem-min(nbtiles-k, colgroupsize)
    while begincol<=nbtiles
        addtocache(A,endcol-begincol+2)

        setfirst!(A,k, LQ ,begincol,endcol,colinmem )
        QRandmulQt1!(A, Tau,k;LQ=LQ,begincol=begincol,endcol=endcol, colinmem=colinmem)
        for row in k+1+Int(LQ):nbtiles
            QRandmulQt2!(A, Tau,k,row;LQ=LQ,begincol=begincol,endcol=endcol, colinmem=colinmem)
        end
        getlast!(A, k,LQ, begincol,endcol, colinmem)

        begincol=endcol+1
        endcol=min(endcol+colgroupsize,nbtiles)
    end
end


@inline function QRandmulQt1!(A::LargeTiledMatrix{T}, Tau::AbstractGPUMatrix{T}, k::Int; LQ::Bool=false,  begincol::Int=k+1, endcol::Int=A.nb_tiles, colinmem::Int=k) where {T}
    push!(A.TileRows,  A.TileRows[2])
    begincol==k+1 && QR1!(A.TileRows[1],Tau, k;singlerow=true,colinmem=colinmem)
    Qtapply1_par!(A.TileRows[1], Tau, k;  singlerow=true,colinmem=colinmem)
    endcol==A.nb_tiles && triu!(get_tileview(A.TileRows[1], 1,colinmem))
    #have kernels run while doing data movements

    (k+Int(LQ)<A.nb_tiles) &&  set_tilerow(A, k, k+1+Int(LQ),  2, colbegin=begincol,endcol=endcol,colinmem=colinmem)

    
    KernelAbstractions.synchronize(A.backend) #only sync all here
    deleteat!(A.TileRows,2)
end

@inline function QRandmulQt2!(A::LargeTiledMatrix,Tau::AbstractGPUMatrix{T},k::Int, row::Int ; LQ::Bool=false, begincol::Int=k+1, endcol::Int=A.nb_tiles, colinmem::Int=k) where {T}
    
    
    begincol==k+1 && QR2!(A.TileRows[1],Tau, k, row; koffset=Int(LQ), singlerow=true, A2=A.TileRows[4],colinmem=colinmem)
    Qtapply2_par!(A.TileRows[1],Tau, k,row; koffset=Int(LQ), singlerow=true, A2=A.TileRows[4],colinmem=colinmem)
    endcol==A.nb_tiles && fill!(get_tileview(A.TileRows[4], 1,colinmem),0)
    #have kernels run while doing data movements

    (row<A.nb_tiles) && set_tilerow(A, k, row+1,  2,  colbegin=begincol,endcol=endcol,colinmem=colinmem)
    (row>k+1+Int(LQ)) && (get_tilerow(A,k,row-1 , 3 , colbegin=begincol,endcol=endcol,colinmem=colinmem)) 

    push!(A.TileRows,  A.TileRows[2])
    KernelAbstractions.synchronize(A.backend) #only sync all here
    deleteat!(A.TileRows,2)
end



function setfirst!(A::LargeTiledMatrix,k2::Int,  LQ::Bool , begincol::Int,endcol::Int, colinmem::Int)
    set_tilerow(A, k2, k2+Int(LQ),1,colbegin=begincol,endcol=endcol,colinmem=colinmem )

end

function getlast!(A::LargeTiledMatrix, k::Int, prevLQ::Bool, begincol::Int,endcol::Int, colinmem::Int)
    (k<A.nb_tiles) &&  get_tilerow(A ,k,A.nb_tiles , 3, colbegin=begincol,endcol=endcol,colinmem=colinmem )
    get_tilerow(A, k, k+Int(prevLQ),  1, colbegin=begincol,endcol=endcol,colinmem=colinmem)
end


