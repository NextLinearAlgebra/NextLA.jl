
using KernelAbstractions.Extras: @unroll
const SUBTILEFACTOR = (BRDMULSIZE >= BRDWIDTH ? 1 : Int(BRDWIDTH/BRDMULSIZE))

@inline packedrowidx(rowidx::Int,colidx::Int, packed::Bool) = (packed ? (2BW+1+(rowidx-colidx),colidx) : (rowidx,colidx)) 

struct PackedBandGPUMatrix{T} 
    data::AbstractGPUMatrix{T}     # diagonals plus bufferspace above and below
    bw::Int
    function PackedBandGPUMatrix{T}(d::AbstractGPUMatrix{T},bw::Int) where {T}
        new{T}(d,bw)
    end
end
function PackedBandGPUMatrix(input::AbstractGPUMatrix{T},bw::Int) where {T}
    n=size(input,1)
    d=KernelAbstractions.zeros(backend,T,bw*3+1,size(input,2))
    for i in 0:bw
        d[2bw+1-i,i+1:(end)].=input[i*n+1:n+1:end]
    end
    return PackedBandGPUMatrix{T}(d,bw)
end
function PackedBandGPUMatrix(input::Matrix{T},workspace::AbstractGPUMatrix{T},bw::Int) where {T}
    n=size(input,1)
    workspace.=0
    for i in 0:bw
        currdiag= @view workspace[2bw+1-i,i+1:size(workspace,2)]
        copyto!(currdiag, input[i*n+1:n+1:n*n])
    end
    return PackedBandGPUMatrix{T}(workspace,bw)
end


function Matrix_frombidiag!(output::AbstractGPUMatrix{T},input::PackedBandGPUMatrix{T}) where T
    output.=0
    n=size(output,1)
    bw=input.bw
    output[1:n+1:end].=input.data[2bw+1,1:(end)]
    output[n+1:n+1:end].=input.data[2bw,2:(end)]
    return output
end

@inline function Arrays_frombidiag!(input::PackedBandGPUMatrix{T}) where T
    n=size(input.data,2)
    bw=input.bw
    temp_gpu=KernelAbstractions.zeros(backend,T,n)
    temp_gpu .= view(input.data,2bw+1,1:n)
    d1 = zeros(T,n)
    copyto!(d1, temp_gpu)
    temp_gpu=KernelAbstractions.zeros(backend,T,n-1)
    d2 = zeros(T,n-1)
    temp_gpu .= view(input.data,2bw,2:n)
    copyto!(d2, temp_gpu)
    return d1,d2
end


function brd4!(A::AnyGPUMatrix{T}, noblocks::Int,bwiter::Int,packed::Bool) where T 
    CURRTILE=min(BRDMULSIZE, BRDWIDTH*(bwiter+1))
     brdkernel_large_v2!(backend, (CURRTILE))(A,size(A,2), 0, bwiter, cld(noblocks,MAXBLOCKS), packed, ndrange=(CURRTILE*min(noblocks,MAXBLOCKS)))
        brdkernel_large_v2!(backend, (CURRTILE))(A,size(A,2), 1, bwiter, cld(noblocks,MAXBLOCKS), packed, ndrange=(CURRTILE*min(noblocks,MAXBLOCKS)))
        brdkernel_large_v2!(backend, (CURRTILE))(A,size(A,2), 2, bwiter, cld(noblocks,MAXBLOCKS), packed, ndrange=(CURRTILE*min(noblocks,MAXBLOCKS)))
end

function mygbbrd!(A::AnyGPUMatrix{T}) where T 
    bw=BW
    n=size(A,1)
    for bwiter in Int(bw/BRDWIDTH):-1:1
        for k in 1:(n-1)
            brd4!(view(A,k:n,k:n),min(k,1+cld((n-k), (3BRDWIDTH*bwiter-1))),bwiter,false)
        end
    end
    return A
end 

function mygbbrd_packed!(A::AbstractGPUMatrix{T}) where T 
    bw=BW
    n=size(A,1)
    Apacked=PackedBandGPUMatrix(A,BW)
    KernelAbstractions.synchronize(backend)
    for bwiter in Int(bw/BRDWIDTH):-1:1
        for k in 1:(n-1)
            brd4!(view(Apacked.data,:,k:n),min(k,1+cld((n-k), (3BRDWIDTH*bwiter-1))),bwiter,true)
        end
    end
    KernelAbstractions.synchronize(backend)
    return Arrays_frombidiag!(Apacked)
end 

function mygbbrd_packed!(A::AbstractMatrix{T}, workspace::AbstractGPUMatrix{T}) where T 
    bw=BW
    n=size(A,1)
    Apacked=PackedBandGPUMatrix(A,workspace,BW)
    KernelAbstractions.synchronize(backend)
    for bwiter in Int(bw/BRDWIDTH):-1:1
        for k in 1:(n-1)
            brd4!(view(Apacked.data,:,k:n),min(k,1+cld((n-k), (3BRDWIDTH*bwiter-1))),bwiter,true)
        end
    end
    KernelAbstractions.synchronize(backend)
    return Arrays_frombidiag!(Apacked)
end 

function mygbbrd_packed_nocomm!(A::AbstractGPUMatrix{T}) where T 
    bw=BW
    n=size(A,2)
    Apacked=PackedBandGPUMatrix{T}(A,bw)
    for bwiter in Int(bw/BRDWIDTH):-1:1
        for k in 1:(n-1)
            brd4!(view(Apacked.data,:,k:n),min(k,1+cld((n-k), (3BRDWIDTH*bwiter-1))),bwiter,true)
        end
    end
    return Apacked.data
end 


@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel_large_v2!(input, nbrows::Int, offset::Int,bwiter::Int, nbblocks::Int, packed::Bool)
    i = @index(Local, Linear)
    g = @index(Group, Linear)
    tilecol = @private eltype(input) (BRDWIDTH+1)
    tilecol_cache = @localmem eltype(input) (BRDWIDTH+1)

    for nbrun in 0:(nbblocks-1)
        
        rowidx= -BRDWIDTH+(bwiter==1)+offset*BRDWIDTH*bwiter +(g-1+nbrun*MAXBLOCKS)*(3BRDWIDTH*bwiter-1)
        colidx= rowidx + BRDWIDTH *bwiter
        fullblock= (g>1||nbrun>0|| offset>0 || i>BRDWIDTH)
        
        (i==1) && (tilecol_cache[BRDWIDTH+1]=zero(eltype(input)))
        tilecol[BRDWIDTH+1]=zero(eltype(input))
        @synchronize
            @unroll for dolq in 0:1
                currrowidx=rowidx+BRDWIDTH*bwiter*dolq
                currcolidx=colidx
            
                if (i<=BRDWIDTH)
                    @unroll for k in 1:(SUBTILEFACTOR)
                        idx_x = dolq==1 ? k+SUBTILEFACTOR*(i-1) : 1
                        idx_y = dolq==1 ? 1 : k+SUBTILEFACTOR*(i-1)
                        row= (fullblock ? currrowidx+idx_x : 1 )
                        col= currcolidx+idx_y
                        tilecol_cache[k+SUBTILEFACTOR*(i-1)] =  (row <=nbrows && col <=nbrows) ? 
                                input[packedrowidx(row,col,packed)...] : zero(eltype(input))
                        if (row <=nbrows && col <=nbrows)
                            input[packedrowidx(row,col,packed)...] = zero(eltype(input))
                        end

                    end
                end
                if (i==1 && bwiter>1)
                    idx_x = dolq==1 ? BRDWIDTH+1 : 1
                    idx_y = dolq==1 ? 1 : BRDWIDTH+1
                    row =(fullblock ? currrowidx+idx_x : 1 )
                    col = currcolidx+idx_y
                    tilecol_cache[BRDWIDTH+1]= ( row <=nbrows && col <=nbrows) ? 
                        input[packedrowidx(row,col,packed)...] : zero(eltype(input))
                    if (row <=nbrows && col <=nbrows)
                            input[packedrowidx(row,col,packed)...] = zero(eltype(input))
                        end
                end

                @synchronize
                
                
                tmpsumiter = mulvecvec_nosplit(tilecol_cache, tilecol_cache)
                pivotel=tilecol_cache[1]
                @synchronize
                
                newvalue, execiter,factor = calc_factor(pivotel, tmpsumiter )
                if (i==1)
                            row= (fullblock ? currrowidx+1 : 1 )
                            col= currcolidx+1
                        if ( row<=nbrows && col<=nbrows)
                            input[packedrowidx(row,col,packed)...]  =  pivotel-(execiter ? factor : 0)
                        end
                    execiter && (tilecol_cache[1]=newvalue)
                end

                @synchronize

                for muliter in 0:21
                    if (((1+bwiter)*BRDWIDTH)>=(BRDMULSIZE*(muliter)+i) && ((fullblock && (i>1|| bwiter>1 ||muliter>0)) || muliter>=SUBTILEFACTOR))
                        
                        currrowidx=rowidx+BRDWIDTH*bwiter*dolq+(1-dolq)*(BRDMULSIZE*muliter+Int(bwiter>1))
                        currcolidx=colidx+(dolq)*(BRDMULSIZE*muliter+Int(bwiter>1))   
                        
                        @unroll for j in 1:BRDWIDTH
                            idx_x = dolq==1 ? j : i
                            idx_y = dolq==1 ? i : j
                            tilecol[j] = (currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows) ? 
                                    input[packedrowidx(currrowidx+idx_x,currcolidx+idx_y,packed)...] : zero(eltype(input))
                        end
                        idx_x = dolq==1 ? BRDWIDTH+1 : i
                        idx_y = dolq==1 ? i : BRDWIDTH+1
                        tilecol[BRDWIDTH+1] = (bwiter>1 && currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows ) ?
                            input[ packedrowidx(currrowidx+idx_x,currcolidx+idx_y,packed)...] : zero(eltype(input))

                        tmp_sum = mulvecvec_nosplit(tilecol, tilecol_cache)
                        tilecol1=tilecol[1]
                        factor = calc_factor2(pivotel, tmpsumiter, tmp_sum, tilecol1,newvalue)
                        updatevector(tilecol , tilecol_cache, factor, execiter,newvalue)

                        
                        @unroll for j in 1:BRDWIDTH
                            idx_x = dolq==1 ? j : i
                            idx_y = dolq==1 ? i : j
                            if (currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows)
                                input[packedrowidx(currrowidx+idx_x,currcolidx+idx_y,packed)...] =tilecol[j] 
                            end
                        end
                        idx_x = dolq==1 ? BRDWIDTH+1 : i
                        idx_y = dolq==1 ? i : BRDWIDTH+1
                        if (bwiter>1 && currrowidx+idx_x<=nbrows && currcolidx+idx_y<=nbrows)
                                input[packedrowidx(currrowidx+idx_x,currcolidx+idx_y,packed)...] = tilecol[BRDWIDTH+1]
                        end
                        
                    end
                
                end
            
            
            fullblock=true
            @synchronize
        end
        end
    
end


@inline function mulvecvec_nosplit(vec1, vec2 )
    tmp_sum = zero(eltype(vec1))
    @unroll for j in 2:BRDWIDTH+1
        tmp_sum+=  vec1[j]*vec2[j]
    end
    return tmp_sum
end



@inline function calc_factor(u1::T, unorm::T) where {T<:Number}
    execiter=false
    newvalue = u1 + (u1<0 ? -1 : 1)  *sqrt(u1*u1+unorm)
    factor = (unorm +newvalue*u1)*2/ (unorm/newvalue + newvalue)
    execiter = !(abs(unorm)<2floatmin(T)) 
    if (abs(newvalue)<2floatmin(T))
        newvalue=2u1
        factor = unorm/u1 +2u1
        execiter = execiter && !(abs(u1)<2floatmin(T))
    end
    execiter=true
    return newvalue, execiter,factor
end

@inline function calc_factor2( u1::T, unorm::T, uv::T, v1::T, newvalue::T) where {T<:Number}
    newvalue = u1 + (u1<0 ? -1 : 1)  *sqrt(u1*u1+unorm)
    factor = (uv +newvalue*v1)*2/ (unorm/newvalue + newvalue)
    if (abs(newvalue)<2floatmin(T))
        newvalue=2u1
        factor = uv/u1 +2v1
    end
    return factor
end

@inline function updatevector(vec1, vec2, factor::Number,execute::Bool,newvalue)
    if (execute)
        @unroll for j in 1:BRDWIDTH+1
            vec1[j]-=  factor* (vec2[j]/newvalue)
        end
    end
end
