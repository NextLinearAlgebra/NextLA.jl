
using KernelAbstractions.Extras: @unroll
const BRDSPLITFACTOR = Int(TILESIZE/BRDSPLIT)

function brd1!(A::AnyGPUMatrix{T}, noblocks) where T 
    brdkernel!(backend, (BRDSPLIT, TILESIZE,2))(A,size(A,1), false, ndrange=(BRDSPLIT*noblocks,TILESIZE,2))
    brdkernel!(backend, (BRDSPLIT, TILESIZE,2))(A,size(A,1), true, ndrange=(BRDSPLIT*noblocks,TILESIZE,2))
end
function brd2!(A::AnyGPUMatrix{T}, noblocks) where T 
    brdkernel_lowmem!(backend, (BRDSPLIT, TILESIZE))(A,size(A,1), false, ndrange=(BRDSPLIT*noblocks,TILESIZE))
    brdkernel_lowmem!(backend, (BRDSPLIT, TILESIZE))(A,size(A,1), true, ndrange=(BRDSPLIT*noblocks,TILESIZE))
end


function mygbbrd!(A::AnyGPUMatrix{T}) where T 
    n=size(A,1)
    for k in 1:(n-1)
        brd!(view(A,k:n,k:n),min(k,1+cld((n-k), (4TILESIZE-1))))
    end
end




@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel!(input, nbrows, secondsweep)
    k,i,l = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    cache = @localmem eltype(input) (TILESIZE+1, TILESIZE,2)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,TILESIZE,2)

    krange = (k-1)*BRDSPLITFACTOR.+(1:BRDSPLITFACTOR)
    idxcurr = l-1 
    idxiter=0
    fullblock= (g>1 || l==2 || secondsweep)
    
    if (l==1)
        if (fullblock) 
            loadblockintocache(view(cache,krange,i,1),input, k,i,rowidx,colidx, nbrows)
        else
            loadonlyfirstrow(view(cache,i,krange,1), input, k,i,colidx,nbrows)
        end
    end
    @synchronize

    @unroll for idx in 0:1
        @unroll for dolq in 0:1
        idxiter= idxiter % 2 +1
        idxcurr = idxcurr % 2 +1
        idx_x = dolq==0 ? 1 : i
        idx_y = dolq==0 ? i : 1
        
        if (l==2)
            loadblockintocache(view(cache,krange,i,idxcurr),input, k,i,rowidx+TILESIZE*(1+idx),colidx+(idx+dolq)*TILESIZE, nbrows)
        end

        @synchronize
        mulvecvec_split_exclfirst((dolq==0) ? view(cache,1,krange,idxiter) : view(cache,krange,1,idxiter),
                                (dolq==0) ? view(cache,i,krange,idxcurr) : view(cache,krange,i,idxcurr),view(cache2,:,i,idxcurr), k,0)
 
        @synchronize
        
        tmpsumiter = accumulatevals(view(cache2,:,1,idxiter))
        tmp_sum = accumulatevals(view(cache2,:,i,idxcurr))
        
        newvalue, factor, execiter = calc_factor(cache[1,1,idxiter], tmpsumiter, tmp_sum,  cache[idx_y,idx_x,idxcurr] )
        @synchronize
        updatevector((dolq==0) ? view(cache, i,krange,idxcurr) : view(cache, krange,i,idxcurr) , 
                            (dolq==0) ? view(cache,1,krange,idxiter) : view(cache,krange,1,idxiter) , factor,k,execiter && fullblock && (i>1 || l==2),newvalue)
        
        @synchronize
        
        (k==1 && i>1 &&l==1) && (cache[idx_x,idx_y,idxiter]= zero(eltype(input)))
        (k==1 && execiter ) &&  (cache[idx_y,idx_x,idxcurr]-=factor)

        @synchronize

        if (l==1) 
            if (fullblock)
                sendblockbacktomem(input, view(cache,krange,i,idxiter), k, i, rowidx+TILESIZE*(idx+dolq), colidx+idx*TILESIZE, nbrows)
            else
                sendbackonlyfirstrow(view(cache,i,krange,1), input, k,i,colidx,nbrows)
            end
        end
        fullblock=true
        @synchronize

    end
    end

    if (l==2)
        sendblockbacktomem(input, view(cache,krange,i,idxcurr), k, i, rowidx+2TILESIZE, colidx+2TILESIZE, nbrows)
    end

end


@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel_lowmem!(input, nbrows, secondsweep)
    k,i = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    cache = @localmem eltype(input) (TILESIZE+1, TILESIZE)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,TILESIZE)
    tilecol_cache = @localmem eltype(input) (TILESIZE)

    krange = (k-1)*BRDSPLITFACTOR.+(1:BRDSPLITFACTOR)
    fullblock= (g>1|| secondsweep)
    
        if (fullblock)
            loadblockintocache(view(cache, krange,i),input, k,i,rowidx,colidx, nbrows)
        else
            loadonlyfirstrow(view(cache,i,krange), input, k,i,colidx,nbrows)
        end

    @synchronize

    @unroll for idx in 0:1
        @unroll for dolq in 0:1
        idx_x = dolq==0 ? 1 : i
        idx_y = dolq==0 ? i : 1
            
        pivotel=cache[1,1]
        mulvecvec_split_exclfirst((dolq==0) ? view(cache,1,krange) : view(cache,krange,1),
                                (dolq==0) ? view(cache,i,krange) : view(cache,krange,i),view(cache2,:,i), k,0)
 
        @synchronize
        
        tmpsumiter = accumulatevals(view(cache2,:,1))
        tmp_sum = accumulatevals(view(cache2,:,i))

        newvalue, factor,execiter = calc_factor(cache[1,1], tmpsumiter, tmp_sum,cache[idx_y,idx_x] )
        @synchronize
        updatevector((dolq==0) ? view(cache, i,krange) : view(cache, krange,i) , 
                            (dolq==0) ? view(cache,1,krange) : view(cache,krange,1) , factor,k,execiter && fullblock && (i>1),newvalue)
        if (k+BRDSPLIT*(i-1) <= TILESIZE) 
            tilecol_cache[k+BRDSPLIT*(i-1)]= (dolq == 0) ?  cache[1,k+BRDSPLIT*(i-1)] : cache[k+BRDSPLIT*(i-1),1]
        end

        @synchronize

        (k==1 && i>1) && (cache[idx_x,idx_y]= zero(eltype(input)))
        (k==1 && execiter) &&  (cache[idx_y,idx_x]-=factor)
        
        @synchronize


        if (fullblock)
            sendblockbacktomem(input, view(cache,krange,i), k, i, rowidx+TILESIZE*(idx+dolq), colidx+idx*TILESIZE, nbrows)
        else
            sendbackonlyfirstrow(view(cache,i,krange), input, k,i,colidx,nbrows)
        end        
        (i==1 && k==1) &&  (tilecol_cache[1]=newvalue)

        @synchronize

        loadblockintocache(view(cache,krange,i),input, k,i,rowidx+TILESIZE*(1+idx),colidx+(idx+dolq)*TILESIZE, nbrows)

        @synchronize

        mulvecvec_split_exclfirst(view(tilecol_cache,krange),  (dolq==0) ? view(cache,i,krange) : view(cache,krange,i),view(cache2,:,i), k,1)

        @synchronize

        tmp_sum = accumulatevals(view(cache2,:,i))
        factor = calc_factor2(pivotel, tmpsumiter, tmp_sum,dolq==0 ? cache[i,1] : cache[1,i] ,newvalue)
        @synchronize
        updatevector((dolq==0) ? view(cache, i,krange) : view(cache, krange,i) , view(tilecol_cache,krange), factor,k, execiter,newvalue)
        @synchronize
        (k==1 && execiter) &&  (cache[idx_y,idx_x]-=factor)
        @synchronize

        fullblock=true
        @synchronize

    end
    end

    sendblockbacktomem(input, view(cache,krange,i), k, i, rowidx+TILESIZE*2, colidx+2*TILESIZE, nbrows)
end
#=
@kernel cpu=false inbounds=true unsafe_indices=false function brdkernel_large!(input, nbrows, secondsweep; packed::Bool=false)
    k,i = @index(Local, NTuple)
    g = @index(Group, Linear)
    rowidx= -TILESIZE+1+Int(secondsweep)*2TILESIZE +(g-1)*(4TILESIZE-1)
    colidx= rowidx + TILESIZE
    tilecol = @private eltype(input) (BRDSPLITFACTOR)
    cache2 = @localmem eltype(input) (BRDSPLIT+1,BRDSUBTILE)
    tilecol_cache = @localmem eltype(input) (TILESIZE)
    tilecol_first = @localmem eltype(input) (BRDSUBTILE)

    krange = (k-1)*BRDSPLITFACTOR.+(1:BRDSPLITFACTOR)
    fullblock= (g>1|| secondsweep)
    

    @unroll for idx in 0:1
        @unroll for dolq in 0:1
    
        if (fullblock)
            loadblockintocache(tilecol,input, k,i,rowidx+TILESIZE*(idx+dolq),colidx+TILESIZE*(idx), nbrows;doqr= (dolq==1))
        else
            loadonlyfirstrow(tilecol, input, k,i,colidx,nbrows)
        end

        if (i==1)
            @unroll for j in 1:BRDSPLITFACTOR
                tilecol_cache[j+BRDSPLITFACTOR*(k-1)]=  tilecol[j]
            end
        end
        (k==1) &&  (tilecol_first[i]=tilecol[1])
        
        @synchronize
        mulvecvec_split_exclfirst(view(tilecol_cache,krange), tilecol,view(cache2,:,i), k,0)
 
        @synchronize
        pivotel=tilecol_cache[1]
        tmpsumiter = accumulatevals(view(cache2,:,1))
        tmp_sum = accumulatevals(view(cache2,:,i))
        
        newvalue, factor,execiter = calc_factor(pivotel, tmpsumiter, tmp_sum, tilecol_first[i] )
        updatevector(tilecol , view(tilecol_cache,krange) , factor,k, execiter && fullblock && (i>1))

        if (i==1)  
                @unroll for j in (1+Int(k==1)):BRDSPLITFACTOR
                    tilecol[j] = zero(eltype(input))
                end
        end
        (k==1 && execiter) && (tilecol[1]-=factor*newvalue)
        

        if (fullblock)
            sendblockbacktomem(input, tilecol, k, i, rowidx+TILESIZE*(idx+dolq),colidx+TILESIZE*(idx), nbrows;doqr= (dolq==1))
        elseif (i==1)
            sendbackonlyfirstrow(tilecol, input, k,i,colidx,nbrows)
        end
        (i==1 && k==1) && (tilecol_cache[1]=newvalue)
        
        @synchronize
        @unroll for blockidx in 0:1
            if !fullblock
                fullblock=true
                continue
            end
            @unroll for muliter in (1-blockidx):(BRDTILESPERTILE-1)

                loadblockintocache(tilecol,input, k,i,rowidx+TILESIZE*(idx+dolq)+(1-dolq)*(BRDSUBTILE*muliter+blockidx*TILESIZE),colidx+TILESIZE*(idx)+(dolq)*(BRDSUBTILE*muliter+blockidx*TILESIZE), nbrows; doqr= (dolq==1))
                (k==1) && (tilecol_first[i]=tilecol[1])
                mulvecvec_split_exclfirst(view(tilecol_cache,krange),  tilecol,view(cache2,:,i), k,1)

                @synchronize

                tmp_sum = accumulatevals(view(cache2,:,i))
                factor = calc_factor2(pivotel, tmpsumiter, tmp_sum,tilecol_first[i] ,newvalue)
                updatevector(tilecol , view(tilecol_cache,krange) , factor,k, execiter)
                (k==1 && execiter) &&  (tilecol[1]-=factor*newvalue)

                sendblockbacktomem(input, tilecol, k, i, rowidx+TILESIZE*(idx+dolq)+(1-dolq)*(BRDSUBTILE*muliter+blockidx*TILESIZE),colidx+TILESIZE*(idx)+(dolq)*(BRDSUBTILE*muliter+blockidx*TILESIZE), nbrows;doqr= (dolq==1))
                
                @synchronize

            end
        end
        

    end
    end

    
end
=#
@inline function loadonlyfirstrow(cache, input, k,i,colidx,nbrows)
    @unroll for j in 1:BRDSPLITFACTOR
        cache[j] = (i==1 && colidx+(k-1)*BRDSPLITFACTOR+j <=nbrows) ? input[1,colidx+(k-1)*BRDSPLITFACTOR+j] :  zero(eltype(input))
    end
end

@inline function sendbackonlyfirstrow(cache, input, k,i,colidx,nbrows)
    @unroll for j in 1:BRDSPLITFACTOR
        if (i==1 && colidx+(k-1)*BRDSPLITFACTOR+j <=nbrows)
            input[1,colidx+(k-1)*BRDSPLITFACTOR+j] =cache[j]
        end
    end
end



@inline function loadblockintocache(cache, input, k::Int, i::Int, rowidx::Int, colidx::Int, nbrows::Int; doqr::Bool=true)
        @unroll for j in 1:BRDSPLITFACTOR
            idx_x = doqr ? (k-1)*BRDSPLITFACTOR+j : i
            idx_y = doqr ? i : (k-1)*BRDSPLITFACTOR+j
            cache[j] = 
                    (rowidx+idx_x<=nbrows && colidx+idx_y<=nbrows) ? 
                    input[rowidx+idx_x,colidx+idx_y] : zero(eltype(input))
        end
end
@inline function sendblockbacktomem(input, cache, k::Int, i::Int, rowidx::Int, colidx::Int, nbrows::Int; doqr::Bool=true)
        @unroll for j in 1:BRDSPLITFACTOR
            idx_x = doqr ? (k-1)*BRDSPLITFACTOR+j : i
            idx_y = doqr ? i : (k-1)*BRDSPLITFACTOR+j
            if (rowidx+idx_x<=nbrows && colidx+idx_y<=nbrows)
                input[rowidx+idx_x,colidx+idx_y] = cache[j]
            end
        end
end


@inline function mulvecvec_split_exclfirst(vec1, vec2,cache2, k::Int, nonfirst::Int )
    tmp_sum = zero(eltype(vec1))
    @unroll for j in 1:BRDSPLITFACTOR
        tmp_sum+= ((k+j+nonfirst==2) ? zero(eltype(vec1)) : vec1[j]*vec2[j])
    end
    cache2[k]=tmp_sum
end




@inline function calc_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    newvalue = u1 + (u1<0 ? -1 : 1) *sqrt(u1*u1+unorm)
    execiter = !(abs(newvalue)<2floatmin(T) || abs(unorm)<2floatmin(T)) 
    factor = (uv +newvalue*v1)*2/ (unorm/newvalue + newvalue)
    return newvalue, factor, execiter
end

@inline function calc_factor2( u1::T, unorm::T, uv::T, v1::T, newvalue::T) where {T<:Number}

    factor = 2uv/ (unorm/newvalue + newvalue)
    return factor
end

@inline function updatevector(vec1, vec2, factor::Number,k::Int,execute::Bool,newvalue)
    if (execute)
        @unroll for j in 1:BRDSPLITFACTOR
            vec1[j]-=((k+j==2) ?  zero(eltype(vec1)) : factor* (vec2[j]/newvalue))
        end
    end
end




@inline function accumulatevals(toacc)
    res = zero(eltype(toacc))
        @unroll for j = 1:BRDSPLIT
            res+= toacc[j]
        end
    return res
end


